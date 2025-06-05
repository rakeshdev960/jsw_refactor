import streamlit as st
import cv2
import numpy as np
import time
import math
import requests
from collections import defaultdict
from ultralytics import YOLO
from tracker import Tracker

# Set page config
st.set_page_config(
    page_title="JSW RTSP Cement Bag Detection",
    page_icon="📦",
    layout="wide"
)

# Flask API configuration
FLASK_API_URL = "http://localhost:5000"

# Title
st.title("JSW Cement Bag Detection - RTSP Camera Feed")
st.write("Process RTSP camera feeds to detect and count cement bags")

# Cache the models
@st.cache_resource
def load_models():
    # Use absolute path to avoid path issues
    cement_model = YOLO('F:/jsw20042025/best_cement_bags.pt')
    return cement_model

# Function to load clusters from Flask API
def load_clusters():
    try:
        response = requests.get(f"{FLASK_API_URL}/clusters")
        clusters = response.json()
        st.session_state.available_clusters = clusters
        return clusters
    except Exception as e:
        st.error(f"Error loading clusters: {str(e)}")
        return []

# Function to send data to inventory system
def send_to_inventory_system(cluster_name, in_count, out_count):
    """Send counting data to Flask backend's inventory system"""
    try:
        # First check if cluster exists
        response = requests.get(f"{FLASK_API_URL}/clusters")
        clusters = response.json()
        
        cluster_exists = False
        cluster_id = None
        
        for cluster in clusters:
            if cluster['name'] == cluster_name:
                cluster_exists = True
                cluster_id = cluster['id']
                break
        
        if cluster_exists:
            # Create movement records for IN and OUT
            movement_success = True
            
            # Record IN movement if any
            if in_count > 0:
                in_data = {"movement_type": "IN", "quantity": in_count}
                in_response = requests.post(f"{FLASK_API_URL}/clusters/{cluster_id}/movement", json=in_data)
                if in_response.status_code not in (200, 201):
                    movement_success = False
            
            # Record OUT movement if any
            if out_count > 0:
                out_data = {"movement_type": "OUT", "quantity": out_count}
                out_response = requests.post(f"{FLASK_API_URL}/clusters/{cluster_id}/movement", json=out_data)
                if out_response.status_code not in (200, 201):
                    movement_success = False
            
            if movement_success:
                return {"status": "success", "message": "Movement records created successfully"}, 200
            else:
                return {"error": "Failed to create movement records"}, 500
        else:
            # Create new cluster if it doesn't exist
            net_count = in_count - out_count
            data = {"name": cluster_name, "bag_count": net_count}
            response = requests.post(f"{FLASK_API_URL}/clusters", json=data)
            return response.json(), 201
    except Exception as e:
        st.error(f"Error connecting to inventory system: {str(e)}")
        return {"error": str(e)}, 500

# Define line colors
LINE_COLORS = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (128, 0, 0),    # Dark Blue
    (0, 128, 0),    # Dark Green
    (0, 0, 128)     # Dark Red
]

def draw_lines(frame, lines, colors):
    """Draw crossing lines and counts on frame"""
    frame_with_lines = frame.copy()
    for i, line in enumerate(lines):
        # Get the color for this line
        color = colors[i % len(colors)]
        # Draw the line
        cv2.line(frame_with_lines, line[0], line[1], color, 2)
        
        # Calculate midpoint of line to place text
        mid_x = (line[0][0] + line[1][0]) // 2
        mid_y = (line[0][1] + line[1][1]) // 2
        
        # Get counts for this line
        if i in st.session_state.crossing_stats:
            in_count = st.session_state.crossing_stats[i]['in']
            out_count = st.session_state.crossing_stats[i]['out']
            
            # Draw counts near the line
            cv2.putText(frame_with_lines, f"IN: {in_count}", (mid_x - 40, mid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame_with_lines, f"OUT: {out_count}", (mid_x - 40, mid_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Add totals at the top of the frame
    total_in = sum(stats['in'] for stats in st.session_state.crossing_stats.values())
    total_out = sum(stats['out'] for stats in st.session_state.crossing_stats.values())
    total_net = total_in - total_out
    
    # Add totals at the top of the frame
    cv2.putText(frame_with_lines, f"Total IN: {total_in}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame_with_lines, f"Total OUT: {total_out}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame_with_lines, f"NET Total: {total_net}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame_with_lines

def check_line_crossing(point, line):
    """Determine which side of a line a point is on using the cross product"""
    x, y = point
    (x1, y1), (x2, y2) = line
    # Calculate the cross product
    return (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)

def get_movement_direction(prev_pos, current_pos, line):
    """Determine direction of movement relative to a line"""
    prev_side = check_line_crossing(prev_pos, line)
    current_side = check_line_crossing(current_pos, line)
    
    # If the signs differ, the object has crossed the line
    if prev_side * current_side < 0:
        # Determine direction:
        # If moving from negative to positive -> IN
        # If moving from positive to negative -> OUT
        if prev_side < 0:
            return "IN"
        else:
            return "OUT"
    return None

# Initialize session state variables
if 'tracker' not in st.session_state:
    st.session_state.tracker = Tracker()
if 'crossing_stats' not in st.session_state:
    # Dictionary to hold line crossing statistics for each line
    st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
if 'last_positions' not in st.session_state:
    # Track previous positions of objects
    st.session_state.last_positions = {}
if 'available_clusters' not in st.session_state:
    st.session_state.available_clusters = []
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'lines' not in st.session_state:
    st.session_state.lines = []
if 'line_colors' not in st.session_state:
    st.session_state.line_colors = []
if 'processed_files' not in st.session_state:
    # Track processed files to prevent reprocessing
    st.session_state.processed_files = {}
if 'rtsp_connection_active' not in st.session_state:
    st.session_state.rtsp_connection_active = False
if 'rtsp_frame_count' not in st.session_state:
    st.session_state.rtsp_frame_count = 0
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

# Load model
cement_model = load_models()

# RTSP Camera Configuration
st.sidebar.title("RTSP Camera Configuration")
rtsp_base = st.sidebar.text_input("RTSP Base URL", value="rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels/")
channel_number = st.sidebar.text_input("Channel Number", value="101")

# Form full RTSP URL
rtsp_url = f"{rtsp_base}{channel_number}"

# Line Configuration
st.sidebar.title("Line Configuration")
num_lines = st.sidebar.number_input("Number of lines to draw", min_value=1, max_value=5, value=1)

if len(st.session_state.lines) != num_lines:
    # Initialize lines if the number has changed
    frame_width, frame_height = 640, 480  # Default size until we get a frame
    st.session_state.lines = []
    st.session_state.line_colors = []
    for i in range(num_lines):
        st.session_state.lines.append([(50, 100 + i*50), (550, 100 + i*50)])
        st.session_state.line_colors.append(LINE_COLORS[i % len(LINE_COLORS)])

# Cluster selection
st.sidebar.title("Inventory System")
if st.sidebar.button("Refresh Clusters"):
    clusters = load_clusters()
else:
    if not st.session_state.available_clusters:
        clusters = load_clusters()
    else:
        clusters = st.session_state.available_clusters

if clusters:
    cluster_options = {cluster['name']: cluster for cluster in clusters}
    selected = st.sidebar.selectbox("Select Cluster", options=list(cluster_options.keys()), index=0 if cluster_options else None)

    if selected and st.sidebar.button("Reset Cluster Count and History", help="This will set the bag count to 0 and delete all movement history"):
        cluster_id = cluster_options[selected]['id']
        try:
            response = requests.post(f"{FLASK_API_URL}/clusters/{cluster_id}/reset")
            if response.status_code == 200:
                st.sidebar.success(f"✅ Reset {selected} to 0 bags and cleared history")
                # Update the cluster in the session state
                new_clusters = load_clusters()
                # Force UI refresh
                st.rerun()
            else:
                st.sidebar.error(f"Failed to reset: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error resetting cluster: {str(e)}")

    if selected:
        # Store the selected cluster in session state
        st.session_state.selected_cluster = cluster_options[selected]
else:
    st.sidebar.warning("No clusters available. Connect to backend API or create a new cluster.")
    new_cluster_name = st.sidebar.text_input("New Cluster Name")
    if st.sidebar.button("Create Cluster") and new_cluster_name:
        try:
            response = requests.post(f"{FLASK_API_URL}/clusters", json={"name": new_cluster_name, "bag_count": 0})
            if response.status_code in (200, 201):
                st.sidebar.success(f"Created cluster '{new_cluster_name}'")
                # Reload clusters
                clusters = load_clusters()
                st.rerun()
            else:
                st.sidebar.error(f"Failed to create cluster: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error creating cluster: {str(e)}")

# Configure line coordinates
st.write("### Line Configuration")
col1, col2 = st.columns(2)

for i in range(num_lines):
    with st.expander(f"Line {i+1} Settings", expanded=i == 0):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Start Point")
            x1 = st.number_input(f"Start X {i+1}", min_value=0, max_value=1920, value=st.session_state.lines[i][0][0], key=f"x1_{i}")
            y1 = st.number_input(f"Start Y {i+1}", min_value=0, max_value=1080, value=st.session_state.lines[i][0][1], key=f"y1_{i}")
        with col2:
            st.write("End Point")
            x2 = st.number_input(f"End X {i+1}", min_value=0, max_value=1920, value=st.session_state.lines[i][1][0], key=f"x2_{i}")
            y2 = st.number_input(f"End Y {i+1}", min_value=0, max_value=1080, value=st.session_state.lines[i][1][1], key=f"y2_{i}")
        
        # Color selection
        color_idx = LINE_COLORS.index(st.session_state.line_colors[i]) if i < len(st.session_state.line_colors) else 0
        new_color_idx = st.selectbox(f"Line {i+1} Color", range(len(LINE_COLORS)), 
                                index=color_idx, format_func=lambda x: f"Color {x+1}", key=f"color_{i}")
        if i < len(st.session_state.line_colors):
            st.session_state.line_colors[i] = LINE_COLORS[new_color_idx]
        else:
            st.session_state.line_colors.append(LINE_COLORS[new_color_idx])
        
        st.session_state.lines[i] = [(x1, y1), (x2, y2)]

# Reset button
if st.button("Reset Crossing Statistics"):
    st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
    st.session_state.last_positions = {}
    st.session_state.rtsp_frame_count = 0
    st.success("✅ Crossing statistics have been reset")

def process_frame(frame, lines, colors):
    # Detect cement bags
    results = cement_model(frame, conf=0.2)[0]  # Lower confidence for better recall
    
    # Extract detection results
    boxes = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        if score > 0.2:  # Additional confidence threshold check
            boxes.append([int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)])
    
    # Track objects
    bbox_idx = st.session_state.tracker.update(boxes)
    
    # Current positions of tracked objects
    current_positions = {}
    
    # Draw boxes and IDs
    for obj in bbox_idx:
        x, y, w, h, id = obj
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Store current position
        current_positions[id] = (center_x, center_y)
        
        # Draw bounding box and ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check if object has crossed any line
        if id in st.session_state.last_positions:
            prev_pos = st.session_state.last_positions[id]
            
            # Check each line
            for i, line in enumerate(lines):
                # Check for line crossing
                direction = get_movement_direction(prev_pos, (center_x, center_y), line)
                if direction:
                    # Update count for this line
                    st.session_state.crossing_stats[i][direction.lower()] += 1
                    # Draw crossing point with direction
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, direction, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 0, 255), 2)
    
    # Update last positions for next frame
    st.session_state.last_positions = current_positions
    
    # Draw lines and counts
    frame_with_lines = draw_lines(frame, lines, colors)
    
    return frame_with_lines

# Start/Stop RTSP Stream
if st.session_state.selected_cluster:
    if st.button("Start RTSP Stream" if not st.session_state.rtsp_connection_active else "Stop RTSP Stream"):
        st.session_state.rtsp_connection_active = not st.session_state.rtsp_connection_active
        if st.session_state.rtsp_connection_active:
            st.session_state.rtsp_frame_count = 0
            st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
            st.session_state.last_positions = {}
else:
    st.warning("⚠️ Please select a cluster before starting the RTSP stream")

# Display RTSP stream
if st.session_state.rtsp_connection_active:
    # Create a placeholder for the video
    video_placeholder = st.empty()
    
    # Create a placeholder for stats
    stats_placeholder = st.empty()
    
    # Create placeholders for metrics
    col1, col2, col3 = st.columns(3)
    in_metric = col1.empty()
    out_metric = col2.empty()
    net_metric = col3.empty()
    
    # Initialize video capture
    try:
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            st.error(f"❌ Failed to open RTSP stream: {rtsp_url}")
            st.session_state.rtsp_connection_active = False
        else:
            st.success(f"✅ Connected to RTSP stream: {rtsp_url}")
            
            # Display "Processing..." message
            processing_text = st.empty()
            processing_text.text("Processing RTSP stream... (click 'Stop RTSP Stream' to end)")
            
            # Process frames in a loop while connection is active
            while st.session_state.rtsp_connection_active:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Error reading from RTSP stream. Reconnecting...")
                    cap.release()
                    time.sleep(2)  # Wait before reconnecting
                    cap = cv2.VideoCapture(rtsp_url)
                    continue
                
                # Process the frame
                processed_frame = process_frame(frame, st.session_state.lines, st.session_state.line_colors)
                
                # Count frames
                st.session_state.rtsp_frame_count += 1
                
                # Convert to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display the processed frame
                video_placeholder.image(processed_frame_rgb, caption=f"RTSP Stream - {rtsp_url}", use_container_width=True)
                
                # Update metrics every 30 frames
                if st.session_state.rtsp_frame_count % 30 == 0:
                    # Calculate totals
                    total_in = sum(stats['in'] for stats in st.session_state.crossing_stats.values())
                    total_out = sum(stats['out'] for stats in st.session_state.crossing_stats.values())
                    net_total = total_in - total_out
                    
                    # Update metrics
                    in_metric.metric("Total IN", total_in, delta=None, delta_color="normal")
                    out_metric.metric("Total OUT", total_out, delta=None, delta_color="normal")
                    net_metric.metric("NET Change", net_total, delta=None, delta_color="normal")
                
                # Update inventory every minute if there are changes
                current_time = time.time()
                if current_time - st.session_state.last_update_time >= 60:  # Update every 60 seconds
                    total_in = sum(stats['in'] for stats in st.session_state.crossing_stats.values())
                    total_out = sum(stats['out'] for stats in st.session_state.crossing_stats.values())
                    
                    if total_in > 0 or total_out > 0:
                        # Update inventory
                        cluster_info = st.session_state.selected_cluster
                        response, status_code = send_to_inventory_system(cluster_info['name'], total_in, total_out)
                        
                        if status_code in (200, 201):
                            stats_placeholder.success(f"✅ Updated inventory at {time.strftime('%H:%M:%S')}")
                        else:
                            stats_placeholder.error(f"❌ Failed to update inventory: {response.get('error', 'Unknown error')}")
                        
                        # Reset counters after update
                        st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
                    
                    # Update last update time
                    st.session_state.last_update_time = current_time
                
                # Add a small delay to prevent UI freezing
                time.sleep(0.01)
            
            # Release resources when stopped
            cap.release()
            processing_text.text("RTSP stream stopped.")
    
    except Exception as e:
        st.error(f"❌ Error processing RTSP stream: {str(e)}")
        st.session_state.rtsp_connection_active = False
else:
    st.write("Click 'Start RTSP Stream' to begin processing camera feed")

# Add information section
with st.expander("Information", expanded=False):
    st.write("""
    ### RTSP Cement Bag Detection
    
    This application detects cement bags in real-time from RTSP camera feeds and tracks when they cross defined lines. 
    It updates the inventory system in the backend, which maintains records of bag movements.
    
    #### How to use:
    1. Select a cluster for inventory tracking
    2. Configure the detection lines 
    3. Click 'Start RTSP Stream' to begin processing
    4. Statistics will update automatically
    5. Inventory will update every minute if changes are detected
    
    #### RTSP URL Format:
    - Base URL: rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels/
    - Channel: 101-701
    """)

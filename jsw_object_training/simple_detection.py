import streamlit as st
import cv2
import numpy as np
import time
import math
import requests
import tempfile
import os
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from tracker import Tracker

# Set page config
st.set_page_config(
    page_title="JSW Cement Bag Detection",
    page_icon="📦",
    layout="wide"
)

# Flask API configuration
FLASK_API_URL = "http://localhost:5000"

# Title
st.title("JSW Cement Bag Detection System")
st.write("Process RTSP camera feeds or upload videos to detect and count cement bags")

# Cache the models
import torch

@st.cache_resource
def load_models(device=None):
    """Load YOLO model with error handling for CUDA issues"""
    try:
        # Try to load model on specified device (or default)
        if device:
            cement_model = YOLO('F:/jsw20042025/best_cement_bags.pt', device=device)
        else:
            cement_model = YOLO('F:/jsw20042025/best_cement_bags.pt')
            
        # Test the model with a small tensor to catch CUDA errors early
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.no_grad():
            cement_model(test_img)
            
        # If successful, return the model
        return cement_model
    except Exception as e:
        if device != 'cpu':
            # If error occurs and not already trying CPU, fall back to CPU
            st.warning(f"GPU error: {str(e)}. Falling back to CPU processing.")
            return load_models(device='cpu')
        else:
            # If error occurs even on CPU, raise it
            st.error(f"Error loading model: {str(e)}")
            raise

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

def check_line_crossing(point, line):
    """Determine which side of a line a point is on using the cross product"""
    x, y = point
    (x1, y1), (x2, y2) = line
    # Calculate the cross product
    return (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)

def get_movement_direction(prev_pos, current_pos, line):
    """Determine direction of movement relative to a line
    
    Always count right-to-left as OUT and left-to-right as IN
    regardless of line orientation.
    """
    prev_side = check_line_crossing(prev_pos, line)
    current_side = check_line_crossing(current_pos, line)
    
    # If the signs differ, the object has crossed the line
    if prev_side * current_side < 0:
        # Get horizontal movement direction
        prev_x, _ = prev_pos
        current_x, _ = current_pos
        
        # If moving right to left (decreasing x)
        if current_x < prev_x:
            return "OUT"  # Right to left is always OUT
        else:
            return "IN"   # Left to right is always IN
    return None

def draw_lines(frame, lines, colors):
    """Draw counting lines and counts on frame"""
    frame_with_lines = frame.copy()
    for i, line in enumerate(lines):
        if i < len(colors):
            color = colors[i]
        else:
            color = colors[i % len(colors)]
        
        # Get line endpoints
        pt1, pt2 = line
        
        # Draw line
        cv2.line(frame_with_lines, pt1, pt2, color, 2)
        
        # Add line number
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        cv2.putText(frame_with_lines, f"Line {i+1}", (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # If we have crossing stats, show them
        if 'crossing_stats' in st.session_state and i in st.session_state.crossing_stats:
            in_count = st.session_state.crossing_stats[i]['in']
            out_count = st.session_state.crossing_stats[i]['out']
            net_count = in_count - out_count
            
            # Display counts near line
            stats_text = f"IN: {in_count} | OUT: {out_count} | NET: {net_count}"
            cv2.putText(frame_with_lines, stats_text, (mid_x, mid_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame_with_lines

def process_frame(frame, lines, colors, cement_model, tracker):
    """Process a single frame for cement bag detection and tracking"""
    # Make a copy of the frame for drawing
    frame_with_detections = frame.copy()
    
    # Detect cement bags with error handling
    try:
        with torch.no_grad():
            results = cement_model(frame, conf=0.2)[0]  # Lower confidence for better recall
    except Exception as e:
        st.error(f"Error during inference: {str(e)}. Trying CPU...")
        # Try processing on CPU for this frame
        try:
            with torch.no_grad():
                results = cement_model(frame, conf=0.2, device='cpu')[0]
        except Exception as inner_e:
            st.error(f"CPU fallback also failed: {str(inner_e)}")
            # Return original frame if detection fails
            return frame
    
    # Extract detection results
    boxes = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        if score > 0.2:
            boxes.append([int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)])
    
    # Track objects
    bbox_idx = tracker.update(boxes)
    
    # Draw boxes and IDs
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        track_id = bbox_idx[idx]
        
        # Get object center point (used for line crossing detection)
        center_x = x + w//2
        center_y = y + h//2
        center_point = (center_x, center_y)
        
        # Draw bounding box
        cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw ID
        cv2.putText(frame_with_detections, f"ID: {track_id}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check for line crossing
        if track_id in st.session_state.last_positions:
            prev_point = st.session_state.last_positions[track_id]
            
            # Check each line for crossing
            for i, line in enumerate(lines):
                direction = get_movement_direction(prev_point, center_point, line)
                if direction:
                    # Update crossing stats for this line
                    if direction == "IN":
                        st.session_state.crossing_stats[i]['in'] += 1
                    else:  # OUT
                        st.session_state.crossing_stats[i]['out'] += 1
        
        # Update last known position
        st.session_state.last_positions[track_id] = center_point
    
    # Draw lines and counts on the frame
    if lines:
        frame_with_detections = draw_lines(frame_with_detections, lines, colors)
    
    return frame_with_detections

# Initialize session state variables
if 'tracker' not in st.session_state:
    st.session_state.tracker = Tracker()
if 'rtsp_connection_active' not in st.session_state:
    st.session_state.rtsp_connection_active = False
if 'rtsp_frame_count' not in st.session_state:
    st.session_state.rtsp_frame_count = 0
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = "rtsp"  # Default mode
if 'crossing_stats' not in st.session_state:
    # Dictionary to hold line crossing statistics for each line
    st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
if 'last_positions' not in st.session_state:
    # Dictionary to track last known positions of objects
    st.session_state.last_positions = {}
if 'lines' not in st.session_state:
    # Default lines - empty list to be populated
    st.session_state.lines = []
if 'line_colors' not in st.session_state:
    # Colors for the lines
    st.session_state.line_colors = []

# Load model
cement_model = load_models()

# Create tabs for different detection modes
detection_tab, line_tab, cluster_tab = st.tabs(["Detection Mode", "Line Configuration", "Cluster Management"])

with detection_tab:
    # Choose detection mode
    st.session_state.detection_mode = st.radio(
        "Choose Detection Mode:",
        options=["RTSP Camera", "Upload Video"],
        horizontal=True
    )
    
    if st.session_state.detection_mode == "RTSP Camera":
        # RTSP Camera Configuration
        st.subheader("RTSP Camera Configuration")
        col1, col2 = st.columns(2)
        with col1:
            rtsp_base = st.text_input("RTSP Base URL", 
                                     value="rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels/")
        with col2:
            channel_number = st.text_input("Channel Number", value="101")
        
        # Form full RTSP URL
        rtsp_url = f"{rtsp_base}{channel_number}"
        st.info(f"Full RTSP URL: {rtsp_url}")
        
        # Start/Stop RTSP Stream button
        rtsp_control_col1, rtsp_control_col2 = st.columns(2)
        
        button_label = "Start RTSP Stream" if not st.session_state.rtsp_connection_active else "Stop RTSP Stream"
        if rtsp_control_col1.button(button_label, key="rtsp_control_btn"):
            # Toggle the active state
            st.session_state.rtsp_connection_active = not st.session_state.rtsp_connection_active
            if st.session_state.rtsp_connection_active:
                # Reset stats when starting
                st.session_state.rtsp_frame_count = 0
                # Force a rerun to start the stream processing
                st.rerun()
        
        # Create placeholders for video and metrics
        video_frame = st.empty()
        status_message = st.empty()
        
        # Only process stream if active flag is True
        if st.session_state.rtsp_connection_active:
            # Initialize variable outside try block
            cap = None
            
            try:
                # Show a message that we're connecting to the stream
                status_message.info("Connecting to RTSP stream...")
                
                # Set up video capture
                cap = cv2.VideoCapture(rtsp_url)
                
                if not cap.isOpened():
                    status_message.error("❌ Error: Cannot open RTSP stream. Check URL and try again.")
                    st.session_state.rtsp_connection_active = False
                    st.rerun()
                
                status_message.success("Connected to RTSP stream successfully")
                
                # Initialize stop flag
                stop_rtsp = False
                
                # Process frames in a loop while connection is active
                while cap.isOpened() and st.session_state.rtsp_connection_active and not stop_rtsp:
                    ret, frame = cap.read()
                    
                    if not ret:
                        status_message.warning("⚠️ Error reading from RTSP stream. Reconnecting...")
                        cap.release()
                        time.sleep(2)  # Wait before reconnecting
                        cap = cv2.VideoCapture(rtsp_url)
                        continue  # Skip this iteration and try again
                    
                    # Process the frame
                    processed_frame = process_frame(frame, st.session_state.lines, st.session_state.line_colors, cement_model, st.session_state.tracker)
                    
                    # Count frames
                    st.session_state.rtsp_frame_count += 1
                    
                    # Convert to RGB for display
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the processed frame
                    video_frame.image(processed_frame_rgb, caption=f"RTSP Stream - {rtsp_url}", 
                                      use_container_width=True)
                    
                    # Small delay to prevent UI from freezing
                    time.sleep(0.01)
                
                # Release camera when done
                cap.release()
                status_message.info("RTSP stream has stopped.")
            
            except Exception as e:
                st.error(f"❌ Error processing RTSP stream: {str(e)}")
                st.session_state.rtsp_connection_active = False
                if cap is not None:
                    cap.release()
            
        else:
            # Display a message when the stream is not active
            st.write("Click 'Start RTSP Stream' to begin processing camera feed")
    
    else:  # Upload Video mode
        # Video upload section
        st.subheader("Upload Video")
        uploaded_file = st.file_uploader("Choose a video file to process", type=['mp4', 'avi', 'mov'])
        
        # Check if file was already processed
        file_already_processed = False
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
            file_already_processed = file_id in st.session_state.processed_files
            
            if file_already_processed:
                st.warning(f"⚠️ This video has already been processed. Re-processing will not update inventory again.")
                reprocess_anyway = st.checkbox("Process again anyway")
                if not reprocess_anyway:
                    uploaded_file = None  # Prevent processing
        
        # Process video button placeholder
        video_process_placeholder = st.empty()
        
        if uploaded_file:
            if video_process_placeholder.button("Process Video"):
                try:
                    # Create temporary file to save the uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file_path = tmp_file.name
                        tmp_file.write(uploaded_file.getvalue())
                        
                    # Process video
                    st.write("### Processing Video...")
                    
                    # Track processed files to avoid double-counting
                    file_id = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
                    st.session_state.processed_files.add(file_id)
                    
                    # Setup video capture
                    cap = cv2.VideoCapture(tmp_file_path)
                    
                    # Reset stats for the new video
                    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Setup output video
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    frame_count = 0
                    
                    # Setup output video
                    output_path = os.path.join(os.path.dirname(tmp_file_path), "processed_" + os.path.basename(tmp_file_path))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
                    
                    # Reset tracker
                    tracker = Tracker()
                    
                    # Process the video frame by frame
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        # Process frame
                        processed_frame = process_frame(frame, st.session_state.lines, st.session_state.line_colors, cement_model, tracker)
                        
                        # Write frame to output video
                        out.write(processed_frame)
                        
                        # Update progress
                        frame_count += 1
                        progress_value = int(frame_count / total_frames * 100)
                        progress_bar.progress(progress_value)
                    
                    # Release resources
                    cap.release()
                    out.release()
                    
                    # Display results
                    st.success("✅ Video processing complete!")
                    
                    # Provide download button
                    st.write("### Save Processed Video")
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="Download processed video",
                            data=f,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    # Clean up temporary file
                    if 'tmp_file_path' in locals():
                        os.unlink(tmp_file_path)
            
            else:
                st.write("Click 'Process Video' to start detection")
        else:
            st.write("Please upload a video file to process")

with line_tab:
    st.subheader("Line Configuration")
    
    # Number of lines
    num_lines = st.number_input("Number of counting lines", min_value=0, max_value=5, value=len(st.session_state.lines))
    
    if num_lines != len(st.session_state.lines):
        # Adjust the number of lines
        if num_lines > len(st.session_state.lines):
            # Add more lines
            for i in range(len(st.session_state.lines), num_lines):
                st.session_state.lines.append([(100, 100 + i*50), (500, 100 + i*50)])
                st.session_state.line_colors.append(LINE_COLORS[i % len(LINE_COLORS)])
        else:
            # Remove excess lines
            st.session_state.lines = st.session_state.lines[:num_lines]
            st.session_state.line_colors = st.session_state.line_colors[:num_lines]
    
    # Configure each line
    for i in range(num_lines):
        st.write(f"### Line {i+1}")
        col1, col2 = st.columns(2)
        
        # Get current line coordinates
        if i < len(st.session_state.lines):
            (x1, y1), (x2, y2) = st.session_state.lines[i]
        else:
            x1, y1, x2, y2 = 100, 100 + i*50, 500, 100 + i*50
        
        with col1:
            st.write("Start Point")
            new_x1 = st.number_input(f"X1 for Line {i+1}", value=x1, key=f"x1_{i}")
            new_y1 = st.number_input(f"Y1 for Line {i+1}", value=y1, key=f"y1_{i}")
        
        with col2:
            st.write("End Point")
            new_x2 = st.number_input(f"X2 for Line {i+1}", value=x2, key=f"x2_{i}")
            new_y2 = st.number_input(f"Y2 for Line {i+1}", value=y2, key=f"y2_{i}")
        
        # Update line coordinates if changed
        if (new_x1, new_y1) != (x1, y1) or (new_x2, new_y2) != (x2, y2):
            if i < len(st.session_state.lines):
                st.session_state.lines[i] = [(new_x1, new_y1), (new_x2, new_y2)]
            else:
                st.session_state.lines.append([(new_x1, new_y1), (new_x2, new_y2)])
                st.session_state.line_colors.append(LINE_COLORS[i % len(LINE_COLORS)])
    
    # Reset counting button
    if st.button("Reset Counting Statistics"):
        st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
        st.session_state.last_positions = {}
        st.success("Counting statistics reset successfully!")
    
    # Display current counting statistics
    if st.session_state.lines:
        st.write("### Current Counting Statistics")
        for i, line in enumerate(st.session_state.lines):
            if i in st.session_state.crossing_stats:
                stats = st.session_state.crossing_stats[i]
                st.write(f"**Line {i+1}**: IN: {stats['in']} | OUT: {stats['out']} | NET: {stats['in'] - stats['out']}")

with cluster_tab:
    st.subheader("Cluster Management")
    
    # Load clusters from Flask API
    clusters = load_clusters()
    
    # Display clusters in a selectbox
    if clusters:
        cluster_names = [cluster['name'] for cluster in clusters]
        selected_cluster_name = st.selectbox("Select a cluster", cluster_names)
        
        # Find the selected cluster object
        for cluster in clusters:
            if cluster['name'] == selected_cluster_name:
                st.session_state.selected_cluster = cluster
                break
        
        # Display cluster details
        st.write(f"Selected cluster: **{st.session_state.selected_cluster['name']}**")
        st.write(f"Current bag count: **{st.session_state.selected_cluster['bag_count']}**")
        st.write(f"Cluster ID: {st.session_state.selected_cluster['id']}")
    else:
        st.warning("No clusters found. Create a new cluster below.")
        st.session_state.selected_cluster = None
    
    # Create new cluster section
    st.write("---")
    st.subheader("Create New Cluster")
    
    new_cluster_name = st.text_input("Cluster Name")
    initial_bag_count = st.number_input("Initial Bag Count", min_value=0, value=0)
    
    if st.button("Create Cluster"):
        try:
            data = {"name": new_cluster_name, "bag_count": initial_bag_count}
            response = requests.post(f"{FLASK_API_URL}/clusters", json=data)
            
            if response.status_code in (200, 201):
                st.success(f"Created cluster '{new_cluster_name}'")
                # Reload clusters
                clusters = load_clusters()
                st.rerun()
            else:
                st.error(f"Failed to create cluster: {response.text}")
        except Exception as e:
            st.error(f"Error creating cluster: {str(e)}")

# Add information section
with st.expander("Information", expanded=False):
    st.write("""
    ### JSW Cement Bag Detection System
    
    This application detects cement bags in real-time from RTSP camera feeds or uploaded videos,
    and tracks when cement bags are detected.
    
    #### How to use:
    1. Select a cluster for inventory tracking in the Cluster Management tab
    2. Choose between RTSP camera feed or video upload mode
    3. Start processing with either 'Start RTSP Stream' or 'Process Video'
    
    #### RTSP URL Format:
    - Base URL: rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels/
    - Channel Range: 101-701
    """)

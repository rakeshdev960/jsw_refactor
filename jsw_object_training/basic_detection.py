import streamlit as st
import cv2
import numpy as np
import time
import math
import requests
import tempfile
import os
import torch
import subprocess
import sys
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

# Define line colors
LINE_COLORS = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]

# Resize factor for incoming frames to reduce memory usage
RESIZE_FACTOR = 0.5  # 50% of original size; adjust as needed

# Cache the models
@st.cache_resource
def load_models():
    """Load YOLO model, prefer GPU, fallback to CPU if necessary"""
    try:
        model = YOLO('F:/jsw20042025/best_cement_bags.pt')
        if torch.cuda.is_available():
            try:
                model.to('cuda')
            except Exception as e:
                st.warning(f"Unable to move model to GPU: {e}. Using CPU instead.")
                model.to('cpu')
        else:
            model.to('cpu')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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

# -------------------------- FRAME PROCESSING ----------------------------- #

def process_frame(frame, lines, colors, cement_model, tracker):
    """Process a single frame for cement bag detection and tracking"""
    # Make a copy of the frame for drawing
    frame_with_detections = frame.copy()
    
    try:
        # Try inference on current device (GPU if available)
        results = cement_model(frame, conf=0.2)[0]
    except RuntimeError as e:
        # Handle CUDA out of memory by falling back to CPU
        if "out of memory" in str(e).lower():
            st.warning("CUDA OOM detected – switching inference to CPU for this session")
            try:
                torch.cuda.empty_cache()
                cement_model.to('cpu')
                results = cement_model(frame, conf=0.2)[0]
                # (continue with normal processing below by re‑entering boxes extraction)
                boxes = []
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    if score > 0.2:
                        boxes.append([int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)])
                # Track objects
                bbox_idx = tracker.update(boxes)
                # Drawing logic identical to main path (duplicated minimal to keep patch short)
                for idx, box in enumerate(boxes):
                    x, y, w, h = box
                    track_id = bbox_idx[idx]
                    center_point = (x + w // 2, y + h // 2)
                    cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_with_detections, f"ID: {track_id}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if track_id in st.session_state.last_positions:
                        prev_point = st.session_state.last_positions[track_id]
                        for i, line in enumerate(lines):
                            direction = get_movement_direction(prev_point, center_point, line)
                            if direction:
                                if direction == "IN":
                                    st.session_state.crossing_stats[i]['in'] += 1
                                else:
                                    st.session_state.crossing_stats[i]['out'] += 1
                    st.session_state.last_positions[track_id] = center_point
                if lines:
                    frame_with_detections = draw_lines(frame_with_detections, lines, colors)
            except Exception as cpu_e:
                st.error(f"CPU fallback failed: {cpu_e}")
        else:
            st.error(f"Error processing frame: {str(e)}")
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
    
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

# Initialize default lines
def create_default_lines(frame_width, frame_height, num_lines=2):
    lines = []
    for i in range(num_lines):
        y_pos = (i + 1) * frame_height // (num_lines + 1)
        lines.append([(50, y_pos), (frame_width - 50, y_pos)])
    return lines

# Initialize session state variables
if 'tracker' not in st.session_state:
    st.session_state.tracker = Tracker()
if 'crossing_stats' not in st.session_state:
    # Dictionary to hold line crossing statistics for each line
    st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
if 'last_positions' not in st.session_state:
    # Dictionary to track last known positions of objects
    st.session_state.last_positions = {}
if 'lines' not in st.session_state:
    # Placeholder for lines, will be populated based on frame size
    st.session_state.lines = []
if 'line_colors' not in st.session_state:
    # Colors for the first few lines
    st.session_state.line_colors = LINE_COLORS.copy()
if 'rtsp_connection_active' not in st.session_state:
    st.session_state.rtsp_connection_active = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

# Load model - try GPU first with CPU fallback
import torch
cement_model = load_models()

# Create tabs for different detection modes
detection_tab, line_tab, cluster_tab = st.tabs(["Detection Mode", "Line Configuration", "Cluster Management"])

with line_tab:
    st.subheader("Line Configuration")
    
    # Number of lines
    num_lines = st.number_input("Number of counting lines", min_value=1, max_value=5, value=len(st.session_state.lines) if st.session_state.lines else 2)
    
    # Configure lines
    st.write("### Configure Counting Lines")
    st.write("Set the coordinates for each counting line:")
    
    # Adjust number of lines if needed
    if num_lines != len(st.session_state.lines):
        # Create default lines if we don't have frame dimensions yet
        if not st.session_state.lines:
            # Default to 640x480 resolution if we don't know the actual frame size
            st.session_state.lines = create_default_lines(640, 480, num_lines)
        else:
            # Get dimensions from existing lines
            frame_width = max(max(pt[0] for pt in line) for line in st.session_state.lines)
            frame_height = max(max(pt[1] for pt in line) for line in st.session_state.lines)
            
            if num_lines > len(st.session_state.lines):
                # Add more lines
                st.session_state.lines.extend(create_default_lines(frame_width, frame_height, num_lines - len(st.session_state.lines)))
                # Add more colors if needed
                while len(st.session_state.line_colors) < num_lines:
                    st.session_state.line_colors.append(LINE_COLORS[len(st.session_state.line_colors) % len(LINE_COLORS)])
            else:
                # Remove excess lines
                st.session_state.lines = st.session_state.lines[:num_lines]
                st.session_state.line_colors = st.session_state.line_colors[:num_lines]
    
    # Configure each line
    for i in range(num_lines):
        st.write(f"#### Line {i+1}")
        
        # Get current coordinates
        if i < len(st.session_state.lines):
            (x1, y1), (x2, y2) = st.session_state.lines[i]
        else:
            x1, y1, x2, y2 = 100, 100 + i*50, 500, 100 + i*50
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Start Point")
            new_x1 = st.number_input(f"X1 for Line {i+1}", value=x1, key=f"x1_{i}")
            new_y1 = st.number_input(f"Y1 for Line {i+1}", value=y1, key=f"y1_{i}")
        
        with col2:
            st.write("End Point")
            new_x2 = st.number_input(f"X2 for Line {i+1}", value=x2, key=f"x2_{i}")
            new_y2 = st.number_input(f"Y2 for Line {i+1}", value=y2, key=f"y2_{i}")
        
        # Update line
        if i < len(st.session_state.lines):
            st.session_state.lines[i] = [(new_x1, new_y1), (new_x2, new_y2)]
    
    # Reset button
    if st.button("Reset Counts"):
        st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
        st.session_state.last_positions = {}
        st.success("Counting statistics have been reset.")
    
    # Display statistics
    st.write("### Current Counts")
    for i in range(min(len(st.session_state.lines), num_lines)):
        if i in st.session_state.crossing_stats:
            in_count = st.session_state.crossing_stats[i]['in']
            out_count = st.session_state.crossing_stats[i]['out']
            net_count = in_count - out_count
            
            st.write(f"**Line {i+1}:** IN: {in_count} | OUT: {out_count} | NET: {net_count}")

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

with detection_tab:
    # Choose detection mode
    detection_mode = st.radio(
        "Choose Detection Mode:",
        options=["RTSP Camera", "Upload Video"],
        horizontal=True
    )
    
    if detection_mode == "RTSP Camera":
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
        
        # Verify selected cluster
        if not st.session_state.selected_cluster:
            st.warning("⚠️ Please select a cluster in the Cluster Management tab before starting detection")
        else:
            # Start/Stop RTSP Stream button
            if st.session_state.rtsp_connection_active:
                if st.button("Stop RTSP Stream"):
                    st.session_state.rtsp_connection_active = False
                    
                    # Update inventory when stopping
                    total_in = sum(stats['in'] for stats in st.session_state.crossing_stats.values())
                    total_out = sum(stats['out'] for stats in st.session_state.crossing_stats.values())
                    
                    if total_in > 0 or total_out > 0:
                        cluster_info = st.session_state.selected_cluster
                        send_to_inventory_system(cluster_info['name'], total_in, total_out)
                        st.success(f"Updated inventory: IN={total_in}, OUT={total_out}")
                        
                        # Reset counters after update
                        st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
                        st.session_state.last_positions = {}
            else:
                if st.button("Start RTSP Stream"):
                    st.session_state.rtsp_connection_active = True
                    st.rerun()
            
            # Display stream
            stream_placeholder = st.empty()
            
            # Process RTSP stream if active
            if st.session_state.rtsp_connection_active:
                try:
                    # Display status
                    status = st.empty()
                    status.info("Connecting to RTSP stream...")
                    
                    # Initialize video capture
                    cap = cv2.VideoCapture(rtsp_url)
                    
                    if not cap.isOpened():
                        status.error("Failed to open RTSP stream. Please check the URL and try again.")
                        st.session_state.rtsp_connection_active = False
                    else:
                        # Get frame dimensions
                        ret, frame = cap.read()
                        if not ret:
                            status.error("Failed to read from RTSP stream.")
                            st.session_state.rtsp_connection_active = False
                            cap.release()
                        else:
                            # Initialize lines if we don't have any
                            if not st.session_state.lines:
                                height, width = frame.shape[:2]
                                st.session_state.lines = create_default_lines(width, height, num_lines)
                            
                            status.success("Connected to RTSP stream")
                            
                            # Create metrics display
                            col1, col2, col3 = st.columns(3)
                            in_metric = col1.empty()
                            out_metric = col2.empty()
                            net_metric = col3.empty()
                            
                            # Process frames
                            while cap.isOpened() and st.session_state.rtsp_connection_active:
                                ret, frame = cap.read()
                                if not ret:
                                    status.warning("Error reading from stream. Reconnecting...")
                                    cap.release()
                                    time.sleep(2)
                                    cap = cv2.VideoCapture(rtsp_url)
                                    continue
                                
                                # Optional downscale to reduce memory usage
                                if RESIZE_FACTOR != 1.0:
                                    frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
                                
                                # Process the frame
                                processed_frame = process_frame(frame, st.session_state.lines, 
                                                             st.session_state.line_colors, 
                                                             cement_model, st.session_state.tracker)
                                
                                # Convert to RGB for display
                                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                
                                # Display the frame
                                stream_placeholder.image(processed_frame_rgb, caption="RTSP Stream", 
                                                      use_container_width=True)
                                
                                # Update metrics every 15 frames
                                if st.session_state.rtsp_connection_active:
                                    total_in = sum(stats['in'] for stats in st.session_state.crossing_stats.values())
                                    total_out = sum(stats['out'] for stats in st.session_state.crossing_stats.values())
                                    net_total = total_in - total_out
                                    
                                    in_metric.metric("Total IN", total_in)
                                    out_metric.metric("Total OUT", total_out)
                                    net_metric.metric("NET", net_total)
                                
                                # Short delay to prevent UI freezing
                                time.sleep(0.01)
                            
                            # Release resources when done
                            cap.release()
                            status.info("RTSP stream stopped")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.rtsp_connection_active = False
    
    else:  # Upload Video mode
        st.subheader("Upload Video")
        
        # Verify selected cluster
        if not st.session_state.selected_cluster:
            st.warning("⚠️ Please select a cluster in the Cluster Management tab before uploading a video")
        else:
            # Video upload
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
            
            if uploaded_file:
                # Check if file was already processed
                file_id = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
                file_already_processed = file_id in st.session_state.processed_files
                
                if file_already_processed:
                    st.warning("This video has already been processed. Re-processing will not update inventory again.")
                    reprocess_anyway = st.checkbox("Process again anyway")
                    if not reprocess_anyway:
                        uploaded_file = None
                
                if uploaded_file and st.button("Process Video"):
                    # Process the video
                    st.write("### Processing Video")
                    
                    try:
                        # Save file to disk
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file_path = tmp_file.name
                            tmp_file.write(uploaded_file.getvalue())
                        
                        # Add to processed files
                        st.session_state.processed_files.add(file_id)
                        
                        # Create video capture
                        cap = cv2.VideoCapture(tmp_file_path)
                        
                        if not cap.isOpened():
                            st.error("Failed to open video file.")
                        else:
                            # Get video properties
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            # Reset stats
                            st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
                            st.session_state.last_positions = {}
                            
                            # Initialize lines for this video
                            st.session_state.lines = create_default_lines(width, height, num_lines)
                            
                            # Setup output video
                            output_path = os.path.join(os.path.dirname(tmp_file_path), f"processed_{os.path.basename(tmp_file_path)}")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            
                            # Progress bar
                            progress_bar = st.progress(0)
                            frame_count = 0
                            
                            # Create tracker for this video
                            tracker = Tracker()
                            
                            # Process frames
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # Optional downscale
                                if RESIZE_FACTOR != 1.0:
                                    frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
                                
                                # Process frame
                                processed_frame = process_frame(frame, st.session_state.lines, 
                                                             st.session_state.line_colors, 
                                                             cement_model, tracker)
                                
                                # Write frame to output
                                out.write(processed_frame)
                                
                                # Update progress
                                frame_count += 1
                                if frame_count % 10 == 0:  # Update every 10 frames
                                    progress = min(int(frame_count / total_frames * 100), 100)
                                    progress_bar.progress(progress)
                            
                            # Release resources
                            cap.release()
                            out.release()
                            
                            # Display results
                            st.success("Video processing complete!")
                            
                            # Show metrics
                            total_in = sum(stats['in'] for stats in st.session_state.crossing_stats.values())
                            total_out = sum(stats['out'] for stats in st.session_state.crossing_stats.values())
                            net_total = total_in - total_out
                            
                            st.write("### Results")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total IN", total_in)
                            col2.metric("Total OUT", total_out)
                            col3.metric("NET", net_total)
                            
                            # Update inventory
                            if total_in > 0 or total_out > 0:
                                cluster_info = st.session_state.selected_cluster
                                with st.spinner("Updating inventory..."):
                                    response, status_code = send_to_inventory_system(
                                        cluster_info['name'], total_in, total_out)
                                    
                                    if status_code in (200, 201):
                                        st.success(f"Updated inventory for {cluster_info['name']}")
                                    else:
                                        st.error(f"Failed to update inventory: {response.get('error', 'Unknown error')}")
                            
                            # Provide download button
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="Download Processed Video",
                                    data=f,
                                    file_name=f"processed_{uploaded_file.name}",
                                    mime="video/mp4"
                                )
                        
                        # Clean up
                        os.unlink(tmp_file_path)
                    
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        if 'tmp_file_path' in locals():
                            os.unlink(tmp_file_path)

# Add information section
with st.expander("Information", expanded=False):
    st.write("""
    ### JSW Cement Bag Detection System
    
    This application detects cement bags in real-time from RTSP camera feeds or uploaded videos,
    and tracks when bags cross defined lines. The system counts bags crossing from right to left as OUT
    and bags crossing from left to right as IN.
    
    #### How to use:
    1. Select a cluster for inventory tracking in the Cluster Management tab
    2. Configure counting lines in the Line Configuration tab
    3. Choose between RTSP camera feed or video upload mode
    4. Start processing with either 'Start RTSP Stream' or 'Process Video'
    5. Statistics will update automatically
    6. Inventory system will be updated with bag movement records
    
    #### RTSP URL Format:
    - Base URL: rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels/
    - Channel Range: 101-701
    """)

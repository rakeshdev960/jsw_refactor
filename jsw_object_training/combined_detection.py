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


st.set_page_config(
    page_title="JSW Cement Bag Detection",
    page_icon="📦",
    layout="wide"
)

FLASK_API_URL = "http://localhost:5000"

st.title("JSW Cement Bag Detection System")
st.write("Process RTSP camera feeds or upload videos to detect and count cement bags")

# Load custom model
@st.cache_resource
def load_cement_model():
    try:
        model = YOLO('best_cement_bags_2025-05-29.pt')
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model(test_img)
    except Exception as e:
        st.warning(f"GPU acceleration not available for cement model, falling back to CPU: {str(e)}")
        model = YOLO('best_cement_bags_2025-05-29.pt', device='cpu')
    return model

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO('yolov8n.pt')
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model(test_img)
    except Exception as e:
        st.warning(f"GPU acceleration not available for YOLO model, falling back to CPU: {str(e)}")
        model = YOLO('yolov8n.pt', device='cpu')
    return model

# Create combined class names dictionary
@st.cache_resource
def get_combined_names(_cement_model, _yolo_model):
    combined_names = {}
    # Add custom classes with prefix
    for class_id, name in _cement_model.names.items():
        combined_names[class_id] = f"Custom: {name}"
    
    # Add YOLO default classes with offset
    offset = 100  # Use an offset to separate custom and default classes
    for class_id, name in _yolo_model.names.items():
        combined_names[class_id + offset] = f"YOLO: {name}"
    
    return combined_names, offset

# Main function to load all models and combine them
def load_models():
    cement_model = load_cement_model()
    yolo_model = load_yolo_model()
    combined_names, offset = get_combined_names(cement_model, yolo_model)
    
    # Store these in st.session_state for access throughout the app
    st.session_state.cement_model = cement_model
    st.session_state.yolo_model = yolo_model
    st.session_state.combined_names = combined_names
    st.session_state.class_offset = offset
    
    return cement_model

def load_clusters():
    try:
        response = requests.get(f"{FLASK_API_URL}/clusters")
        clusters = response.json()
        st.session_state.available_clusters = clusters
        return clusters
    except Exception as e:
        st.error(f"Error loading clusters: {str(e)}")
        return []

def send_to_inventory_system(cluster_name, in_count, out_count):
    # Check if we should use mock mode (when API is unavailable)
    use_mock = st.session_state.get('use_mock_api', False)
    
    if use_mock:
        st.info(f"Using mock mode: IN={in_count}, OUT={out_count}, NET={in_count-out_count}")
        return {"status": "success", "message": "Mock inventory update successful"}, 200
    
    # Display the counts being sent to inventory
    st.info(f"Sending to inventory: Cluster={cluster_name}, IN={in_count}, OUT={out_count}, NET={in_count-out_count}")
    
    try:
        # Test connection to API server first
        try:
            st.info(f"Connecting to API at {FLASK_API_URL}...")
            response = requests.get(f"{FLASK_API_URL}/clusters", timeout=5)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            clusters = response.json()
            st.success(f"Successfully connected to API. Found {len(clusters)} clusters.")
        except requests.exceptions.RequestException as conn_err:
            st.error(f"Cannot connect to inventory API at {FLASK_API_URL}: {str(conn_err)}")
            st.info("Enabling mock mode for future updates. Restart the application to try connecting to the API again.")
            st.session_state.use_mock_api = True
            return {"error": f"API connection failed: {str(conn_err)}"}, 500

        # Find if cluster exists
        cluster_exists = False
        cluster_id = None

        for cluster in clusters:
            if cluster['name'] == cluster_name:
                cluster_exists = True
                cluster_id = cluster['id']
                st.info(f"Found existing cluster: {cluster_name} (ID: {cluster_id})")
                break

        # Process based on whether cluster exists
        if cluster_exists:
            movement_success = True
            error_details = []

            # Process IN movement if needed
            if in_count > 0:
                st.info(f"Recording IN movement of {in_count} bags for cluster {cluster_name}")
                in_data = {"movement_type": "IN", "quantity": in_count}
                try:
                    in_response = requests.post(
                        f"{FLASK_API_URL}/clusters/{cluster_id}/movement", 
                        json=in_data,
                        timeout=5
                    )
                    
                    # Check response and log details
                    if in_response.status_code in (200, 201):
                        st.success(f"IN movement recorded successfully")
                    else:
                        st.error(f"IN movement failed with status code: {in_response.status_code}")
                        st.error(f"Response: {in_response.text}")
                        movement_success = False
                        error_details.append(f"IN movement failed: {in_response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to record IN movement: {str(e)}")
                    movement_success = False
                    error_details.append(f"IN movement request failed: {str(e)}")

            # Process OUT movement if needed
            if out_count > 0:
                st.info(f"Recording OUT movement of {out_count} bags for cluster {cluster_name}")
                out_data = {"movement_type": "OUT", "quantity": out_count}
                try:
                    out_response = requests.post(
                        f"{FLASK_API_URL}/clusters/{cluster_id}/movement", 
                        json=out_data,
                        timeout=5
                    )
                    
                    # Check response and log details
                    if out_response.status_code in (200, 201):
                        st.success(f"OUT movement recorded successfully")
                    else:
                        st.error(f"OUT movement failed with status code: {out_response.status_code}")
                        st.error(f"Response: {out_response.text}")
                        movement_success = False
                        error_details.append(f"OUT movement failed: {out_response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to record OUT movement: {str(e)}")
                    movement_success = False
                    error_details.append(f"OUT movement request failed: {str(e)}")

            # Return final result
            if movement_success:
                return {"status": "success", "message": "Inventory updated successfully"}, 200
            else:
                return {"error": "Movement recording failed", "details": error_details}, 500
        else:
            # Cluster doesn't exist yet, create it first
            st.info(f"Cluster '{cluster_name}' does not exist. Creating it now...")
            try:
                # Create the cluster
                create_data = {
                    "name": cluster_name,
                    "bag_count": 0,  # Start with zero bags
                }
                
                response = requests.post(
                    f"{FLASK_API_URL}/clusters", 
                    json=create_data,
                    timeout=5
                )
                
                if response.status_code == 201:
                    new_cluster = response.json()
                    st.success(f"Created new cluster: {cluster_name} (ID: {new_cluster['id']})")
                    
                    # Now recursively call this function again to record the movements
                    return send_to_inventory_system(cluster_name, in_count, out_count)
                else:
                    st.error(f"Cluster creation failed with status code: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    return {"error": f"Cluster creation failed with status {response.status_code}"}, 500
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to create new cluster: {str(e)}")
                return {"error": f"Cluster creation failed: {str(e)}"}, 500
    except Exception as e:
        st.error(f"Unexpected error in inventory system: {str(e)}")
        return {"error": str(e)}, 500

# Define color constants for lines
LINE_COLORS = [
    (0, 255, 0),   # Green
    (0, 0, 255),   # Red 
    (255, 0, 0),   # Blue
    (0, 255, 255), # Yellow
    (255, 0, 255), # Magenta
    (255, 255, 0), # Cyan
    (128, 0, 0),   # Dark blue
    (0, 128, 0),   # Dark green
    (0, 0, 128)    # Dark red
]

COLOR_NAMES = [
    "Green",
    "Red",
    "Blue", 
    "Yellow",
    "Magenta",
    "Cyan",
    "Dark Blue",
    "Dark Green",
    "Dark Red"
]

def main():
    cement_model = load_models()

    st.sidebar.header("Detection Mode")
    mode = st.sidebar.radio("Select Mode", ("RTSP Camera", "Upload Video"))
    
    # Initialize mock API mode if not already set
    if 'use_mock_api' not in st.session_state:
        st.session_state.use_mock_api = False
    
    # Option to toggle mock mode
    mock_mode = st.sidebar.checkbox("Use Mock API (when inventory API unavailable)", 
                                   value=st.session_state.use_mock_api)
    st.session_state.use_mock_api = mock_mode
    
    # Class selection feature
    st.sidebar.header("Class Selection")
    # Get all available classes from the combined model
    all_classes = list(st.session_state.combined_names.values())
    
    # Initialize selected classes in session state if not already there
    if 'selected_classes' not in st.session_state:
        st.session_state.selected_classes = all_classes.copy()
    
    # Option to select all or none
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Select All"):
        st.session_state.selected_classes = all_classes.copy()
    if col2.button("Clear All"):
        st.session_state.selected_classes = []
        
    # Add quick selection options for class types
    col3, col4 = st.sidebar.columns(2)
    if col3.button("Custom Classes Only"):
        st.session_state.selected_classes = [name for name in all_classes if name.startswith("Custom:")]
    if col4.button("YOLO Classes Only"):
        st.session_state.selected_classes = [name for name in all_classes if name.startswith("YOLO:")]
    
    # Group classes by type for better organization
    custom_classes = [name for name in all_classes if name.startswith("Custom:")]
    yolo_classes = [name for name in all_classes if name.startswith("YOLO:")]
    
    # Display custom classes first
    st.sidebar.write("Custom Classes:")
    custom_selected = []
    for class_name in custom_classes:
        if st.sidebar.checkbox(class_name, value=class_name in st.session_state.selected_classes):
            custom_selected.append(class_name)
    
    # Display YOLO classes with expander to save space
    with st.sidebar.expander("YOLO Default Classes"):
        yolo_selected = []
        for class_name in yolo_classes:
            if st.checkbox(class_name, value=class_name in st.session_state.selected_classes):
                yolo_selected.append(class_name)
    
    # Combine selected classes
    selected_classes = custom_selected + yolo_selected
    
    # Update session state
    st.session_state.selected_classes = selected_classes
    
    # Show warning if no classes selected
    if not selected_classes:
        st.sidebar.warning("No classes selected. Nothing will be detected.")
    else:
        st.sidebar.success(f"Detecting {len(selected_classes)} classes")
        
    # Convert selected class names to class IDs for filtering
    selected_class_ids = [class_id for class_id, name in st.session_state.combined_names.items() 
                         if name in st.session_state.selected_classes]

    st.sidebar.header("Cluster")
    clusters = load_clusters()
    cluster_names = [c['name'] for c in clusters] if clusters else []
    selected_cluster = st.sidebar.selectbox("Select Cluster", cluster_names) if cluster_names else None

    if not selected_cluster:
        st.warning("No cluster selected or available. Please add one via backend.")
        return

    # Initialize crossing lines in session state if not already there
    if 'crossing_lines' not in st.session_state:
        st.session_state.crossing_lines = [
            {
                'start': (100, 300),
                'end': (500, 300),
                'color_index': 0,
                'enabled': True,
                'name': 'Line 1',
                'in_count': 0,
                'out_count': 0
            }
        ]
    
    # Crossing Lines configuration
    st.sidebar.header("Crossing Lines")
    
    # Add a new line button
    if st.sidebar.button("Add New Line"):
        # Get a default name with incremented number
        next_line_num = len(st.session_state.crossing_lines) + 1
        next_color_idx = next_line_num % len(LINE_COLORS)
        st.session_state.crossing_lines.append({
            'start': (100, 300),
            'end': (500, 300),
            'color_index': next_color_idx,
            'enabled': True,
            'name': f'Line {next_line_num}',
            'in_count': 0,
            'out_count': 0
        })
    
    # Display and edit crossing lines
    lines_to_delete = []
    for i, line in enumerate(st.session_state.crossing_lines):
        with st.sidebar.expander(f"{line['name']}", expanded=True):
            # Line name
            new_name = st.text_input("Line Name", line['name'], key=f"name_{i}")
            line['name'] = new_name
            
            # Line coordinates
            col1, col2 = st.columns(2)
            with col1:
                start_str = st.text_input("Start (x,y)", f"{line['start'][0]}, {line['start'][1]}", key=f"start_{i}")
                try:
                    line['start'] = tuple(map(int, start_str.split(',')))
                except:
                    st.error("Invalid coordinates format. Use 'x, y'")
            
            with col2:
                end_str = st.text_input("End (x,y)", f"{line['end'][0]}, {line['end'][1]}", key=f"end_{i}")
                try:
                    line['end'] = tuple(map(int, end_str.split(',')))
                except:
                    st.error("Invalid coordinates format. Use 'x, y'")
            
            # Line color
            color_idx = st.selectbox("Line Color", range(len(COLOR_NAMES)), 
                                   format_func=lambda idx: COLOR_NAMES[idx],
                                   index=line['color_index'],
                                   key=f"color_{i}")
            line['color_index'] = color_idx
            
            # Line enable/disable
            line['enabled'] = st.checkbox("Enable Line", line['enabled'], key=f"enabled_{i}")
            
            # Delete button
            if st.button("Delete Line", key=f"delete_{i}"):
                lines_to_delete.append(i)
    
    # Remove lines marked for deletion (in reverse to avoid index issues)
    for idx in sorted(lines_to_delete, reverse=True):
        if len(st.session_state.crossing_lines) > 1:  # Keep at least one line
            st.session_state.crossing_lines.pop(idx)
    
    # Total counts across all lines
    total_in = sum(line['in_count'] for line in st.session_state.crossing_lines)
    total_out = sum(line['out_count'] for line in st.session_state.crossing_lines)

    # Confidence threshold slider
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)

    tracker = Tracker()
    
    # Dictionary to store previous positions of objects
    previous_positions = {}
    
    # Dictionary to track which lines objects have crossed and in which direction
    # Format: {object_id_line_idx: (timestamp, direction)}
    crossed_lines = {}

    def is_crossing_line(point, line):
        """Check if a point is on the line side"""
        x, y = point
        (x1, y1), (x2, y2) = line
        
        # Calculate line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        # Calculate the value of the line equation at the point
        value = a * x + b * y + c
        
        # Return the sign of the value (positive or negative)
        return 1 if value > 0 else -1
        
    def calculate_direction(old_point, new_point, line_start, line_end):
        """Calculate the direction of movement relative to the line"""
        # Get the line vector
        line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])
        
        # Get the movement vector
        movement_vector = (new_point[0] - old_point[0], new_point[1] - old_point[1])
        
        # Calculate the cross product for direction
        cross_product = line_vector[0] * movement_vector[1] - line_vector[1] * movement_vector[0]
        
        # Determine the direction based on the cross product
        return "IN" if cross_product > 0 else "OUT"

    if mode == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame if it's too large to avoid CUDA memory issues
                orig_height, orig_width = frame.shape[:2]
                if orig_width > 1280 or orig_height > 720:
                    frame = cv2.resize(frame, (min(orig_width, 1280), min(orig_height, 720)))
                
                # Run inference with custom model
                custom_results = st.session_state.cement_model(frame, conf=confidence_threshold)[0]
                
                # Run inference with default YOLO model
                yolo_results = st.session_state.yolo_model(frame, conf=confidence_threshold)[0]
                
                boxes = []
                detection_info = {}  # Store detection info (class_id, confidence) for each box
                
                # Process custom model results
                for r in custom_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    # Only process if class is selected and confidence is above threshold
                    if score > confidence_threshold and int(class_id) in selected_class_ids:
                        box_key = f"{int(x1)}_{int(y1)}_{int(x2)-int(x1)}_{int(y2)-int(y1)}"
                        detection_info[box_key] = (int(class_id), float(score))
                        boxes.append([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)])
                
                # Process YOLO model results with offset to avoid ID conflicts
                offset = st.session_state.class_offset
                for r in yolo_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    # Apply offset to class_id to match our combined_names dictionary
                    offset_class_id = int(class_id) + offset
                    # Only process if class is selected and confidence is above threshold
                    if score > confidence_threshold and offset_class_id in selected_class_ids:
                        box_key = f"{int(x1)}_{int(y1)}_{int(x2)-int(x1)}_{int(y2)-int(y1)}"
                        detection_info[box_key] = (offset_class_id, float(score))
                        boxes.append([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)])
                bbox_idx = tracker.update(boxes)
                for x, y, w, h, id in bbox_idx:
                    # Calculate center of the bounding box
                    center_x = x + w // 2
                    center_y = y + h // 2
                    center = (center_x, center_y)
                    
                    # Check if we have seen this object before
                    if id in previous_positions:
                        # Check crossing for each enabled line
                        for line_idx, line in enumerate(st.session_state.crossing_lines):
                            if not line['enabled']:
                                continue
                                
                            prev_side = is_crossing_line(previous_positions[id], (line['start'], line['end']))
                            current_side = is_crossing_line(center, (line['start'], line['end']))
                            
                            # Create a unique key for this object-line pair
                            crossing_key = f"{id}_{line_idx}"
                            
                            # Only process if the object has crossed the line
                            if prev_side != current_side:
                                # Determine the direction using vector calculation
                                direction = calculate_direction(
                                    previous_positions[id], center, 
                                    line['start'], line['end']
                                )
                                
                                # Invert direction for Line 1 only
                                if line['name'] == 'Line 1':
                                    direction = "OUT" if direction == "IN" else "IN"
                                
                                # Get current time
                                current_time = time.time()
                                
                                # Check if object already crossed this line recently
                                if crossing_key in crossed_lines:
                                    last_cross_time, last_direction = crossed_lines[crossing_key]
                                    # Only count if it's been more than 1.5 seconds since last crossing
                                    # or the direction has changed
                                    if (current_time - last_cross_time > 1.5) or (direction != last_direction):
                                        if direction == "IN":
                                            line['in_count'] += 1
                                        else:
                                            line['out_count'] += 1
                                        # Update with new timestamp and direction
                                        crossed_lines[crossing_key] = (current_time, direction)
                                else:
                                    # First time crossing this line
                                    if direction == "IN":
                                        line['in_count'] += 1
                                    else:
                                        line['out_count'] += 1
                                    # Record the crossing
                                    crossed_lines[crossing_key] = (current_time, direction)
                    
                    # Update the previous position
                    previous_positions[id] = center
                    
                    # Draw bounding box and label with ID, class name, and confidence
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Get class name and confidence if available
                    box_key = f"{x}_{y}_{w}_{h}"
                    if box_key in detection_info:
                        class_id, conf = detection_info[box_key]
                        class_name = st.session_state.combined_names[class_id]
                        label = f"ID:{id} {class_name} {conf:.2f}"
                    else:
                        label = f"ID:{id}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw all crossing lines
                for line in st.session_state.crossing_lines:
                    if line['enabled']:
                        color = LINE_COLORS[line['color_index']]
                        cv2.line(frame, line['start'], line['end'], color, 2)
                        
                        # Add line name near the center of the line
                        line_center_x = (line['start'][0] + line['end'][0]) // 2
                        line_center_y = (line['start'][1] + line['end'][1]) // 2
                        cv2.putText(frame, line['name'], (line_center_x, line_center_y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Display total counts at the top
                cv2.putText(frame, f"TOTAL IN: {total_in}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"TOTAL OUT: {total_out}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"TOTAL NET: {total_in - total_out}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Display individual line counts
                y_offset = 120
                for line in st.session_state.crossing_lines:
                    if line['enabled']:
                        color = LINE_COLORS[line['color_index']]
                        net_count = line['in_count'] - line['out_count']
                        cv2.putText(frame, f"{line['name']} - IN: {line['in_count']} OUT: {line['out_count']} NET: {net_count}", 
                                  (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 30
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Clean up
            cap.release()
            if os.path.exists(tfile.name):
                try:
                    os.unlink(tfile.name)
                except Exception as e:
                    st.warning(f"Could not delete temporary file: {e}")

            st.success("Video processing complete.")
            response, status = send_to_inventory_system(selected_cluster, total_in, total_out)
            if status == 200:
                st.success("Inventory updated successfully.")
            else:
                st.error("Failed to update inventory.")

    elif mode == "RTSP Camera":
        rtsp_url = st.text_input("Enter RTSP URL", "")
        
        # Add FPS control and performance settings
        col1, col2 = st.columns(2)
        with col1:
            target_fps = st.slider("Target FPS", min_value=1, max_value=30, value=10, 
                               help="Lower values improve performance but reduce smoothness")
        with col2:
            processing_resolution = st.selectbox("Processing Resolution", 
                                            ["Very Low (480p)", "Low (720p)", "Medium (1080p)", "Original"], 
                                            index=1,
                                            help="Lower resolution improves performance")
        
        # Map resolution selection to actual dimensions
        resolution_map = {
            "Very Low (480p)": (640, 480),
            "Low (720p)": (1280, 720),
            "Medium (1080p)": (1920, 1080),
            "Original": None
        }
        selected_resolution = resolution_map[processing_resolution]
        
        if st.button("Start Stream", key="start_stream_btn") and rtsp_url:
            # Use lower buffer size and specific RTSP transport for better performance
            cap = cv2.VideoCapture(rtsp_url)
            
            # Try to set RTSP buffer size as low as possible to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                st.error(f"Failed to open RTSP stream: {rtsp_url}")
                return

            stframe = st.empty()
            stop_button = st.button("Stop Stream", key="stop_stream_btn_1")
            
            # Variables for frame rate control
            frame_count = 0
            skip_count = 0
            last_displayed = time.time()
            frame_interval = 1.0 / target_fps  # Time between frames
            
            while cap.isOpened() and not stop_button:
                # Use frame skipping for performance improvement
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot read from the stream anymore. Connection may have been lost.")
                    break
                
                # Check if enough time has passed to process this frame based on target FPS
                current_time = time.time()
                elapsed = current_time - last_displayed
                
                # Skip frames to maintain target FPS
                if elapsed < frame_interval:
                    skip_count += 1
                    continue
                
                frame_count += 1
                last_displayed = current_time
                
                # Resize frame based on selected resolution
                orig_height, orig_width = frame.shape[:2]
                if selected_resolution is not None and (orig_width > selected_resolution[0] or orig_height > selected_resolution[1]):
                    # Calculate aspect-ratio preserving dimensions
                    target_width, target_height = selected_resolution
                    ratio = min(target_width/orig_width, target_height/orig_height)
                    new_width, new_height = int(orig_width * ratio), int(orig_height * ratio)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                try:
                    # Run inference with custom model
                    custom_results = st.session_state.cement_model(frame, conf=confidence_threshold)[0]
                    
                    # Run inference with default YOLO model
                    yolo_results = st.session_state.yolo_model(frame, conf=confidence_threshold)[0]
                except Exception as e:
                    st.error(f"Error processing frame: {e}")
                    # Try to recover by continuing with next frame
                    continue
                    
                boxes = []
                detection_info = {}  # Store detection info (class_id, confidence) for each box
                
                # Process custom model results
                for r in custom_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    # Only process if class is selected and confidence is above threshold
                    if score > confidence_threshold and int(class_id) in selected_class_ids:
                        box_key = f"{int(x1)}_{int(y1)}_{int(x2)-int(x1)}_{int(y2)-int(y1)}"
                        detection_info[box_key] = (int(class_id), float(score))
                        boxes.append([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)])
                
                # Process YOLO model results with offset to avoid ID conflicts
                offset = st.session_state.class_offset
                for r in yolo_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    # Apply offset to class_id to match our combined_names dictionary
                    offset_class_id = int(class_id) + offset
                    # Only process if class is selected and confidence is above threshold
                    if score > confidence_threshold and offset_class_id in selected_class_ids:
                        box_key = f"{int(x1)}_{int(y1)}_{int(x2)-int(x1)}_{int(y2)-int(y1)}"
                        detection_info[box_key] = (offset_class_id, float(score))
                        boxes.append([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)])
                
                bbox_idx = tracker.update(boxes)
                for x, y, w, h, id in bbox_idx:
                    # Calculate center of the bounding box
                    center_x = x + w // 2
                    center_y = y + h // 2
                    center = (center_x, center_y)
                    
                    # Check if we have seen this object before
                    if id in previous_positions:
                        # Check crossing for each enabled line
                        for line_idx, line in enumerate(st.session_state.crossing_lines):
                            if not line['enabled']:
                                continue
                                
                            prev_side = is_crossing_line(previous_positions[id], (line['start'], line['end']))
                            current_side = is_crossing_line(center, (line['start'], line['end']))
                            
                            # Create a unique key for this object-line pair
                            crossing_key = f"{id}_{line_idx}"
                            
                            # Only process if the object has crossed the line
                            if prev_side != current_side:
                                # Determine the direction using vector calculation
                                direction = calculate_direction(
                                    previous_positions[id], center, 
                                    line['start'], line['end']
                                )
                                
                                # Invert direction for Line 1 only
                                if line['name'] == 'Line 1':
                                    direction = "OUT" if direction == "IN" else "IN"
                                
                                # Get current time
                                current_time = time.time()
                                
                                # Check if object already crossed this line recently
                                if crossing_key in crossed_lines:
                                    last_cross_time, last_direction = crossed_lines[crossing_key]
                                    # Only count if it's been more than 1.5 seconds since last crossing
                                    # or the direction has changed
                                    if (current_time - last_cross_time > 1.5) or (direction != last_direction):
                                        if direction == "IN":
                                            line['in_count'] += 1
                                        else:
                                            line['out_count'] += 1
                                        # Update with new timestamp and direction
                                        crossed_lines[crossing_key] = (current_time, direction)
                                else:
                                    # First time crossing this line
                                    if direction == "IN":
                                        line['in_count'] += 1
                                    else:
                                        line['out_count'] += 1
                                    # Record the crossing
                                    crossed_lines[crossing_key] = (current_time, direction)
                    
                    # Update the previous position
                    previous_positions[id] = center
                    
                    # Draw bounding box and label with ID, class name, and confidence
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Get class name and confidence if available
                    box_key = f"{x}_{y}_{w}_{h}"
                    if box_key in detection_info:
                        class_id, conf = detection_info[box_key]
                        class_name = st.session_state.combined_names[class_id]
                        label = f"ID:{id} {class_name} {conf:.2f}"
                    else:
                        label = f"ID:{id}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw all crossing lines
                for line in st.session_state.crossing_lines:
                    if line['enabled']:
                        color = LINE_COLORS[line['color_index']]
                        cv2.line(frame, line['start'], line['end'], color, 2)
                        
                        # Add line name near the center of the line
                        line_center_x = (line['start'][0] + line['end'][0]) // 2
                        line_center_y = (line['start'][1] + line['end'][1]) // 2
                        cv2.putText(frame, line['name'], (line_center_x, line_center_y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Update total counts
                total_in = sum(line['in_count'] for line in st.session_state.crossing_lines)
                total_out = sum(line['out_count'] for line in st.session_state.crossing_lines)
                
                # Display total counts at the top
                cv2.putText(frame, f"TOTAL IN: {total_in}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"TOTAL OUT: {total_out}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"TOTAL NET: {total_in - total_out}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Display individual line counts
                y_offset = 120
                for line in st.session_state.crossing_lines:
                    if line['enabled']:
                        color = LINE_COLORS[line['color_index']]
                        net_count = line['in_count'] - line['out_count']
                        cv2.putText(frame, f"{line['name']} - IN: {line['in_count']} OUT: {line['out_count']} NET: {net_count}", 
                                  (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 30
                
                # Show performance metrics on frame
                fps_text = f"Processing: {1/max(elapsed, 0.001):.1f} FPS | Target: {target_fps} FPS"
                cv2.putText(frame, fps_text, (10, orig_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Adaptive sleep based on processing time to maintain target FPS
                processing_time = time.time() - current_time
                sleep_time = max(0, frame_interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # We don't need a second stop button check - use the first one
                # The first stop_button variable is already checked in the while condition
                
            # Clean up resources
            cap.release()

if __name__ == "__main__":
    main()

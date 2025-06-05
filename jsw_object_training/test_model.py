import cv2
import json
import math
import numpy as np
import os
import random
import streamlit as st
import requests
import tempfile
import time
import json
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


# Set page config
st.set_page_config(
    page_title="Cement Bag Detection",
    page_icon="📦",
    layout="wide"
)

# Title
st.title("Cement Bag Detection System")
st.write("Upload images or videos to detect cement bags and trucks using YOLOv8 model")

# Load the models
@st.cache_resource
def load_models():
    # Use absolute path to avoid path issues
    cement_model = YOLO('F:/jsw20042025/best_cement_bags.pt')  
    # Disable tracking as we'll use a position-based approach
    # cement_model.tracker = "bytetrack.yaml"  
    truck_model = YOLO('yolov8n.pt')  # Using YOLOv8n for truck detection
    return cement_model, truck_model

cement_model, truck_model = load_models()

# Initialize session state
if 'lines' not in st.session_state:
    st.session_state.lines = []
if 'reference_image' not in st.session_state:
    st.session_state.reference_image = None
if 'image_dimensions' not in st.session_state:
    st.session_state.image_dimensions = None
if 'line_colors' not in st.session_state:
    st.session_state.line_colors = []
if 'crossing_stats' not in st.session_state:
    # Dictionary to hold line crossing statistics for each line
    st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
if 'saved_configs' not in st.session_state:
    st.session_state.saved_configs = {}
if 'enable_truck_detection' not in st.session_state:
    st.session_state.enable_truck_detection = False
if 'reference_dimensions' not in st.session_state:
    st.session_state.reference_dimensions = None
if 'last_frame_bags' not in st.session_state:
    # Track the positions of bags in the last frame
    st.session_state.last_frame_bags = []
if 'prev_frame_time' not in st.session_state:
    # Track the timestamp of the previous frame
    st.session_state.prev_frame_time = time.time()
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'available_clusters' not in st.session_state:
    st.session_state.available_clusters = []
if 'processed_files' not in st.session_state:
    # Track processed files to prevent reprocessing
    st.session_state.processed_files = {}

# Flask API configuration
FLASK_API_URL = "http://localhost:5000"

# Load available clusters from Flask API
def load_clusters():
    try:
        response = requests.get(f"{FLASK_API_URL}/clusters")
        clusters = response.json()
        st.session_state.available_clusters = clusters
        return clusters
    except Exception as e:
        st.error(f"Error loading clusters: {str(e)}")
        return []

# Send counting data to inventory system
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
            # Update existing cluster
            net_count = in_count - out_count
            data = {"bag_count": net_count}
            response = requests.put(f"{FLASK_API_URL}/clusters/{cluster_id}", json=data)
            return response.json(), 200
        else:
            # Create new cluster
            net_count = in_count - out_count
            data = {"name": cluster_name, "bag_count": net_count}
            response = requests.post(f"{FLASK_API_URL}/clusters", json=data)
            return response.json(), 201
    except Exception as e:
        st.error(f"Error connecting to inventory system: {str(e)}")
        return {"error": str(e)}, 500

# Add the Tracker class from tracker.py here for self-contained implementation
class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:  # If distance is less than threshold, consider as same object
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

# Initialize session state variables if needed
if 'crossing_stats' not in st.session_state:
    # Dictionary to hold line crossing statistics for each line
    st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})

if 'counted_objects' not in st.session_state:
    # Track which objects have been counted
    st.session_state.counted_objects = set()

if 'tracker' not in st.session_state:
    # Initialize tracker if it doesn't exist
    st.session_state.tracker = Tracker()

if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Line colors
LINE_COLORS = [
    (0, 255, 0),   # Green
    (255, 0, 0),   # Blue
    (0, 0, 255),   # Red
    (255, 255, 0), # Cyan
    (255, 0, 255), # Magenta
    (0, 255, 255), # Yellow
    (128, 0, 0),   # Dark Blue
    (0, 128, 0),   # Dark Green
    (0, 0, 128),   # Dark Red
    (128, 128, 0)  # Dark Yellow
]

def draw_lines(frame, lines, colors):
    """Draw lines with counts on an image"""
    frame_copy = frame.copy()
    
    for i, (line, color) in enumerate(zip(lines, colors)):
        # Draw the line
        cv2.line(frame_copy, line[0], line[1], color, 2)
        
        # Add line number and counts
        mid_point = ((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2)
        stats = st.session_state.crossing_stats[i]
        total = stats['in'] - stats['out']
        
        # Draw counts with background for better visibility
        text = f'Line {i+1} | IN: {stats["in"]} | OUT: {stats["out"]} | Total: {total}'
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        padding = 5
        bg_pt1 = (mid_point[0] - text_width//2 - padding, mid_point[1] - text_height - padding)
        bg_pt2 = (mid_point[0] + text_width//2 + padding, mid_point[1] + padding)
        cv2.rectangle(frame_copy, bg_pt1, bg_pt2, (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame_copy, text, 
                   (mid_point[0] - text_width//2, mid_point[1]),
                   font, font_scale, color, thickness)
                   
    return frame_copy

def normalize_lines(lines, frame_width, frame_height):
    """Normalize line coordinates based on actual frame dimensions"""
    if not lines:
        return lines
    
    normalized_lines = []
    
    # If reference dimensions don't exist or aren't properly formatted, return original lines
    if not st.session_state.reference_dimensions or not isinstance(st.session_state.reference_dimensions, tuple):
        return lines
    
    try:
        # Try to unpack reference dimensions safely
        if len(st.session_state.reference_dimensions) >= 2:
            ref_height, ref_width = st.session_state.reference_dimensions[0], st.session_state.reference_dimensions[1]
            scale_x = frame_width / ref_width
            scale_y = frame_height / ref_height
            
            for line in lines:
                start_point = (int(line[0][0] * scale_x), int(line[0][1] * scale_y))
                end_point = (int(line[1][0] * scale_x), int(line[1][1] * scale_y))
                normalized_lines.append([start_point, end_point])
            
            return normalized_lines
        else:
            return lines
    except (ValueError, TypeError, ZeroDivisionError):
        # Return original lines if any error occurs
        return lines

def check_line_crossing(point, line):
    """Check which side of a line a point is on"""
    # Line is defined as start_point, end_point
    x, y = point
    (x1, y1), (x2, y2) = line
    
    # Calculate the side (sign of the cross product)
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def get_movement_direction(prev_pos, current_pos, line):
    """Determine if the movement is from right to left or left to right relative to the line"""
    # Line direction vector
    line_start, line_end = line
    line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    
    # Movement vector
    movement_vector = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
    
    # Calculate the dot product to determine if movement is similar to line direction
    dot_product = line_vector[0] * movement_vector[0] + line_vector[1] * movement_vector[1]
    
    # If dot product is positive, movement is similar to line direction (left to right)
    # If negative, movement is opposite (right to left)
    return 'right_to_left' if dot_product < 0 else 'left_to_right'

def process_frame(frame, lines, colors):
    # Make a copy of the frame for drawing
    frame_with_lines = frame.copy()
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Draw the lines directly without normalization
    for i, line in enumerate(lines):
        if i < len(colors):
            color = colors[i]
        else:
            color = (0, 255, 0)  # Default to green
            
        # Draw the line
        cv2.line(frame_with_lines, line[0], line[1], color, 2)
        
        # Add counter text near line
        stats = st.session_state.crossing_stats[i]
        text = f'Line {i+1} | IN: {stats["in"]} | OUT: {stats["out"]} | Total: {stats["in"] - stats["out"]}'
        mid_point = ((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2)
        cv2.putText(frame_with_lines, text, (mid_point[0] - 100, mid_point[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Get detections using the model
    results = cement_model.predict(frame, conf=0.2, verbose=True)
    
    # Extract detection boxes
    detection_boxes = []
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            if box.conf > 0.2:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                detection_boxes.append([x1, y1, w, h])
                
                # Draw the detection box
                cv2.rectangle(frame_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_with_lines, f'Cement Bag', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Use the tracker to get consistent IDs for bags (like in tracker.py)
    if 'tracker' not in st.session_state:
        # Initialize tracker if it doesn't exist
        st.session_state.tracker = Tracker()
        
    # Update tracker with current detections
    tracked_objects = st.session_state.tracker.update(detection_boxes)
    
    # Process tracked objects
    for obj in tracked_objects:
        x, y, w, h, object_id = obj
        cx = x + w // 2
        cy = y + h // 2
        
        # Draw center point and ID
        cv2.circle(frame_with_lines, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame_with_lines, str(object_id), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Check for line crossing (similar to Packer2D4Prog.py)
        for i, line in enumerate(lines):
            # Define the line
            x1, y1 = line[0]
            x2, y2 = line[1]
            
            # Create a buffer zone around the line
            offset = 5
            
            # Check if the bag's center is near the line
            # Simplification: check if point is near line segment using point-to-line distance
            # Calculate distance from point to line segment
            line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length == 0:  # Avoid division by zero
                continue
                
            # Calculate distance from point to line
            dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / line_length
            
            # If bag is close to the line and hasn't been counted yet
            if dist < offset:
                if object_id not in st.session_state.counted_objects:
                    st.session_state.counted_objects.add(object_id)
                    
                    # Determine direction (simplified for horizontal/vertical lines)
                    if abs(y2 - y1) > abs(x2 - x1):  # More vertical line
                        if cy < y1:  # Moving from top to bottom
                            st.session_state.crossing_stats[i]['out'] += 1
                            print(f'🔵 COUNTING OUT: Bag {object_id} crossed line {i+1}')
                            cv2.putText(frame_with_lines, 'OUT', (cx - 20, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        else:  # Moving from bottom to top
                            st.session_state.crossing_stats[i]['in'] += 1
                            print(f'🔴 COUNTING IN: Bag {object_id} crossed line {i+1}')
                            cv2.putText(frame_with_lines, 'IN', (cx - 20, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:  # More horizontal line
                        if cx < x1:  # Moving from left to right
                            st.session_state.crossing_stats[i]['in'] += 1
                            print(f'🔴 COUNTING IN: Bag {object_id} crossed line {i+1}')
                            cv2.putText(frame_with_lines, 'IN', (cx - 20, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:  # Moving from right to left
                            st.session_state.crossing_stats[i]['out'] += 1
                            print(f'🔵 COUNTING OUT: Bag {object_id} crossed line {i+1}')
                            cv2.putText(frame_with_lines, 'OUT', (cx - 20, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Calculate and display overall counts on the frame
    total_in = sum(stats['in'] for stats in st.session_state.crossing_stats.values())
    total_out = sum(stats['out'] for stats in st.session_state.crossing_stats.values())
    total_net = total_in - total_out
    
    # Add totals at the top of the frame
    cv2.putText(frame_with_lines, f"Total IN: {total_in}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame_with_lines, f"Total OUT: {total_out}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame_with_lines, f"NET Total: {total_net}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame_with_lines

# Flask API configuration
FLASK_API_URL = "http://localhost:5000"

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
            # Update existing cluster
            net_count = in_count - out_count
            data = {"bag_count": net_count}
            response = requests.put(f"{FLASK_API_URL}/clusters/{cluster_id}", json=data)
            return response.json(), 200
        else:
            # Create new cluster
            net_count = in_count - out_count
            data = {"name": cluster_name, "bag_count": net_count}
            response = requests.post(f"{FLASK_API_URL}/clusters", json=data)
            return response.json(), 201
    except Exception as e:
        st.error(f"Error connecting to inventory system: {str(e)}")
        return {"error": str(e)}, 500

# Line configuration interface
st.sidebar.title("Line Configuration")
line_mode = st.sidebar.radio("Choose line input method:", ["Draw on reference image", "Upload reference image with lines"])

# Load saved configurations
if os.path.exists('line_configs.json'):
    with open('line_configs.json', 'r') as f:
        st.session_state.saved_configs = json.load(f)

# Configuration management
st.sidebar.write("### Saved Configurations")
config_name = st.sidebar.text_input("Configuration Name")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Save Current Config"):
        if config_name:
            save_line_config(config_name, st.session_state.lines, st.session_state.line_colors)
            st.sidebar.success(f"Configuration '{config_name}' saved!")
with col2:
    if st.session_state.saved_configs:
        selected_config = st.selectbox("Load Configuration", list(st.session_state.saved_configs.keys()))
        if st.button("Load Selected Config"):
            load_line_config(selected_config)

if line_mode == "Draw on reference image":
    st.write("### Draw Lines on Reference Image")
    reference_file = st.file_uploader("Upload a reference image to draw lines", type=['jpg', 'jpeg', 'png'])
    
    if reference_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(reference_file.name).suffix) as tmp_file:
            tmp_file.write(reference_file.getvalue())
            tmp_file_path = tmp_file.name
        
        reference_image = cv2.imread(tmp_file_path)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        st.session_state.image_dimensions = reference_image.shape
        st.session_state.reference_dimensions = reference_image.shape  # Store reference dimensions
        if st.session_state.debug_mode:
            st.write(f"[DEBUG] reference_dimensions set from DRAW: {st.session_state.reference_dimensions}")
        
        st.image(reference_image, caption="Draw lines on this image", use_column_width=True)
        
        num_lines = st.number_input("Number of lines to draw", min_value=1, max_value=10, value=1)
        
        if len(st.session_state.lines) != num_lines:
            st.session_state.lines = []
            st.session_state.line_colors = []
        
        for i in range(num_lines):
            if i >= len(st.session_state.lines):
                st.session_state.lines.append([(0, 0), (reference_image.shape[1], reference_image.shape[0])])
                st.session_state.line_colors.append(LINE_COLORS[i % len(LINE_COLORS)])
            
            with st.expander(f"Line {i+1} Settings", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Start Point")
                    x1 = st.number_input(f"Start X {i+1}", min_value=0, max_value=reference_image.shape[1], value=st.session_state.lines[i][0][0], key=f"x1_{i}")
                    y1 = st.number_input(f"Start Y {i+1}", min_value=0, max_value=reference_image.shape[0], value=st.session_state.lines[i][0][1], key=f"y1_{i}")
                with col2:
                    st.write("End Point")
                    x2 = st.number_input(f"End X {i+1}", min_value=0, max_value=reference_image.shape[1], value=st.session_state.lines[i][1][0], key=f"x2_{i}")
                    y2 = st.number_input(f"End Y {i+1}", min_value=0, max_value=reference_image.shape[0], value=st.session_state.lines[i][1][1], key=f"y2_{i}")
                
                # Color selection
                color_idx = LINE_COLORS.index(st.session_state.line_colors[i])
                new_color_idx = st.selectbox(f"Line {i+1} Color", range(len(LINE_COLORS)), 
                                           index=color_idx, format_func=lambda x: f"Color {x+1}")
                st.session_state.line_colors[i] = LINE_COLORS[new_color_idx]
                
                st.session_state.lines[i] = [(x1, y1), (x2, y2)]
        
        image_with_lines = draw_lines(reference_image.copy(), st.session_state.lines, st.session_state.line_colors)
        st.image(image_with_lines, caption="Image with lines", use_column_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Lines"):
                st.session_state.lines = []
                st.session_state.line_colors = []
                st.session_state.crossing_stats.clear()
                st.session_state.crossed_bags.clear()
                st.session_state.bag_positions.clear()
                st.experimental_rerun()
        with col2:
            if st.button("Reset Line Positions"):
                st.session_state.lines = [[(0, 0), (reference_image.shape[1], reference_image.shape[0])] for _ in range(num_lines)]
                st.session_state.line_colors = [LINE_COLORS[i % len(LINE_COLORS)] for i in range(num_lines)]
                st.session_state.crossing_stats.clear()
                st.session_state.crossed_bags.clear()
                st.session_state.bag_positions.clear()
                st.experimental_rerun()
        
        os.unlink(tmp_file_path)

elif line_mode == "Upload reference image with lines":
    st.write("### Upload Reference Image with Lines")
    reference_file = st.file_uploader("Upload an image with drawn lines", type=['jpg', 'jpeg', 'png'])
    
    if reference_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(reference_file.name).suffix) as tmp_file:
            tmp_file.write(reference_file.getvalue())
            tmp_file_path = tmp_file.name
        
        reference_image = cv2.imread(tmp_file_path)
        st.session_state.reference_dimensions = reference_image.shape
        if st.session_state.debug_mode:
            st.write(f"[DEBUG] reference_dimensions set from UPLOAD: {st.session_state.reference_dimensions}")
        gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            st.session_state.lines = []
            st.session_state.line_colors = []
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                st.session_state.lines.append([(x1, y1), (x2, y2)])
                st.session_state.line_colors.append(LINE_COLORS[i % len(LINE_COLORS)])
        
        image_with_lines = draw_lines(reference_image.copy(), st.session_state.lines, st.session_state.line_colors)
        st.image(image_with_lines, caption="Detected lines", use_column_width=True)
        
        if len(st.session_state.lines) > 0:
            st.write("### Adjust Detected Lines")
            for i, (line, color) in enumerate(zip(st.session_state.lines, st.session_state.line_colors)):
                with st.expander(f"Line {i+1} Settings", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Start Point")
                        x1 = st.slider(f"Start X {i+1}", 0, reference_image.shape[1], line[0][0], key=f"x1_{i}")
                        y1 = st.slider(f"Start Y {i+1}", 0, reference_image.shape[0], line[0][1], key=f"y1_{i}")
                    with col2:
                        st.write("End Point")
                        x2 = st.slider(f"End X {i+1}", 0, reference_image.shape[1], line[1][0], key=f"x2_{i}")
                        y2 = st.slider(f"End Y {i+1}", 0, reference_image.shape[0], line[1][1], key=f"y2_{i}")
                    
                    color_idx = LINE_COLORS.index(color)
                    new_color_idx = st.selectbox(f"Line {i+1} Color", range(len(LINE_COLORS)), 
                                               index=color_idx, format_func=lambda x: f"Color {x+1}")
                    st.session_state.line_colors[i] = LINE_COLORS[new_color_idx]
                    
                    st.session_state.lines[i] = [(x1, y1), (x2, y2)]
            
            updated_image = draw_lines(reference_image.copy(), st.session_state.lines, st.session_state.line_colors)
            st.image(updated_image, caption="Updated lines", use_column_width=True)
        
        os.unlink(tmp_file_path)



# Add detection settings in sidebar
st.sidebar.title("Detection Settings")
st.session_state.enable_truck_detection = st.sidebar.checkbox("Enable Truck Detection", value=st.session_state.enable_truck_detection)
st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode (Show Crossings)", value=st.session_state.debug_mode)

# Cluster selection
st.sidebar.title("Inventory System")
if st.sidebar.button("Refresh Clusters"):
    clusters = load_clusters()
else:
    # Try to load clusters if not already loaded
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

    if selected == "+ Create New Cluster":
        new_cluster_name = st.sidebar.text_input("New Cluster Name")
        if st.sidebar.button("Create Cluster") and new_cluster_name:
            try:
                response = requests.post(f"{FLASK_API_URL}/clusters", json={"name": new_cluster_name, "bag_count": 0})
                if response.status_code in (200, 201):
                    st.sidebar.success(f"Created cluster '{new_cluster_name}'")
                    # Reload clusters
                    clusters = load_clusters()
                    st.experimental_rerun()
                else:
                    st.sidebar.error(f"Failed to create cluster: {response.text}")
            except Exception as e:
                st.sidebar.error(f"Error creating cluster: {str(e)}")
    elif selected:
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
                st.experimental_rerun()
            else:
                st.sidebar.error(f"Failed to create cluster: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error creating cluster: {str(e)}")

# Main video/image processing
if len(st.session_state.lines) > 0 and st.session_state.selected_cluster:
    st.write(f"### Process Video/Image for Cluster: {st.session_state.selected_cluster['name']}")
    
    uploaded_file = st.file_uploader("Choose a video or image file to process", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        is_video = uploaded_file.type.startswith('video/')
        
        # Check if file was already processed (using filename and size as identifier)
        file_id = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
        already_processed = file_id in st.session_state.processed_files
        
        # Flag to track if we need to update inventory
        update_needed = False
        
        if is_video:
            if already_processed:
                st.warning(f"⚠️ This video has already been processed. Re-processing will not update inventory again.")
                if st.button("Process again anyway"):
                    # Clear the existing stats before reprocessing
                    st.session_state.crossing_stats = defaultdict(lambda: {'in': 0, 'out': 0})
                    st.write("Processing video...")
                    cap = cv2.VideoCapture(tmp_file_path)
                else:
                    st.stop()
            else:
                st.write("Processing video...")
                cap = cv2.VideoCapture(tmp_file_path)
            
            
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            output_path = 'output_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frame = process_frame(frame, st.session_state.lines, st.session_state.line_colors)
                out.write(processed_frame)
                
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
            
            cap.release()
            out.release()
            
            # Display processed video with prominent UI elements
            st.write("### Processed Video Result")
            st.video(output_path, start_time=0)
            
            # Add video playback controls and information
            with st.expander("Video Playback Information", expanded=True):
                st.info(f"Video processed at {fps} FPS with {frame_count} total frames")
                st.info(f"Resolution: {width}x{height} pixels")
                st.write("Use the video player controls above to play, pause, and seek through the video.")
            
            # Display simple statistics totals
            st.write("### Bag Movement Summary")
            
            # Calculate totals
            total_in = sum(stats['in'] for stats in st.session_state.crossing_stats.values())
            total_out = sum(stats['out'] for stats in st.session_state.crossing_stats.values())
            net_total = total_in - total_out
            
            # Display only the totals in a clean format
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total IN", total_in, delta=None, delta_color="normal")
            with col2:
                st.metric("Total OUT", total_out, delta=None, delta_color="normal")
            with col3:
                st.metric("NET Change", net_total, delta=None, delta_color="normal")
            
            # Automatically send data to inventory system
            cluster_info = st.session_state.selected_cluster
            st.write(f"Selected Cluster: **{cluster_info['name']}** (ID: {cluster_info['id']})")
            
            # Auto-update the inventory when video is done processing
            with st.spinner("Automatically updating inventory system..."):
                try:
                    # Get all clusters and find the target one by ID
                    get_response = requests.get(f"{FLASK_API_URL}/clusters")
                    if get_response.status_code == 200:
                        all_clusters = get_response.json()
                        current_cluster = None
                        
                        # Find our cluster by ID
                        for cluster in all_clusters:
                            if cluster['id'] == cluster_info['id']:
                                current_cluster = cluster
                                break
                        
                        if current_cluster:
                            # Get current count from the cluster
                            current_count = current_cluster.get('bag_count', 0)
                            
                            # Calculate new count: current - out + in
                            new_count = current_count - total_out + total_in
                            
                            # Create proper movement records for IN and OUT
                            if total_in > 0:
                                in_data = {"movement_type": "IN", "quantity": total_in}
                                in_response = requests.post(
                                    f"{FLASK_API_URL}/clusters/{cluster_info['id']}/movement",
                                    json=in_data
                                )
                                
                            if total_out > 0:
                                out_data = {"movement_type": "OUT", "quantity": total_out}
                                out_response = requests.post(
                                    f"{FLASK_API_URL}/clusters/{cluster_info['id']}/movement",
                                    json=out_data
                                )
                                
                            movement_success = True
                            if total_in > 0 and in_response.status_code not in (200, 201):
                                movement_success = False
                            if total_out > 0 and out_response.status_code not in (200, 201):
                                movement_success = False
                                
                            if movement_success:
                                st.success(f"✅ Updated inventory for {cluster_info['name']}")
                                st.info(f"Previous count: {current_count} bags")
                                st.info(f"Out quantity: {total_out} bags")
                                st.info(f"In quantity: {total_in} bags")
                                st.info(f"New count: {new_count} bags")
                            else:
                                st.error(f"Failed to update inventory: {response.text}")
                                if st.button("Retry Update"):
                                    # Try again with proper movement records
                                    retry_success = True
                                    if total_in > 0:
                                        in_response = requests.post(
                                            f"{FLASK_API_URL}/clusters/{cluster_info['id']}/movement",
                                            json={"movement_type": "IN", "quantity": total_in}
                                        )
                                        if in_response.status_code not in (200, 201):
                                            retry_success = False
                                    
                                    if total_out > 0:
                                        out_response = requests.post(
                                            f"{FLASK_API_URL}/clusters/{cluster_info['id']}/movement",
                                            json={"movement_type": "OUT", "quantity": total_out}
                                        )
                                        if out_response.status_code not in (200, 201):
                                            retry_success = False
                                    if retry_success:
                                        st.success("✅ Update successful on retry!")
                                    else:
                                        st.error("Retry failed.")
                        else:
                            st.error(f"Could not find cluster with ID {cluster_info['id']} in the response")
                    else:
                        st.error(f"Failed to get clusters list: {get_response.text}")
                except Exception as e:
                    st.error(f"Error updating inventory: {str(e)}")
                    if st.button("Retry Update"):
                        try:
                            # Fallback to simpler update if detailed one fails
                            get_response = requests.get(f"{FLASK_API_URL}/clusters")
                            if get_response.status_code == 200:
                                for cluster in get_response.json():
                                    if cluster['id'] == cluster_info['id']:
                                        current_count = cluster.get('bag_count', 0)
                                        new_count = current_count - total_out + total_in
                                        movement_success = True
                                        # Record IN movement first if any
                                        if total_in > 0:
                                            in_response = requests.post(
                                                f"{FLASK_API_URL}/clusters/{cluster_info['id']}/movement",
                                                json={"movement_type": "IN", "quantity": total_in}
                                            )
                                            if in_response.status_code not in (200, 201):
                                                movement_success = False
                                                
                                        # Record OUT movement if any
                                        if total_out > 0 and movement_success:
                                            out_response = requests.post(
                                                f"{FLASK_API_URL}/clusters/{cluster_info['id']}/movement",
                                                json={"movement_type": "OUT", "quantity": total_out}
                                            )
                                            if out_response.status_code not in (200, 201):
                                                movement_success = False
                                        if movement_success:
                                            st.success(f"✅ Successfully updated inventory on retry!")
                                        else:
                                            st.error("Retry failed.")
                                        break
                        except Exception as e2:
                            st.error(f"Retry failed: {str(e2)}")
            
            # Provide download option under the video player
            st.write("### Save Processed Video")
            # Mark file as processed to avoid reprocessing
            if not already_processed:
                st.session_state.processed_files[file_id] = {
                    'filename': uploaded_file.name,
                    'timestamp': time.time(),
                    'stats': dict(st.session_state.crossing_stats)
                }
                
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="Download processed video",
                    data=f,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            
        else:
            st.write("Processing image...")
            
            image = cv2.imread(tmp_file_path)
            processed_image = process_frame(image, st.session_state.lines, st.session_state.line_colors)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col2:
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                
                output_path = 'processed_image.jpg'
                cv2.imwrite(output_path, processed_image)
                
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="Download processed image",
                        data=f,
                        file_name="processed_image.jpg",
                        mime="image/jpeg"
                    )
        
        os.unlink(tmp_file_path)

# Add some information about the model
st.sidebar.title("Model Information")
st.sidebar.write("Models: YOLOv8")
st.sidebar.write("Features:")
st.sidebar.write("- Cement Bag Detection")
st.sidebar.write("- Truck Detection")
st.sidebar.write("- Custom Line Detection")
st.sidebar.write("- Line Configuration Saving")
st.sidebar.write("- Detailed Crossing Statistics")
st.sidebar.write("Performance:")
st.sidebar.write("- mAP50: 98.49%")
st.sidebar.write("- Precision: 96.88%")
st.sidebar.write("- Recall: 96.99%") 
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import math
import os
import json
import io
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from tracker import Tracker
from PIL import Image, ImageDraw

app = Flask(__name__)

# Global variables
cement_model = None
yolo_model = None
combined_names = {}
class_offset = 100
tracker = Tracker()
previous_positions = {}
crossed_lines = {}
crossing_lines = [
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

# Flask API URL for inventory system
FLASK_API_URL = "http://localhost:5000"

def load_cement_model():
    try:
        model = YOLO('best_cement_bags_2025-05-29.pt')
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model(test_img)
    except Exception as e:
        print(f"GPU acceleration not available for cement model, falling back to CPU: {str(e)}")
        model = YOLO('best_cement_bags_2025-05-29.pt', device='cpu')
    return model

def load_yolo_model():
    try:
        model = YOLO('yolov8n.pt')
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model(test_img)
    except Exception as e:
        print(f"GPU acceleration not available for YOLO model, falling back to CPU: {str(e)}")
        model = YOLO('yolov8n.pt', device='cpu')
    return model

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

def load_models():
    global cement_model, yolo_model, combined_names, class_offset
    cement_model = load_cement_model()
    yolo_model = load_yolo_model()
    combined_names, class_offset = get_combined_names(cement_model, yolo_model)
    return cement_model

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

def process_frame(frame, confidence_threshold=0.3, selected_class_ids=None):
    global previous_positions, crossed_lines, crossing_lines
    
    if selected_class_ids is None:
        # If no class IDs specified, use all available
        selected_class_ids = list(combined_names.keys())
    
    # Resize frame if it's too large to avoid CUDA memory issues
    orig_height, orig_width = frame.shape[:2]
    if orig_width > 1280 or orig_height > 720:
        frame = cv2.resize(frame, (min(orig_width, 1280), min(orig_height, 720)))
    
    # Run inference with custom model
    custom_results = cement_model(frame, conf=confidence_threshold)[0]
    
    # Run inference with default YOLO model
    yolo_results = yolo_model(frame, conf=confidence_threshold)[0]
    
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
    for r in yolo_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        # Apply offset to class_id to match our combined_names dictionary
        offset_class_id = int(class_id) + class_offset
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
            for line_idx, line in enumerate(crossing_lines):
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
            class_name = combined_names[class_id]
            label = f"ID:{id} {class_name} {conf:.2f}"
        else:
            label = f"ID:{id}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw all crossing lines
    for line in crossing_lines:
        if line['enabled']:
            color = LINE_COLORS[line['color_index']]
            cv2.line(frame, line['start'], line['end'], color, 2)
            
            # Add line name near the center of the line
            line_center_x = (line['start'][0] + line['end'][0]) // 2
            line_center_y = (line['start'][1] + line['end'][1]) // 2
            cv2.putText(frame, line['name'], (line_center_x, line_center_y - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Update total counts
    total_in = sum(line['in_count'] for line in crossing_lines)
    total_out = sum(line['out_count'] for line in crossing_lines)
    
    # Display total counts at the top
    cv2.putText(frame, f"TOTAL IN: {total_in}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"TOTAL OUT: {total_out}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"TOTAL NET: {total_in - total_out}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Display individual line counts
    y_offset = 120
    for line in crossing_lines:
        if line['enabled']:
            color = LINE_COLORS[line['color_index']]
            net_count = line['in_count'] - line['out_count']
            cv2.putText(frame, f"{line['name']} - IN: {line['in_count']} OUT: {line['out_count']} NET: {net_count}", 
                      (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
    
    return frame

def gen_frames(source="0", confidence=0.3, selected_class_ids=None):
    """Video streaming generator function."""
    cap = cv2.VideoCapture(source)
    
    # Try to set RTSP buffer size as low as possible to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    
    target_fps = 10
    frame_interval = 1.0 / target_fps
    last_processed = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Control frame rate for processing
        current_time = time.time()
        elapsed = current_time - last_processed
        
        if elapsed < frame_interval:
            continue
        
        last_processed = current_time
        
        # Process the frame
        processed_frame = process_frame(frame, confidence, selected_class_ids)
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Adaptive sleep to maintain target FPS
        processing_time = time.time() - current_time
        sleep_time = max(0, frame_interval - processing_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    cap.release()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/modern')
def modern_ui():
    """Modern UI version of the application."""
    return render_template('modern_ui.html')

@app.route('/static/img/placeholder.jpg')
def placeholder_image():
    """Serve a placeholder image when none exists."""
    # Create a simple placeholder image using PIL
    
    # Create a blank image with a light gray background
    img = Image.new('RGB', (640, 360), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Add text
    draw.text((320, 180), "No video feed available", fill=(150, 150, 150))
    
    # Convert to bytes
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    
    return Response(img_io.getvalue(), mimetype='image/jpeg')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    rtsp_url = request.args.get('rtsp_url', '0')  # Default to webcam if no URL provided
    confidence = float(request.args.get('confidence', '0.3'))
    
    # Get selected class IDs from query parameters
    selected_classes = request.args.get('classes', '')
    selected_class_ids = [int(id) for id in selected_classes.split(',')] if selected_classes else None
    
    return Response(gen_frames(rtsp_url, confidence, selected_class_ids),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/classes')
def get_classes():
    """Return all available classes."""
    return jsonify(combined_names)

@app.route('/api/lines', methods=['GET'])
def get_lines():
    """Return all crossing lines."""
    return jsonify(crossing_lines)

@app.route('/api/lines', methods=['POST'])
def update_lines():
    """Update crossing lines configuration."""
    global crossing_lines
    data = request.json
    if data and isinstance(data, list):
        crossing_lines = data
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid data format"}), 400

@app.route('/api/counts')
def get_counts():
    """Return current counts."""
    total_in = sum(line['in_count'] for line in crossing_lines)
    total_out = sum(line['out_count'] for line in crossing_lines)
    
    return jsonify({
        "total_in": total_in,
        "total_out": total_out,
        "total_net": total_in - total_out,
        "lines": crossing_lines
    })

@app.route('/api/reset_counts', methods=['POST'])
def reset_counts():
    """Reset all counts."""
    global crossing_lines
    for line in crossing_lines:
        line['in_count'] = 0
        line['out_count'] = 0
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Load models before starting the app
    load_models()
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create static directory for CSS and JS if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('static/css'):
        os.makedirs('static/css')
    if not os.path.exists('static/js'):
        os.makedirs('static/js')
    if not os.path.exists('static/img'):
        os.makedirs('static/img')
    
    # Install required packages if not already installed
    try:
        import PIL
    except ImportError:
        print("Installing Pillow package...")
        os.system("pip install pillow")
    
    print("\n" + "="*50)
    print("JSW Cement Bag Detection System")
    print("="*50)
    print("Access the application at:")
    print("  Original UI: http://localhost:8080/")
    print("  Modern UI:   http://localhost:8080/modern")
    print("="*50 + "\n")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)

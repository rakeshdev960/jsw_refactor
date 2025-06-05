import cv2
import numpy as np
import time
import threading
import os
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from flask import jsonify
# Import line counter
from line_counter import get_counter

# Path to the YOLOv8 cement bag detection model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../jsw_object_training/best_cement_bags_2025-05-29.pt'))

class BagDetector:
    """
    Class to detect cement bags in camera feeds using YOLOv8
    """
    def __init__(self, confidence_threshold=0.5):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.camera_feeds = {}  # {camera_name: {'url': rtsp_url, 'detection_data': {...}}}
        self.detection_results = {}  # {camera_name: {'timestamp': time, 'bags_detected': count, 'confidence_scores': [], 'frame': img_bytes}}
        self.processing_threads = {}  # {camera_name: thread}
        self.stop_threads = False
        self.load_model()
        
        # Initialize the line counter
        self.line_counter = get_counter()
        
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            print(f"Loading YOLOv8 model from {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            
    def add_camera_feed(self, camera_name, camera_url):
        """Add a camera feed for bag detection"""
        if camera_name in self.camera_feeds:
            return False, f"Camera {camera_name} already exists"
        
        self.camera_feeds[camera_name] = {
            'url': camera_url,
            'detection_enabled': False,
            'last_frame': None,
            'detection_data': {
                'timestamp': None,
                'bags_detected': 0,
                'confidence_scores': [],
                'detection_boxes': []  # List of [x1, y1, x2, y2] coordinates
            }
        }
        return True, f"Camera {camera_name} added successfully"
            
    def start_detection(self, camera_name):
        """Start bag detection for a camera feed"""
        if camera_name not in self.camera_feeds:
            return False, f"Camera {camera_name} not found"
        
        if self.camera_feeds[camera_name]['detection_enabled']:
            return False, f"Detection already running for camera {camera_name}"
            
        if camera_name in self.processing_threads and self.processing_threads[camera_name].is_alive():
            return False, f"Detection thread already exists for camera {camera_name}"
            
        self.camera_feeds[camera_name]['detection_enabled'] = True
        self.processing_threads[camera_name] = threading.Thread(
            target=self._process_camera_feed,
            args=(camera_name,),
            daemon=True
        )
        self.processing_threads[camera_name].start()
        return True, f"Detection started for camera {camera_name}"
            
    def stop_detection(self, camera_name=None):
        """Stop bag detection for a specific or all camera feeds"""
        if camera_name:
            if camera_name not in self.camera_feeds:
                return False, f"Camera {camera_name} not found"
                
            self.camera_feeds[camera_name]['detection_enabled'] = False
            if camera_name in self.processing_threads and self.processing_threads[camera_name].is_alive():
                # Let the thread exit naturally in next iteration
                return True, f"Detection stopping for camera {camera_name}"
            return False, f"No detection running for camera {camera_name}"
        else:
            # Stop all detections
            for cam in self.camera_feeds:
                self.camera_feeds[cam]['detection_enabled'] = False
            return True, f"Detection stopping for all cameras"
    
    def get_detection_results(self, camera_name=None):
        """Get detection results for a specific or all camera feeds"""
        if camera_name:
            if camera_name not in self.camera_feeds:
                return None
            
            detection_data = self.camera_feeds[camera_name]['detection_data']
            return {
                'camera_name': camera_name,
                'timestamp': detection_data['timestamp'],
                'bags_detected': detection_data['bags_detected'],
                'detection_enabled': self.camera_feeds[camera_name]['detection_enabled'],
                'detection_status': 'running' if self.camera_feeds[camera_name]['detection_enabled'] else 'stopped',
            }
        else:
            # Return results for all cameras
            results = {}
            for cam_name in self.camera_feeds:
                detection_data = self.camera_feeds[cam_name]['detection_data']
                results[cam_name] = {
                    'timestamp': detection_data['timestamp'],
                    'bags_detected': detection_data['bags_detected'],
                    'detection_enabled': self.camera_feeds[cam_name]['detection_enabled'],
                    'detection_status': 'running' if self.camera_feeds[cam_name]['detection_enabled'] else 'stopped',
                }
            return results
    
    def _process_camera_feed(self, camera_name):
        """Process camera feed and detect cement bags"""
        if not self.model and self.camera_feeds[camera_name]['detection_enabled']:
            print(f"Model not loaded, cannot start detection for {camera_name}")
            self.camera_feeds[camera_name]['detection_enabled'] = False
        
        camera_url = self.camera_feeds[camera_name]['url']
        print(f"Starting camera feed for {camera_name} at {camera_url}")
        
        # Try to open the camera feed with timeout
        cap = cv2.VideoCapture(camera_url)
        
        if not cap.isOpened():
            print(f"Failed to open camera feed at {camera_url}")
            self.camera_feeds[camera_name]['detection_enabled'] = False
            return
            
        # Set resolution for processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Processing loop
        frames_to_skip = 5  # Only process every 5th frame to reduce CPU usage
        frame_count = 0
        frames_buffer = deque(maxlen=10)  # Keep last 10 detection frames
        
        # Flag to keep the camera thread running even if detection is disabled
        keep_running = True
        
        try:
            while keep_running:
                # Check if we should keep this camera feed running
                if camera_name not in self.camera_feeds:
                    keep_running = False
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read frame from camera {camera_name}")
                    # Try to reconnect
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(camera_url)
                    continue
                
                # Store the last frame regardless of detection
                self.camera_feeds[camera_name]['last_frame'] = frame.copy()
                
                # Only perform detection if enabled
                if self.camera_feeds[camera_name]['detection_enabled'] and self.model:
                    # Skip frames to reduce CPU usage
                    frame_count += 1
                    if frame_count % frames_to_skip != 0:
                        continue
                        
                    # Perform detection
                    start_time = time.time()
                    results = self.model(frame)
                    inference_time = time.time() - start_time
                    
                    # Extract detection results
                    detections = results[0]
                    
                    # Filter detections for cement bags
                    bags = []
                    confidence_scores = []
                    detection_boxes = []
                    
                    if len(detections) > 0:
                        boxes = detections.boxes.xyxy.cpu().numpy()
                        confidences = detections.boxes.conf.cpu().numpy()
                        class_ids = detections.boxes.cls.cpu().numpy().astype(int)

                        for box, confidence, class_id in zip(boxes, confidences, class_ids):
                            if confidence >= self.confidence_threshold:
                                x1, y1, x2, y2 = box.astype(int)
                                bags.append([x1, y1, x2, y2, confidence, class_id])
                                confidence_scores.append(float(confidence))
                                detection_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    
                    # Update detection data
                    self.camera_feeds[camera_name]['detection_data'] = {
                        'timestamp': datetime.now().isoformat(),
                        'bags_detected': len(bags),
                        'confidence_scores': confidence_scores,
                        'detection_boxes': detection_boxes,
                        'inference_time': inference_time
                    }
                    
                    # Update line counter if there's a counting line defined
                    if self.line_counter.get_line(camera_name):
                        # Use a copy of the frame for the line counter update
                        frame_for_counter = frame.copy() if frame is not None else None
                        counter_results = self.line_counter.update(camera_name, detection_boxes, frame_for_counter)
                        
                        # Add counter results to detection data
                        self.camera_feeds[camera_name]['detection_data']['counter_results'] = counter_results
                
                # Sleep to reduce CPU usage - shorter sleep when just capturing frames
                if self.camera_feeds[camera_name]['detection_enabled']:
                    time.sleep(0.1)  # Longer sleep when doing detection
                else:
                    time.sleep(0.03)  # Shorter sleep when just capturing frames
                
        except Exception as e:
            print(f"Error in detection thread for camera {camera_name}: {str(e)}")
        finally:
            cap.release()
            print(f"Detection stopped for camera {camera_name}")
            self.camera_feeds[camera_name]['detection_enabled'] = False
    
    def get_latest_annotated_frame(self, camera_name, format='jpeg', quality='low'):
        """Get the latest annotated frame with bag detections as a byte array"""
        if camera_name not in self.camera_feeds:
            return None
            
        last_frame = self.camera_feeds[camera_name]['last_frame']
        if last_frame is None:
            return None
        
        # Always create a copy of the frame to avoid modifying the original
        annotated_frame = last_frame.copy()
        
        # Choose resolution based on quality setting
        if quality == 'thumbnail':
            # Extremely small thumbnails for grid view
            target_width = 240  # Tiny thumbnail for grid view
            h, w = annotated_frame.shape[:2]
            aspect_ratio = w/h
            target_height = int(target_width / aspect_ratio)
            
            # Resize the frame to thumbnail resolution
            annotated_frame = cv2.resize(annotated_frame, (target_width, target_height))
            
            # For thumbnails, skip most annotations to simplify image
            timestamp_str = datetime.now().strftime("%H:%M:%S")
            cv2.putText(annotated_frame, timestamp_str, (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                       
            # Return early for thumbnails - skipping all other annotations
            # Encode the frame with very high compression
            ret, buffer = cv2.imencode(f'.{format}', 
                                      annotated_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 15])
            if not ret:
                return None
                
            return buffer.tobytes()
            
        elif quality == 'low':
            # Reduced resolution for regular view
            target_width = 480
            h, w = annotated_frame.shape[:2]
            aspect_ratio = w/h
            target_height = int(target_width / aspect_ratio)
            
            # Resize the frame to lower resolution
            annotated_frame = cv2.resize(annotated_frame, (target_width, target_height))
        
        # Add timestamp to the frame
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp_str, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Check if there's a counting line defined for this camera
        line = self.line_counter.get_line(camera_name)
        if line:
            start_point = tuple(line["start"])
            end_point = tuple(line["end"])
            # Draw the counting line
            cv2.line(annotated_frame, start_point, end_point, (0, 255, 255), 2)
            
            # Draw direction indicator
            mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
            if line["direction"] == "horizontal":
                if line["in_direction"] == "up":
                    cv2.arrowedLine(annotated_frame, (mid_point[0], mid_point[1]+20), 
                                    (mid_point[0], mid_point[1]-20), (0, 255, 0), 2)
                    direction_text = "IN ↑ / OUT ↓"
                else:
                    cv2.arrowedLine(annotated_frame, (mid_point[0], mid_point[1]-20), 
                                    (mid_point[0], mid_point[1]+20), (0, 255, 0), 2)
                    direction_text = "IN ↓ / OUT ↑"
            else:
                if line["in_direction"] == "right":
                    cv2.arrowedLine(annotated_frame, (mid_point[0]-20, mid_point[1]), 
                                    (mid_point[0]+20, mid_point[1]), (0, 255, 0), 2)
                    direction_text = "IN → / OUT ←"
                else:
                    cv2.arrowedLine(annotated_frame, (mid_point[0]+20, mid_point[1]), 
                                    (mid_point[0]-20, mid_point[1]), (0, 255, 0), 2)
                    direction_text = "IN ← / OUT →"
            
            # Add direction text
            cv2.putText(annotated_frame, direction_text, 
                       (mid_point[0] - 50, mid_point[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Get counts
            counts = self.line_counter.get_counts(camera_name)
            cv2.putText(annotated_frame, f"IN: {counts['in']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"OUT: {counts['out']}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # If detection is enabled, draw detection boxes
        if self.camera_feeds[camera_name]['detection_enabled']:
            detection_boxes = self.camera_feeds[camera_name]['detection_data']['detection_boxes']
            bags_detected = len(detection_boxes)
            
            # Draw detection boxes on the frame
            for x1, y1, x2, y2 in detection_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add detection count
            cv2.putText(annotated_frame, f"Detected: {bags_detected}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode the frame as JPEG with compression
        if format == 'jpeg':
            # Use appropriate compression level based on quality setting
            if quality == 'low':
                compress_quality = 30  # Higher compression for regular low quality
            elif quality == 'high':
                compress_quality = 80  # Lower compression for high quality
            else:
                # Default quality
                compress_quality = 50
                
            ret, buffer = cv2.imencode(f'.{format}', 
                                      annotated_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, compress_quality])
        else:
            ret, buffer = cv2.imencode(f'.{format}', annotated_frame)
            
        if not ret:
            return None
            
        return buffer.tobytes()
    
    def cleanup(self):
        """Clean up resources"""
        for camera_name in list(self.camera_feeds.keys()):
            self.stop_detection(camera_name)
        
        # Wait for all threads to exit
        for thread in self.processing_threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)

# Global bag detector instance
bag_detector = None

def initialize_detector(confidence_threshold=0.5):
    """Initialize the global bag detector instance"""
    global bag_detector
    if bag_detector is None:
        bag_detector = BagDetector(confidence_threshold=confidence_threshold)
    return bag_detector

def get_detector():
    """Get the global bag detector instance"""
    global bag_detector
    if bag_detector is None:
        bag_detector = initialize_detector()
    return bag_detector

import cv2
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import math

db = SQLAlchemy()

class CountingLine(db.Model):
    """
    Model to store counting lines for cameras to track bag movement
    """
    id = db.Column(db.Integer, primary_key=True)
    camera_name = db.Column(db.String(80), unique=True, nullable=False)
    # Store line as JSON: {"start": [x1,y1], "end": [x2,y2], "direction": "horizontal", "in_direction": "right"}
    line_data = db.Column(db.Text, nullable=False, default='{}')
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    
    def __repr__(self):
        return f'<CountingLine {self.camera_name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'camera_name': self.camera_name,
            'line_data': json.loads(self.line_data),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class BagCounter:
    """Class to track and count bags as they cross a defined line"""
    def __init__(self):
        self.tracked_objects = {}  # {object_id: {center_points: [..previous centers..], crossed: bool}}
        self.object_id_count = 0
        self.in_count = 0
        self.out_count = 0
        self.counting_lines = {}  # {camera_name: {"start": [x1,y1], "end": [x2,y2], "direction": "horizontal", "in_direction": "right"}}
        self.history = {}  # {camera_name: {"in": [timestamps], "out": [timestamps]}}
        self.crossing_threshold = 5  # Minimum pixels between centers to consider movement
    
    def set_counting_line(self, camera_name, start_point, end_point, in_direction="right"):
        """
        Set a counting line for a camera
        
        Args:
            camera_name: Name of the camera
            start_point: [x1, y1] coordinates of line start
            end_point: [x2, y2] coordinates of line end
            in_direction: "right" or "left" - direction considered as "in"
        """
        # Determine if line is more horizontal or vertical
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        
        direction = "horizontal" if abs(dx) > abs(dy) else "vertical"
        
        self.counting_lines[camera_name] = {
            "start": start_point, 
            "end": end_point, 
            "direction": direction, 
            "in_direction": in_direction
        }
        
        # Initialize history for this camera if it doesn't exist
        if camera_name not in self.history:
            self.history[camera_name] = {"in": [], "out": []}
            
        # Save to database
        self.save_line_to_db(camera_name)
        
        return self.counting_lines[camera_name]
    
    def save_line_to_db(self, camera_name):
        """Save the counting line to database"""
        if camera_name not in self.counting_lines:
            return False
        
        line = CountingLine.query.filter_by(camera_name=camera_name).first()
        if not line:
            line = CountingLine(camera_name=camera_name)
        
        line.line_data = json.dumps(self.counting_lines[camera_name])
        db.session.add(line)
        db.session.commit()
        return True
    
    def load_lines_from_db(self):
        """Load all counting lines from database"""
        lines = CountingLine.query.all()
        for line in lines:
            try:
                self.counting_lines[line.camera_name] = json.loads(line.line_data)
                if line.camera_name not in self.history:
                    self.history[line.camera_name] = {"in": [], "out": []}
            except:
                continue
        return self.counting_lines
    
    def get_line(self, camera_name):
        """Get the counting line for a camera"""
        return self.counting_lines.get(camera_name, None)
    
    def calculate_centroid(self, bbox):
        """Calculate the center point of a bounding box [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def point_crossed_line(self, point, line, previous_point=None):
        """
        Check if a point has crossed the defined line
        
        Args:
            point: Current point [x, y]
            line: Line dictionary {"start": [x1,y1], "end": [x2,y2], "direction": "horizontal", "in_direction": "right"}
            previous_point: Previous point [x, y] to determine direction
        
        Returns:
            None if not crossed, "in" or "out" if crossed
        """
        if previous_point is None:
            return None
            
        x, y = point
        prev_x, prev_y = previous_point
        start_x, start_y = line["start"]
        end_x, end_y = line["end"]
        
        if line["direction"] == "horizontal":
            # For horizontal line, check if point crosses the y-value of the line
            # and also falls within the x-range of the line
            if ((prev_y < start_y and y >= start_y) or (prev_y > start_y and y <= start_y)):
                # Calculate x-intersection at crossing point
                if end_x == start_x:  # Prevent division by zero
                    line_x_at_crossing = start_x
                else:
                    slope = (end_y - start_y) / (end_x - start_x)
                    if slope == 0:  # Perfectly horizontal line
                        line_x_at_crossing = None  # No specific x-intersection
                    else:
                        # y = m(x - x1) + y1 => x = (y - y1) / m + x1
                        line_x_at_crossing = (y - start_y) / slope + start_x
                
                # Check if crossing point is within line segment x-range
                if line_x_at_crossing is None or (min(start_x, end_x) <= x <= max(start_x, end_x)):
                    # Determine direction
                    if (y > prev_y and line["in_direction"] == "down") or (y < prev_y and line["in_direction"] == "up"):
                        return "in"
                    else:
                        return "out"
        else:
            # For vertical line, check if point crosses the x-value of the line
            # and also falls within the y-range of the line
            if ((prev_x < start_x and x >= start_x) or (prev_x > start_x and x <= start_x)):
                # Calculate y-intersection at crossing point
                if end_y == start_y:  # Prevent division by zero
                    line_y_at_crossing = start_y
                else:
                    inv_slope = (end_x - start_x) / (end_y - start_y)
                    if inv_slope == 0:  # Perfectly vertical line
                        line_y_at_crossing = None  # No specific y-intersection
                    else:
                        # x = m(y - y1) + x1 => y = (x - x1) / m + y1
                        line_y_at_crossing = (x - start_x) / inv_slope + start_y
                
                # Check if crossing point is within line segment y-range
                if line_y_at_crossing is None or (min(start_y, end_y) <= y <= max(start_y, end_y)):
                    # Determine direction
                    if (x > prev_x and line["in_direction"] == "right") or (x < prev_x and line["in_direction"] == "left"):
                        return "in"
                    else:
                        return "out"
                        
        return None
    
    def update(self, camera_name, detections, frame=None):
        """
        Process new detections and update counts
        
        Args:
            camera_name: Name of the camera
            detections: List of bounding boxes [[x1, y1, x2, y2], ...]
            frame: Optional frame for visualization
        """
        if camera_name not in self.counting_lines:
            return {
                "in_count": 0,
                "out_count": 0,
                "status": "No counting line defined"
            }
        
        if camera_name not in self.history:
            self.history[camera_name] = {"in": [], "out": []}
        
        line = self.counting_lines[camera_name]
        
        # Track objects and update counts
        new_objects = {}
        
        for bbox in detections:
            center = self.calculate_centroid(bbox)
            object_id = None
            
            # Find the closest tracked object
            min_dist = float('inf')
            for oid, obj_data in self.tracked_objects.items():
                if len(obj_data["center_points"]) > 0:
                    prev_center = obj_data["center_points"][-1]
                    dist = math.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                    if dist < min_dist and dist < 50:  # Maximum distance threshold
                        min_dist = dist
                        object_id = oid
            
            # If no nearby object, create new one
            if object_id is None:
                object_id = f"{camera_name}_{self.object_id_count}"
                self.object_id_count += 1
                self.tracked_objects[object_id] = {
                    "center_points": [center],
                    "crossed": False
                }
            else:
                # Add center to existing object
                prev_centers = self.tracked_objects[object_id]["center_points"]
                prev_center = prev_centers[-1]
                
                # Check if object crossed the line
                if not self.tracked_objects[object_id]["crossed"]:
                    crossed = self.point_crossed_line(center, line, prev_center)
                    if crossed:
                        self.tracked_objects[object_id]["crossed"] = True
                        
                        if crossed == "in":
                            self.in_count += 1
                            self.history[camera_name]["in"].append(datetime.now().isoformat())
                        else:
                            self.out_count += 1
                            self.history[camera_name]["out"].append(datetime.now().isoformat())
                
                # Add new center
                self.tracked_objects[object_id]["center_points"].append(center)
                if len(self.tracked_objects[object_id]["center_points"]) > 10:
                    self.tracked_objects[object_id]["center_points"].pop(0)
            
            new_objects[object_id] = self.tracked_objects[object_id]
        
        # Keep only objects seen in current frame
        self.tracked_objects = new_objects
        
        # Draw line on frame if provided
        if frame is not None:
            start_point = tuple(line["start"])
            end_point = tuple(line["end"])
            cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
            
            # Draw direction indicator
            mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
            if line["direction"] == "horizontal":
                if line["in_direction"] == "up":
                    cv2.arrowedLine(frame, (mid_point[0], mid_point[1]+20), 
                                    (mid_point[0], mid_point[1]-20), (0, 255, 0), 2)
                else:
                    cv2.arrowedLine(frame, (mid_point[0], mid_point[1]-20), 
                                    (mid_point[0], mid_point[1]+20), (0, 255, 0), 2)
            else:
                if line["in_direction"] == "right":
                    cv2.arrowedLine(frame, (mid_point[0]-20, mid_point[1]), 
                                    (mid_point[0]+20, mid_point[1]), (0, 255, 0), 2)
                else:
                    cv2.arrowedLine(frame, (mid_point[0]+20, mid_point[1]), 
                                    (mid_point[0]-20, mid_point[1]), (0, 255, 0), 2)
            
            # Draw counts
            cv2.putText(frame, f"IN: {self.in_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 255), 2)
        
        return {
            "in_count": self.in_count,
            "out_count": self.out_count,
            "in_today": len(self.history[camera_name]["in"]),
            "out_today": len(self.history[camera_name]["out"]),
            "status": "Active"
        }
        
    def get_counts(self, camera_name=None):
        """Get current counts for a specific camera or all cameras"""
        if camera_name:
            if camera_name not in self.history:
                return {"in": 0, "out": 0}
            return {
                "in": len(self.history[camera_name]["in"]),
                "out": len(self.history[camera_name]["out"])
            }
        else:
            # Return counts for all cameras
            all_counts = {}
            for cam_name in self.history:
                all_counts[cam_name] = {
                    "in": len(self.history[cam_name]["in"]),
                    "out": len(self.history[cam_name]["out"])
                }
            return all_counts
    
    def reset_counts(self, camera_name=None):
        """Reset counts for a specific camera or all cameras"""
        if camera_name:
            if camera_name in self.history:
                self.history[camera_name] = {"in": [], "out": []}
        else:
            # Reset counts for all cameras
            for cam_name in self.history:
                self.history[cam_name] = {"in": [], "out": []}
        
        # Also reset overall counts
        self.in_count = 0
        self.out_count = 0

# Global counter instance
bag_counter = None

def initialize_counter():
    """Initialize the global bag counter instance"""
    global bag_counter
    if bag_counter is None:
        bag_counter = BagCounter()
        bag_counter.load_lines_from_db()
    return bag_counter

def get_counter():
    """Get the global bag counter instance"""
    global bag_counter
    if bag_counter is None:
        bag_counter = initialize_counter()
    return bag_counter

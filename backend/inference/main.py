import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, LineString
import json
import subprocess
import time

# --- CONFIGURATION ---
import os
RTSP_INPUT = os.environ.get("RTSP_INPUT", "rtsp://YOUR_CAMERA_URL")  # Can be set from env
RTSP_OUTPUT = "rtsp://localhost:8554/stream"  # FFmpeg will publish to this endpoint
MODEL_PATH = "../../best_cement_bags.pt"  # Path to your YOLOv8 model
BOUNDARY_PATH = "boundaries.json"  # Path to boundary definition
FRAME_WIDTH = 1280  # Set according to your camera
FRAME_HEIGHT = 720
FPS = 25

# --- LOAD BOUNDARY ---
def load_boundary(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['type'], data['points']

# --- CHECK VIOLATION ---
def check_violation(center, boundary_type, boundary_points):
    if boundary_type == 'line':
        line = LineString(boundary_points)
        point = Point(center)
        # Example: flag as violation if point is above the line (y < line)
        return point.y < min([p[1] for p in boundary_points])
    elif boundary_type == 'polygon':
        from shapely.geometry import Polygon
        poly = Polygon(boundary_points)
        return not poly.contains(Point(center))
    return False

# --- DRAW OVERLAYS ---
def draw_overlays(frame, detections, boundary_type, boundary_points, violations, violation_count):
    # Draw boundary
    if boundary_type == 'line':
        cv2.line(frame, tuple(boundary_points[0]), tuple(boundary_points[1]), (0,255,255), 2)
    elif boundary_type == 'polygon':
        cv2.polylines(frame, [np.array(boundary_points, np.int32)], True, (0,255,255), 2)
    # Draw detections
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cx, cy = int((x1+x2)//2), int((y1+y2)//2)
        cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)
        cv2.putText(frame, f"{int(cls)} {(conf*100):.1f}%", (int(x1),int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # Draw violations
    for v in violations:
        cx, cy = v
        cv2.putText(frame, "VIOLATION", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    # Draw counter
    cv2.putText(frame, f"Violations: {violation_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame

# --- MAIN LOOP ---
def main():
    boundary_type, boundary_points = load_boundary(BOUNDARY_PATH)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(RTSP_INPUT)
    
    # FFmpeg command to publish to RTSP server
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec','rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}', '-r', str(FPS), '-i', '-',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
        '-f', 'rtsp', RTSP_OUTPUT
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    violation_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Retrying...")
            time.sleep(0.1)
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        results = model(frame)
        detections = []
        violations = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            cx, cy = int((x1+x2)//2), int((y1+y2)//2)
            if check_violation((cx, cy), boundary_type, boundary_points):
                violations.append((cx, cy))
                violation_count += 1
            detections.append([x1, y1, x2, y2, conf, cls])
        frame = draw_overlays(frame, detections, boundary_type, boundary_points, violations, violation_count)
        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("[ERROR] FFmpeg pipe broken. Exiting.")
            break
    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

if __name__ == "__main__":
    main()

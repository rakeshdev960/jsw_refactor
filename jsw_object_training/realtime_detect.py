from ultralytics import YOLO
import cv2
import time

def run_real_time_tracking():
    # Load the trained model
    model = YOLO('best.pt')
    
    # Open webcam
    print("Starting webcam for tracking...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam started successfully! Press 'q' to quit.")
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Tracker state
    tracker = 'bytetrack'  # You can use 'bytetrack' or 'botsort' if available
    
    # Counting line (horizontal, center)
    line_color = (255, 0, 0)
    line_thickness = 2
    counted_ids = set()
    total_count = 0
    last_positions = {}  # track previous centroid positions for each ID
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        # Get frame dimensions and line position
        h, w = frame.shape[:2]
        line_y = h // 2
        # Run YOLOv8 with tracking
        results = model.track(frame, persist=True, tracker=tracker, conf=0.25)
        annotated_frame = results[0].plot()
        # Draw the counting line
        cv2.line(annotated_frame, (0, line_y), (w, line_y), line_color, line_thickness)
        # Display FPS and detection count
        detection_count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Cement Bags: {detection_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Counted: {total_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Draw tracking IDs and count crossings
        if hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box.tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.putText(annotated_frame, f'ID: {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # Count if the object crosses the line from top to bottom
                if int(track_id) not in counted_ids:
                    prev_cy = last_positions.get(int(track_id), None)
                    if prev_cy is not None and prev_cy < line_y and cy >= line_y:
                        total_count += 1
                        counted_ids.add(int(track_id))
                last_positions[int(track_id)] = cy
        cv2.imshow("YOLOv8 Real-time Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking stopped.")

if __name__ == "__main__":
    run_real_time_tracking()

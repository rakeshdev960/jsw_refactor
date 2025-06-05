"""
This script updates the combined_detection.py file to display class name and confidence
along with object ID in the detection overlay.
"""

import re
import os

def update_combined_detection_file():
    file_path = 'combined_detection.py'
    backup_path = 'combined_detection_backup.py'
    
    # Create a backup if it doesn't exist
    if not os.path.exists(backup_path):
        with open(file_path, 'r') as f:
            content = f.read()
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Created backup at {backup_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update Upload Video section - display code
    upload_video_pattern = r'# Draw bounding box and ID\s+cv2\.rectangle\(frame, \(x, y\), \(x\+w, y\+h\), \(0, 255, 0\), 2\)\s+cv2\.putText\(frame, f"ID:{id}", \(x, y - 10\), cv2\.FONT_HERSHEY_SIMPLEX, 0\.5, \(0, 255, 0\), 2\)'
    upload_video_replacement = '''# Draw bounding box and label with ID, class name, and confidence
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Get class name and confidence if available
                    box_key = f"{x}_{y}_{w}_{h}"
                    if box_key in detection_info:
                        class_id, conf = detection_info[box_key]
                        class_name = cement_model.names[class_id]
                        label = f"ID:{id} {class_name} {conf:.2f}"
                    else:
                        label = f"ID:{id}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)'''
    
    content = re.sub(upload_video_pattern, upload_video_replacement, content)
    
    # Update RTSP Camera section - detection info
    rtsp_detection_pattern = r'results = cement_model\(frame, conf=confidence_threshold\)\[0\]\s+boxes = \[\]\s+for r in results\.boxes\.data\.tolist\(\):\s+x1, y1, x2, y2, score, class_id = r\s+if score > confidence_threshold:\s+boxes\.append\(\[int\(x1\), int\(y1\), int\(x2\)-int\(x1\), int\(y2\)-int\(y1\)\]\)'
    rtsp_detection_replacement = '''results = cement_model(frame, conf=confidence_threshold)[0]
                    boxes = []
                    detection_info = {}  # Store detection info (class_id, confidence) for each box
                    for r in results.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = r
                        if score > confidence_threshold:
                            box_key = f"{int(x1)}_{int(y1)}_{int(x2)-int(x1)}_{int(y2)-int(y1)}"
                            detection_info[box_key] = (int(class_id), float(score))
                            boxes.append([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)])'''
    
    content = re.sub(rtsp_detection_pattern, rtsp_detection_replacement, content)
    
    # Update RTSP Camera section - display code
    rtsp_display_pattern = r'# Draw bounding box and ID\s+cv2\.rectangle\(frame, \(x, y\), \(x\+w, y\+h\), \(0, 255, 0\), 2\)\s+cv2\.putText\(frame, f"ID:{id}", \(x, y - 10\), cv2\.FONT_HERSHEY_SIMPLEX, 0\.5, \(0, 255, 0\), 2\)'
    rtsp_display_replacement = '''# Draw bounding box and label with ID, class name, and confidence
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Get class name and confidence if available
                    box_key = f"{x}_{y}_{w}_{h}"
                    if box_key in detection_info:
                        class_id, conf = detection_info[box_key]
                        class_name = cement_model.names[class_id]
                        label = f"ID:{id} {class_name} {conf:.2f}"
                    else:
                        label = f"ID:{id}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)'''
    
    content = re.sub(rtsp_display_pattern, rtsp_display_replacement, content)
    
    # Write the updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {file_path} to display class name and confidence along with object ID")

if __name__ == "__main__":
    update_combined_detection_file()

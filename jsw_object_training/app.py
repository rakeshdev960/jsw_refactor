import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Cement Bag Detector",
    page_icon="🏗️",
    layout="wide"
)

# Title and description
st.title("🏗️ Cement Bag Detection")
st.markdown("""
This application uses YOLOv8 to detect cement bags in images. 
Upload an image and see the detection results in real-time!
""")

# Sidebar
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Main content
uploaded_file = st.file_uploader(
    "Choose an image or video...",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    file_suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    col1, col2 = st.columns(2)
    if file_suffix in [".jpg", ".jpeg", ".png"]:
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, use_column_width=True)
        try:
            model = YOLO('best.pt')
            results = model(tmp_path, conf=confidence_threshold)
            result = results[0]
            annotated_img = result.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            with col2:
                st.subheader("Detection Results")
                st.image(annotated_img_rgb, use_column_width=True)
                st.markdown("### Detection Information")
                st.write(f"Number of cement bags detected: **{len(result.boxes)}**")
                detection_data = []
                for i, box in enumerate(result.boxes):
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detection_data.append({
                        "Bag": i+1,
                        "Confidence": f"{conf:.2f}",
                        "Location": f"({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})"
                    })
                st.table(detection_data)
                _, buffer = cv2.imencode('.jpg', annotated_img)
                st.download_button(
                    label="Download Annotated Image",
                    data=buffer.tobytes(),
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(f"An error occurred during detection: {str(e)}")
        finally:
            os.unlink(tmp_path)
    elif file_suffix in [".mp4", ".avi", ".mov"]:
        with col1:
            st.subheader("Original Video")
            st.video(tmp_path)
        try:
            model = YOLO('best.pt')
            cap = cv2.VideoCapture(tmp_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = tmp_path.replace(file_suffix, f"_detected{file_suffix}")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            frame_count = 0
            detection_counts = []
            # For drawing a counting line (horizontal, center)
            line_color = (255, 0, 0)
            line_thickness = 2
            line_y = height // 2
            counted_ids = set()
            total_count = 0
            last_positions = {}
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Use YOLOv8 tracking with botsort tracker
                results = model.track(frame, tracker='botsort', persist=True, conf=confidence_threshold)
                result = results[0]
                annotated_frame = result.plot()
                # Draw the counting line
                cv2.line(annotated_frame, (0, line_y), (width, line_y), line_color, line_thickness)
                # Draw tracking IDs and count crossings
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    for box, track_id in zip(result.boxes.xyxy, result.boxes.id):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        # Draw centroid (bright yellow)
                        cv2.circle(annotated_frame, (cx, cy), 8, (0, 255, 255), -1)
                        cv2.putText(annotated_frame, f'ID: {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        # Count if the object crosses the line (both directions for easier testing)
                        if int(track_id) not in counted_ids:
                            prev_cy = last_positions.get(int(track_id), None)
                            if prev_cy is not None:
                                if (prev_cy < line_y and cy >= line_y) or (prev_cy > line_y and cy <= line_y):
                                    total_count += 1
                                    counted_ids.add(int(track_id))
                                    print(f"Counted ID {int(track_id)} crossing line: prev_cy={prev_cy}, cy={cy}")
                        last_positions[int(track_id)] = cy
                # Show count on video
                cv2.putText(annotated_frame, f"Counted: {total_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                out.write(annotated_frame)
                detection_counts.append(len(result.boxes))
                frame_count += 1
                if frame_count > 200:  # Limit processing for demo purposes
                    break
            cap.release()
            out.release()
            # Ensure file is flushed/closed before reading
            import time as _t; _t.sleep(0.5)
            with col2:
                st.subheader("Detection Results (First 200 frames)")
                # Display the processed video directly in the browser
                try:
                    with open(out_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                except Exception as e:
                    st.warning(f"Could not display video in browser. Please use the download button. Error: {e}")
                st.markdown(f"### Detection Information (first 200 frames)")
                st.write(f"Average cement bags detected per frame: **{sum(detection_counts)/len(detection_counts):.2f}**")
                st.write(f"Total cement bags counted crossing the line: **{total_count}**")
                st.download_button(
                    label="Download Annotated Video",
                    data=video_bytes,
                    file_name=f"detected_{uploaded_file.name}",
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"An error occurred during video detection: {str(e)}")
        finally:
            # Only delete the uploaded temp file, not the output video (which may still be in use)
            os.unlink(tmp_path)
            # Do NOT delete out_path here to avoid PermissionError
    else:
        st.error("Unsupported file type. Please upload an image or video.")

# Footer
st.markdown("---")
st.markdown("""
### How to use:
1. Upload an image or video containing cement bags
2. Adjust the confidence threshold if needed
3. View the detection results
4. Download the annotated image if desired
""") 
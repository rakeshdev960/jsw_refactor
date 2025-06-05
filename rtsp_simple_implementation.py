import streamlit as st
import cv2
import time
import os
import numpy as np
import torch
from ultralytics import YOLO

# Set page configuration
st.set_page_config(page_title="JSW Cement Bag Detection", layout="wide")

# Title
st.title("Simple JSW RTSP Implementation")
st.write("This is a simplified version of the cement bag detection that focuses on stable RTSP connection")

# Load model silently
@st.cache_resource
def load_model():
    try:
        # Load model without showing GPU messages
        model = YOLO('F:/jsw20042025/best_cement_bags.pt')
        
        # Try GPU silently without any UI messages
        if torch.cuda.is_available():
            try:
                model.to('cuda')
            except Exception as e:
                print(f"GPU error (not shown to user): {e}")
                model.to('cpu')
        else:
            model.to('cpu')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Input for RTSP URL
rtsp_url = st.text_input(
    "RTSP URL", 
    value="rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels/101"
)

# Connection control
if "connected" not in st.session_state:
    st.session_state.connected = False

# Start/Stop Button
if not st.session_state.connected:
    if st.button("Start RTSP Stream"):
        st.session_state.connected = True
        st.rerun()
else:
    if st.button("Stop RTSP Stream"):
        st.session_state.connected = False
        st.rerun()

# Create placeholders
video_placeholder = st.empty()
status_placeholder = st.empty()

# Start detection if connected
if st.session_state.connected:
    try:
        # Load YOLO model
        model = load_model()
        if model is None:
            st.error("Failed to load detection model")
            st.session_state.connected = False
            st.rerun()
            
        # Show simple connecting message
        status_placeholder.info("Connecting to camera...")
        
        # Configure RTSP connection - exact settings from tester
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;10000000|buffer_size;10485760"
        
        # Add small delay before opening capture
        time.sleep(0.5)
        
        # Open RTSP stream directly with TCP
        tcp_url = rtsp_url
        if "?" not in rtsp_url:
            tcp_url = rtsp_url + "?tcp"
            
        cap = cv2.VideoCapture(tcp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer as in tester
        
        # Check if stream opened
        if not cap.isOpened():
            status_placeholder.error("Failed to open RTSP stream. Please check URL and try again.")
            st.session_state.connected = False
            st.rerun()
            
        # Show connected message
        status_placeholder.success("Connected to camera!")
        
        # Display frames in a loop
        while st.session_state.connected:
            # Try to read a frame
            ret, frame = cap.read()
            
            if not ret:
                status_placeholder.warning("Lost connection. Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(tcp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                time.sleep(1)
                continue
                
            # Do detection
            if model is not None:
                # Process frame with YOLO
                try:
                    results = model(frame)
                    # Draw results on frame
                    for r in results:
                        annotated_frame = r.plot()
                        # Convert to RGB for display
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        # Display the frame
                        video_placeholder.image(rgb_frame, caption="RTSP Feed with Detection", use_container_width=True)
                except Exception as e:
                    # Continue if detection fails for any reason
                    print(f"Detection error: {e}")
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, caption="RTSP Feed (Detection Error)", use_container_width=True)
            else:
                # Just show the frame if no model
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, caption="RTSP Feed (No Detection)", use_container_width=True)
                
            # Short delay to prevent UI freeze
            time.sleep(0.03)
            
        # Clean up on disconnect
        cap.release()
        status_placeholder.info("Disconnected from camera")
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.connected = False
else:
    video_placeholder.info("Click 'Start RTSP Stream' to begin")

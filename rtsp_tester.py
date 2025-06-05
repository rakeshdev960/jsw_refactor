import streamlit as st
import cv2
import time
import numpy as np
import os

st.set_page_config(page_title="RTSP Tester", layout="wide")

st.title("Simple RTSP Connection Tester")

# RTSP URL Input
rtsp_input_col1, rtsp_input_col2 = st.columns([3, 1])

with rtsp_input_col1:
    rtsp_url = st.text_input(
        "RTSP URL", 
        value="rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels/101"
    )

# Create session state for connection status
if "connected" not in st.session_state:
    st.session_state.connected = False

# Button to connect/disconnect
with rtsp_input_col2:
    if not st.session_state.connected:
        if st.button("Connect"):
            st.session_state.connected = True
            st.rerun()
    else:
        if st.button("Disconnect"):
            st.session_state.connected = False
            st.rerun()

# Display area
video_placeholder = st.empty()

# Status message area
status = st.empty()

if st.session_state.connected:
    try:
        status.info("Connecting to RTSP stream...")
        
        # Set OpenCV options for reliable RTSP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;10000000|buffer_size;10485760"
        
        # Open RTSP stream with buffer size setting
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer to reduce latency
        
        if not cap.isOpened():
            status.error("Failed to open RTSP stream. Please check the URL and try again.")
            st.session_state.connected = False
        else:
            status.success("Connected to stream. Displaying video...")
            
            # Display feed
            while st.session_state.connected:
                ret, frame = cap.read()
                
                if not ret:
                    status.warning("Lost connection. Reconnecting...")
                    # Try to reconnect
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    time.sleep(1)
                    continue
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                video_placeholder.image(frame_rgb, caption="RTSP Stream", use_container_width=True)
                
                # Add small delay
                time.sleep(0.03)  # ~30 FPS
            
            # Release resources on disconnect
            cap.release()
            status.info("Disconnected from stream.")
    
    except Exception as e:
        status.error(f"Error: {str(e)}")
        st.session_state.connected = False

else:
    video_placeholder.info("Click 'Connect' to start viewing the RTSP stream")

# Footer with instructions
st.markdown("---")
st.markdown("""
### Troubleshooting RTSP Connections
1. Verify the RTSP URL format is correct (rtsp://username:password@ip:port/path)
2. Make sure the camera is online and accessible from this computer
3. Check if any firewalls might be blocking the connection
4. Try different cameras or streams to isolate the issue
""")

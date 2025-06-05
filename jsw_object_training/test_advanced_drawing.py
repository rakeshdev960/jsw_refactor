import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import advanced_drawing_component as adc

st.title("Advanced Drawing Demo - Cement Bag Detection")

st.write("Upload an image or video frame, then draw lines/zones interactively.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save to temp file
    import tempfile
    import os
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp.write(uploaded_file.read())
    temp.close()
    image_url = temp.name
    st.image(image_url, caption="Uploaded Image", use_column_width=True)

    shapes = adc.st_advanced_drawing_canvas(image_url=image_url, width=640, height=480)
    st.write("Drawn shapes (coordinates):", shapes)
else:
    st.info("Please upload an image to start drawing.")

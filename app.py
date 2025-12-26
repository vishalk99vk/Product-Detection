import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

st.set_page_config(layout="wide")
st.title("🏪 Retail Product Detector")

@st.cache_resource
def load_model():
    # Use yolov8m or yolov8l for better accuracy than the 'n' (nano) version
    return YOLO("yolov8m.pt") 

model = load_model()

uploaded_file = st.file_uploader("Upload store shelf image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Run YOLO detection
    # Increased 'imgsz' to help detect small objects on high-res shelves
    results = model.predict(source=img_array, conf=0.10, iou=0.3, imgsz=640)
    result = results[0]

    if len(result.boxes) == 0:
        st.warning("No objects detected. The model might not recognize these items.")
    else:
        # Convert RGB to BGR for OpenCV processing
        annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Draw boxes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw a green box (0, 255, 0) for better visibility against red labels
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert back to RGB for Streamlit display
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader(f"Detected {len(result.boxes)} Objects")
            st.image(annotated_img_rgb, use_container_width=True)

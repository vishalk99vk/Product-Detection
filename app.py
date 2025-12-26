import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

st.set_page_config(layout="wide")
st.title("🎯 High-Accuracy SKU Detector")

@st.cache_resource
def load_retail_model():
    # This model is pre-trained specifically for dense retail shelves
    # It focuses on finding "product boxes" rather than general objects
    return YOLO('foduucom/shelf-object-detection-yolov8')

model = load_retail_model()
uploaded_file = st.file_uploader("Upload shelf image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # CRITICAL SETTINGS FOR ACCURACY:
    # imgsz=1280: Zooms in on small products (crucial for dense shelves)
    # conf=0.15: Catches items the model is less certain about
    # iou=0.3: Allows boxes to be very close together
    results = model.predict(source=img_array, conf=0.15, iou=0.3, imgsz=1280)
    
    # Process image for display
    annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Draw the Red Box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.success(f"Detected {len(results[0].boxes)} products with high-precision shelf weights.")

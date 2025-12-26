import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

st.set_page_config(layout="wide")
st.title("🏪 Red Box SKU Finder")

@st.cache_resource
def load_model():
    # Use 'm' (medium) instead of 'n' (nano). It is much better at 
    # seeing the boundaries of boxes on a shelf.
    return YOLO("yolov8m.pt") 

model = load_model()

uploaded_file = st.file_uploader("Upload Store Image", type=["jpg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # SETTINGS FOR FULL SKU DETECTION:
    # 1. imgsz=1280: This is the most important. It zooms in so small packs are clear.
    # 2. conf=0.05: We set this very low to catch EVERY possible box.
    # 3. iou=0.2: This helps prevent boxes from overlapping too much.
    results = model.predict(source=img_array, conf=0.05, iou=0.2, imgsz=1280)
    
    annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    count = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Draw Red Box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        count += 1

    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    st.write(f"Detected {count} potential SKU areas.")

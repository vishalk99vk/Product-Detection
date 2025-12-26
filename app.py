import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

st.set_page_config(layout="wide")
st.title("🔍 Retail Detection Debugger")

@st.cache_resource
def load_model():
    # Switching to Medium model for better context than Nano
    return YOLO("yolov8m.pt") 

model = load_model()
uploaded_file = st.file_uploader("Upload store shelf image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # Increase imgsz to 1024 to help the model see the full pack boundaries
    results = model.predict(source=img_array, conf=0.10, iou=0.25, imgsz=1024)
    result = results[0]

    annotated_img = img_array.copy()
    crops = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        
        # Draw the box and the label it guessed (e.g., 'cell phone')
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        # Extract the crop for inspection
        crop = img_array[y1:y2, x1:x2]
        crops.append((label, crop))

    st.subheader("Detection Results")
    st.image(annotated_img, use_container_width=True)

    if crops:
        st.subheader("What the model 'sees' inside the boxes:")
        cols = st.columns(5)
        for i, (label, crop_img) in enumerate(crops[:15]): # Show first 15 crops
            with cols[i % 5]:
                st.image(crop_img, caption=f"Guessed: {label}")

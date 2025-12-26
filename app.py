import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide")
st.title("🏪 High-Accuracy SKU Detector")

@st.cache_resource
def load_retail_model():
    # Step 1: Download the actual weight file from Hugging Face
    model_path = hf_hub_download(repo_id="foduucom/shelf-object-detection-yolov8", 
                                 filename="best.pt")
    # Step 2: Load the downloaded local file
    return YOLO(model_path)

model = load_retail_model()

uploaded_file = st.file_uploader("Upload shelf image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Note: Using high imgsz (1280) for dense shelf accuracy
    results = model.predict(source=img_array, conf=0.15, iou=0.3, imgsz=1280)
    
    annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red Box

    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.success(f"Detected {len(results[0].boxes)} products.")

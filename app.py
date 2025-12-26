import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

st.set_page_config(layout="wide")
st.title("🎯 Ultra-Accurate SKU Detection")

@st.cache_resource
def load_sahi_model():
    # 1. Download the specialized retail weights
    model_path = hf_hub_download(repo_id="foduucom/product-detection-in-shelf-yolov8", 
                                 filename="best.pt")
    
    # 2. Wrap the model in a SAHI detector
    detection_model = AutoDetectionModel.from_model_type(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.15, # Detect even faint objects
        device="cpu" # Use "cuda" if you have a GPU
    )
    return detection_model

model = load_sahi_model()
uploaded_file = st.file_uploader("Upload shelf image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # 3. Perform Sliced Inference
    # slice_height/width=400: Breaks the image into 400x400 tiles
    # overlap_ratio=0.2: Ensures products on the edge of a tile aren't missed
    result = get_sliced_prediction(
        np.array(image),
        model,
        slice_height=400,
        slice_width=400,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 4. Draw the high-accuracy red boxes
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    st.subheader(f"Found {len(result.object_prediction_list)} products")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

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
    model_path = hf_hub_download(
        repo_id="foduucom/product-detection-in-shelf-yolov8", 
        filename="best.pt"
    )
    
    # 2. Use 'from_pretrained' and specify 'ultralytics' as the type
    # This is the most stable way to load YOLOv8 into SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", # YOLOv8 is handled by the 'ultralytics' type in SAHI
        model_path=model_path,
        confidence_threshold=0.15,
        device="cpu" 
    )
    return detection_model

# Check if model loads correctly
model = load_sahi_model()

uploaded_file = st.file_uploader("Upload shelf image", type=["jpg","png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # 3. Perform Sliced Inference (The Accuracy Booster)
    result = get_sliced_prediction(
        img_array,
        model,
        slice_height=480, # Size of the small "tiles"
        slice_width=480,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 4. Draw the boxes
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy() # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    st.subheader(f"✅ Found {len(result.object_prediction_list)} individual products")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

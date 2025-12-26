import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

st.set_page_config(page_title="Ultra SKU Detector", layout="wide")

@st.cache_resource
def load_sahi_model():
    # Load the specialized shelf-trained model
    model_path = hf_hub_download(
        repo_id="foduucom/product-detection-in-shelf-yolov8", 
        filename="best.pt"
    )
    
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.20, # Baseline confidence
        device="cpu" 
    )

st.title("🎯 High-Precision Shelf Analytics")
st.sidebar.header("Optimization Settings")

# User-adjustable parameters to fix "box side" alignment
slice_res = st.sidebar.slider("Slice Resolution", 480, 1024, 640)
overlap = st.sidebar.slider("Slice Overlap %", 0.1, 0.4, 0.25)
match_threshold = st.sidebar.slider("Box Merging (NMM) Threshold", 0.1, 0.7, 0.3)

model = load_sahi_model()
uploaded_file = st.file_uploader("Upload shelf photo", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner("Refining product boundaries..."):
        # The Secret Sauce: NMM (Non-Maximum Merging)
        # This merges overlapping detections into one tight box around the SKU
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=slice_res,
            slice_width=slice_res,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type="NMM", 
            postprocess_match_threshold=match_threshold,
            perform_standard_pred=False
        )

    # Drawing the optimized Red Boxes
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
        # Draw with thick red lines for visibility
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    st.subheader(f"✅ Found {len(result.object_prediction_list)} products")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

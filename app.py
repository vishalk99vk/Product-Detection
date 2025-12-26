import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

st.set_page_config(page_title="SKU Detector Pro", layout="wide")

# --- SIDEBAR SETTINGS ---
st.sidebar.title("🛠️ Detection Settings")
slice_size = st.sidebar.slider("Slice Size (Accuracy vs Speed)", 400, 1200, 800, help="Lower = More Accurate but Slower. Higher = Faster.")
conf_level = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.15)
iou_level = st.sidebar.slider("Overlap (IOU) Threshold", 0.1, 0.7, 0.3)

@st.cache_resource
def load_retail_model():
    # Download the best-in-class retail shelf model
    model_path = hf_hub_download(
        repo_id="foduucom/product-detection-in-shelf-yolov8", 
        filename="best.pt"
    )
    # Wrap in SAHI AutoDetectionModel for slicing support
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_level,
        device="cpu" 
    )

st.title("🏪 High-Accuracy SKU Shelf Detector")
st.write("Upload a photo of a shelf to get a red box around every individual product.")

model = load_retail_model()
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    # Update model confidence if slider moved
    model.confidence_threshold = conf_level
    
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner("Processing in slices for maximum accuracy..."):
        # PERFORM SLICED INFERENCE
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            perform_standard_pred=False, # SKIP full image pass to save time
            postprocess_type="NMS",       # Clean up duplicate boxes
            postprocess_match_threshold=iou_level
        )

    # DRAWING RESULTS
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    st.subheader(f"✅ Successfully detected {len(result.object_prediction_list)} products")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Optional: Download Button
    st.sidebar.info(f"Detected: {len(result.object_prediction_list)} SKUs")

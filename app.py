import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide")
st.title("🏪 High-Accuracy SKU Detector")

@st.cache_resource
def load_retail_model():
    # FIXED: The correct repository name for the shelf detection model
    try:
        model_path = hf_hub_download(
            repo_id="foduucom/product-detection-in-shelf-yolov8", 
            filename="best.pt"
        )
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None

model = load_retail_model()

uploaded_file = st.file_uploader("Upload store shelf image", type=["jpg","jpeg","png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Use high resolution (1280) to ensure small SKUs are detected
    results = model.predict(source=img_array, conf=0.15, iou=0.3, imgsz=1280)
    result = results[0]

    if len(result.boxes) == 0:
        st.warning("No products detected. Try adjusting the lighting or image quality.")
    else:
        # Convert RGB to BGR for drawing with OpenCV
        annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw a thick Red box (BGR: 0, 0, 255) around the SKU
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        st.subheader(f"Detected {len(result.boxes)} SKUs")
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

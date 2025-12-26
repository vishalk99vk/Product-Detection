import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from huggingface_hub import hf_hub_download # Add this import

st.set_page_config(layout="wide")
st.title("🏪 High-Accuracy Retail SKU Detector")

@st.cache_resource
def load_retail_model():
    # 1. Download the specific weight file from the Hugging Face repo
    # The filename 'best.pt' is standard for this specific repo
    model_path = hf_hub_download(repo_id="foduucom/shelf-object-detection-yolov8", 
                                 filename="best.pt")
    
    # 2. Load the model using the local path returned by hf_hub_download
    return YOLO(model_path)

model = load_retail_model()

uploaded_file = st.file_uploader("Upload store shelf image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Note: imgsz=1280 is used to maintain high accuracy for small items
    results = model.predict(source=img_array, conf=0.15, iou=0.3, imgsz=1280)
    result = results[0]

    if len(result.boxes) == 0:
        st.warning("No products detected. Try adjusting the image or confidence.")
    else:
        # Create a copy for drawing
        annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw solid RED box (BGR format)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        st.subheader(f"Detected {len(result.boxes)} SKUs")
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

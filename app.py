import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🏪 Store Image Object Annotation")

uploaded_file = st.file_uploader(
    "Upload a store image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # 🔴 FORCE lower confidence + enable boxes
    results = model.predict(
        source=img_array,
        conf=0.10,     # LOWER confidence
        iou=0.45,
        device="cpu"
    )

    # 🔴 CHECK detections
    if len(results[0].boxes) == 0:
        st.warning("⚠️ No objects detected. Try a different image.")
    else:
        annotated_img = results[0].plot()

        with col2:
            st.subheader("Annotated Image")
            st.image(annotated_img, use_column_width=True)

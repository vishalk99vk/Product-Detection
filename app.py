import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # lightweight & fast

model = load_model()

st.title("🏪 Store Image Annotation AI")
st.write("Upload a store image and AI will automatically draw annotations.")

uploaded_file = st.file_uploader(
    "Upload Store Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Run YOLO detection
    results = model(img_array)

    # Draw annotations
    annotated_img = results[0].plot()

    st.subheader("Annotated Image")
    st.image(annotated_img, use_column_width=True)

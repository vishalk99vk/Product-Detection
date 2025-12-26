import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

st.set_page_config(layout="wide")
st.title("🏪 Red Boxes Around All Detected Products (Demo)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # default YOLO model

model = load_model()

uploaded_file = st.file_uploader("Upload store shelf image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Run YOLO detection
    results = model.predict(source=img_array, conf=0.15, iou=0.5)
    result = results[0]

    if len(result.boxes) == 0:
        st.warning("No objects detected.")
    else:
        annotated_img = img_array.copy()

        # Draw one red box per detected object
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Red box

        st.subheader("Annotated Image")
        st.image(annotated_img, use_column_width=True)

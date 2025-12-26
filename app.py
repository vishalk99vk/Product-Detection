import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

st.set_page_config(layout="wide")
st.title("🏪 Temporary Red Box Around All Products")

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
        # Merge all detected boxes into one big red box
        x_min = min([int(box.xyxy[0][0]) for box in result.boxes])
        y_min = min([int(box.xyxy[0][1]) for box in result.boxes])
        x_max = max([int(box.xyxy[0][2]) for box in result.boxes])
        y_max = max([int(box.xyxy[0][3]) for box in result.boxes])

        annotated_img = img_array.copy()
        cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)  # Red box

        st.subheader("Annotated Image (All Products Highlighted)")
        st.image(annotated_img, use_column_width=True)

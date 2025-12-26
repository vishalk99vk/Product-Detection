import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("🏪 Retail Shelf Annotation (Red Boxes)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # default YOLO model

model = load_model()

uploaded_file = st.file_uploader("Upload store image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    results = model.predict(source=img_array, conf=0.15, iou=0.5)  # low conf for more boxes
    result = results[0]

    if len(result.boxes) == 0:
        st.warning("No objects detected.")
    else:
        # Draw red boxes for all detected objects
        annotated_img = img_array.copy()
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw red rectangle
            import cv2
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255,0,0), 3)

        st.subheader("Annotated Image")
        st.image(annotated_img, use_column_width=True)

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld # Use YOLO-World for any-object detection
import cv2

st.set_page_config(layout="wide")
st.title("🏪 Full SKU Detection (Red Boxes)")

@st.cache_resource
def load_model():
    # YOLO-World is much better at finding generic "products"
    model = YOLOWorld("yolov8s-world.pt") 
    # Define what a 'product' is for the model
    model.set_classes(["product pack"]) 
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload shelf image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # imgsz=1024 is critical for small items on a shelf
    results = model.predict(source=img_array, conf=0.15, iou=0.3, imgsz=1024)
    result = results[0]

    annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Draw the Red Box (BGR: 0, 0, 255)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.success(f"Detected {len(result.boxes)} individual SKUs.")

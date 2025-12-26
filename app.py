import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Retail Object Detection", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # DEFAULT PRETRAINED MODEL

model = load_model()

st.title("🏪 Retail Store Object Detection (Default Model)")
st.write("Using YOLOv8 pretrained model for pipeline verification")

uploaded_file = st.file_uploader(
    "Upload store image (France / UK)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # 🔴 Run inference
    results = model.predict(
        source=img_array,
        conf=0.15,   # LOW confidence so something shows
        iou=0.5
    )

    result = results[0]

    # 🔍 Debug info
    st.subheader("Detection Summary")
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"• {label} → {conf:.2f}")
    else:
        st.warning("No objects detected by default model.")

    # 🖼️ Draw boxes
    if result.boxes is not None and len(result.boxes) > 0:
        annotated_img = result.plot()
        with col2:
            st.subheader("Annotated Image")
            st.image(annotated_img, use_column_width=True)

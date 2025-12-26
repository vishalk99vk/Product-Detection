import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import json

# ---------------- CONFIG ----------------
st.set_page_config(page_title="YOLOv8 Store Annotation", layout="wide")
st.title("🛒 Store Product Annotation & Shelf Grouping")

# Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
img_size = st.sidebar.selectbox("Image Size", [640, 960, 1280], index=2)
shelf_gap = st.sidebar.slider("Shelf Gap (px)", 30, 150, 80)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# ---------------- FUNCTIONS ----------------
def group_by_shelf(detections, gap):
    """
    Group products into shelves using vertical center clustering
    """
    detections = sorted(detections, key=lambda x: x["y_center"])
    shelf_id = 0
    prev_y = None

    for d in detections:
        if prev_y is None or abs(d["y_center"] - prev_y) > gap:
            shelf_id += 1
        d["shelf_id"] = shelf_id
        prev_y = d["y_center"]

    return detections


def export_coco(detections, img_w, img_h):
    coco = {
        "images": [{
            "id": 1,
            "width": img_w,
            "height": img_h,
            "file_name": "store_image.jpg"
        }],
        "annotations": [],
        "categories": []
    }

    cat_map = {}
    ann_id = 1

    for d in detections:
        if d["class"] not in cat_map:
            cat_id = len(cat_map) + 1
            cat_map[d["class"]] = cat_id
            coco["categories"].append({
                "id": cat_id,
                "name": d["class"]
            })

        x, y, w, h = d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]

        coco["annotations"].append({
            "id": ann_id,
            "image_id": 1,
            "category_id": cat_map[d["class"]],
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
            "shelf_id": d["shelf_id"]
        })
        ann_id += 1

    return coco


# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Store Image", ["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # ---------------- YOLO INFERENCE ----------------
    with st.spinner("Detecting products..."):
        results = model(img_np, conf=conf_threshold, imgsz=img_size)

    boxes = results[0].boxes
    detections = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        y_center = int((y1 + y2) / 2)
        cls_id = int(box.cls[0])

        detections.append({
            "class": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 3),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "y_center": y_center
        })

    # ---------------- SHELF GROUPING ----------------
    detections = group_by_shelf(detections, shelf_gap)

    # ---------------- DRAW ANNOTATIONS ----------------
    annotated = img_np.copy()
    shelf_colors = {}

    for d in detections:
        shelf = d["shelf_id"]
        if shelf not in shelf_colors:
            shelf_colors[shelf] = tuple(np.random.randint(0, 255, 3).tolist())

        color = shelf_colors[shelf]

        cv2.rectangle(
            annotated,
            (d["x1"], d["y1"]),
            (d["x2"], d["y2"]),
            color,
            2
        )

        cv2.putText(
            annotated,
            f"{d['class']} | Shelf {shelf}",
            (d["x1"], d["y1"] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    with col2:
        st.subheader("Shelf-wise Annotated Image")
        st.image(annotated, use_container_width=True)

    # ---------------- TABLE ----------------
    st.subheader("Shelf-wise Product Data")
    st.dataframe(detections, use_container_width=True)

    # ---------------- EXPORTS ----------------
    json_data = json.dumps(detections, indent=2)
    coco_data = json.dumps(export_coco(detections, w, h), indent=2)

    col3, col4 = st.columns(2)

    with col3:
        st.download_button(
            "⬇️ Download JSON",
            json_data,
            file_name="detections_shelf.json",
            mime="application/json"
        )

    with col4:
        st.download_button(
            "⬇️ Download COCO",
            coco_data,
            file_name="detections_shelf_coco.json",
            mime="application/json"
        )

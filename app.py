import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json
import sys

st.set_page_config(page_title="YOLOv8 Store Detection", layout="wide")
st.title("🛒 YOLOv8 Store Product Detection")

st.caption(f"Python version: {sys.version}")

# Sidebar
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.3, 0.05)
imgsz = st.sidebar.selectbox("Image Size", [640, 960, 1280], index=0)
shelf_gap = st.sidebar.slider("Shelf Gap (px)", 40, 200, 90)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def group_by_shelf(dets, gap):
    dets = sorted(dets, key=lambda x: x["y_center"])
    shelf, prev = 0, None
    for d in dets:
        if prev is None or abs(d["y_center"] - prev) > gap:
            shelf += 1
        d["shelf_id"] = shelf
        prev = d["y_center"]
    return dets

def export_coco(dets, w, h):
    coco = {"images":[{"id":1,"width":w,"height":h,"file_name":"image.jpg"}],
            "annotations":[], "categories":[]}
    cat_map, ann_id = {}, 1
    for d in dets:
        if d["class"] not in cat_map:
            cat_map[d["class"]] = len(cat_map) + 1
            coco["categories"].append({"id":cat_map[d["class"]],"name":d["class"]})
        x,y = d["x1"], d["y1"]
        bw,bh = d["x2"]-d["x1"], d["y2"]-d["y1"]
        coco["annotations"].append({
            "id":ann_id,"image_id":1,
            "category_id":cat_map[d["class"]],
            "bbox":[x,y,bw,bh],
            "area":bw*bh,"iscrowd":0,
            "shelf_id":d["shelf_id"]
        })
        ann_id+=1
    return coco

file = st.file_uploader("Upload store image", ["jpg","jpeg","png"])

if file:
    img = Image.open(file).convert("RGB")
    arr = np.array(img)
    h,w = arr.shape[:2]

    col1,col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original", use_container_width=True)

    with st.spinner("Running YOLOv8..."):
        results = model(arr, conf=conf, imgsz=imgsz)

    boxes = results[0].boxes
    dets = []

    for b in boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        dets.append({
            "class": model.names[int(b.cls[0])],
            "confidence": round(float(b.conf[0]),3),
            "x1":x1,"y1":y1,"x2":x2,"y2":y2,
            "y_center": int((y1+y2)/2)
        })

    dets = group_by_shelf(dets, shelf_gap)

    with col2:
        st.image(results[0].plot(), caption="Detected Products", use_container_width=True)

    st.subheader("Detections")
    st.dataframe(dets, use_container_width=True)

    st.download_button(
        "⬇️ Download JSON",
        json.dumps(dets, indent=2),
        "detections.json",
        "application/json"
    )

    st.download_button(
        "⬇️ Download COCO",
        json.dumps(export_coco(dets,w,h), indent=2),
        "detections_coco.json",
        "application/json"
    )

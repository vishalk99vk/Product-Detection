import streamlit as st
import os
import cv2
import numpy as np
import yaml
from PIL import Image
from streamlit_image_annotation import detection
from ultralytics import YOLO

# --- DIRECTORY SETUP ---
BASE_DIR = "sku_training_data"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
for p in [IMG_DIR, LBL_DIR]:
    os.makedirs(p, exist_ok=True)

st.set_page_config(page_title="Retail SKU AI", layout="wide")
panel = st.sidebar.selectbox("Go to", ["🏗️ Training Panel", "👤 Client Panel"])

# --- PANEL 1: TRAINING ---
if panel == "🏗️ Training Panel":
    st.title("🏗️ Training Panel: Label Your Products")
    
    # Upload training images
    uploads = st.file_uploader("Upload Images", accept_multiple_files=True, key="train_up")
    if uploads:
        for f in uploads:
            with open(os.path.join(IMG_DIR, f.name), "wb") as file:
                file.write(f.getbuffer())
    
    img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if img_list:
        selected_img = st.selectbox("Select image to annotate", img_list)
        img_path = os.path.join(IMG_DIR, selected_img)
        
        # Load image for dimension calculation
        img_cv = cv2.imread(img_path)
        h, w, _ = img_cv.shape

        # FIXED: Robust initialization to prevent TypeError: zip() argument must be iterable
        # Passing empty lists [] ensures the component internal logic doesn't crash
        new_annotations = detection(
            image_path=img_path, 
            label_list=["product"], 
            bboxes=[], # Explicit empty list
            labels=[], # Explicit empty list
            key=selected_img
        )

        if st.button("Submit Annotations"):
            if new_annotations:
                txt_name = os.path.splitext(selected_img)[0] + ".txt"
                with open(os.path.join(LBL_DIR, txt_name), "w") as f:
                    for ann in new_annotations:
                        bx = ann['bbox'] # [x_min, y_min, width, height]
                        # YOLO Normalization
                        xc, yc = (bx[0] + bx[2]/2) / w, (bx[1] + bx[3]/2) / h
                        nw, nh = bx[2] / w, bx[3] / h
                        f.write(f"0 {xc} {yc} {nw} {nh}\n")
                st.success(f"Saved {len(new_annotations)} boxes!")

    st.divider()
    if st.button("🔥 Start Training"):
        st.info("Configuring data.yaml and launching YOLOv8...")
        config = {'path': os.path.abspath(BASE_DIR), 'train': 'images', 'val': 'images', 'names': {0: 'product'}}
        with open(os.path.join(BASE_DIR, 'data.yaml'), 'w') as f:
            yaml.dump(config, f)
            
        model = YOLO("yolov8n.pt")
        # Training logic is isolated only to this button click
        model.train(data=os.path.join(BASE_DIR, 'data.yaml'), epochs=5, imgsz=640)
        st.success("Training Complete!")

# --- PANEL 2: CLIENT ---
else:
    st.title("👤 Client Panel: Automated Detection")
    
    # Use custom model if trained, else base model
    model_path = "runs/detect/train/weights/best.pt"
    client_model = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt")
    
    test_file = st.file_uploader("Upload Store Photo", key="client_up")
    if test_file:
        img_pil = Image.open(test_file)
        results = client_model.predict(img_pil, conf=0.25)
        
        # COLOR FIX: YOLO plots in BGR, Streamlit shows in RGB
        res_bgr = results[0].plot()
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        
        st.image(res_rgb, caption="AI Automated Detection", use_container_width=True)

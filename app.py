import streamlit as st
import os
import cv2
import numpy as np
import yaml
from PIL import Image
from streamlit_image_annotation import detection
from ultralytics import YOLO

# --- FILE SYSTEM SETUP ---
# Creating a clean folder structure for your project
BASE_DIR = "sku_project_data"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
for p in [IMG_DIR, LBL_DIR]:
    os.makedirs(p, exist_ok=True)

st.set_page_config(page_title="Retail AI: Train & Deploy", layout="wide")

# Sidebar navigation creates the isolation
page = st.sidebar.selectbox("Select Workspace", ["🏗️ Training Panel", "👤 Client Panel"])

# --- PANEL 1: TRAINING LOGIC ---
if page == "🏗️ Training Panel":
    st.title("🏗️ SKU Annotation & Training")
    st.write("Upload shelf images and draw red boxes around the products to teach the AI.")

    # A. Uploader
    uploads = st.file_uploader("Upload Photos", accept_multiple_files=True, key="train_upload")
    if uploads:
        for f in uploads:
            with open(os.path.join(IMG_DIR, f.name), "wb") as file:
                file.write(f.getbuffer())
    
    # B. Annotation Tool
    img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if img_list:
        selected = st.selectbox("Select image to label", img_list)
        img_path = os.path.join(IMG_DIR, selected)
        img_cv = cv2.imread(img_path)
        h, w, _ = img_cv.shape

        # Initialize the tool. initial_bboxes=[] prevents the TypeError.
        new_anns = detection(
            image_path=img_path, 
            label_list=["product"], 
            initial_bboxes=[], 
            key=selected
        )

        if st.button("Submit Label Data"):
            if new_anns:
                txt_name = os.path.splitext(selected)[0] + ".txt"
                with open(os.path.join(LBL_DIR, txt_name), "w") as f:
                    for ann in new_anns:
                        bx = ann['bbox'] # [x, y, w, h]
                        # Convert to YOLO format
                        xc, yc = (bx[0] + bx[2]/2) / w, (bx[1] + bx[3]/2) / h
                        nw, nh = bx[2] / w, bx[3] / h
                        f.write(f"0 {xc} {yc} {nw} {nh}\n")
                st.success(f"Saved {len(new_anns)} boxes for {selected}!")

    # C. Training Trigger (Strictly isolated here)
    st.divider()
    if st.button("🚀 Start Training Now"):
        with st.spinner("Training... This will only run on this page."):
            # Create the config file for YOLO
            config = {
                'path': os.path.abspath(BASE_DIR),
                'train': 'images', 'val': 'images',
                'names': {0: 'product'}
            }
            with open(os.path.join(BASE_DIR, 'data.yaml'), 'w') as f:
                yaml.dump(config, f)
            
            model = YOLO("yolov8n.pt")
            model.train(data=os.path.join(BASE_DIR, 'data.yaml'), epochs=5, imgsz=640)
            st.success("Training finished! The Client Panel is now updated.")

# --- PANEL 2: CLIENT LOGIC ---
elif page == "👤 Client Panel":
    st.title("👤 Retailer Automated Detection")
    st.write("Upload a photo to see the AI find products using the latest trained model.")
    
    # Search for the most recently trained model
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        st.warning("Custom model not found yet. Using general weights.")
        model_path = "yolov8n.pt"

    client_model = YOLO(model_path)
    client_upload = st.file_uploader("Upload Store Shelf Photo", key="client_upload")
    
    if client_upload:
        img_pil = Image.open(client_upload)
        with st.spinner("Scanning shelf..."):
            results = client_model.predict(img_pil, conf=0.25)
            
            # COLOR FIX: YOLO.plot() is BGR, Streamlit needs RGB
            res_bgr = results[0].plot()
            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
            
            st.image(res_rgb, caption="AI Detection Result", use_container_width=True)
            st.balloons()

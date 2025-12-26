import streamlit as st
import os
import cv2
import numpy as np
import yaml
from PIL import Image
from streamlit_image_annotation import detection
from ultralytics import YOLO

# --- DIRECTORY SETUP ---
BASE_DIR = "retail_project"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")

for p in [IMG_DIR, LBL_DIR]:
    os.makedirs(p, exist_ok=True)

st.set_page_config(page_title="SKU Training & Client Portal", layout="wide")
panel = st.sidebar.radio("Navigation", ["🏗️ Training Panel", "👤 Client Panel"])

# --- PANEL 1: TRAINING (Manual Box Drawing) ---
if panel == "🏗️ Training Panel":
    st.title("🏗️ SKU Training Workshop")
    
    uploads = st.file_uploader("1. Upload Training Photos", accept_multiple_files=True)
    if uploads:
        for f in uploads:
            with open(os.path.join(IMG_DIR, f.name), "wb") as file:
                file.write(f.getbuffer())
    
    img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if img_list:
        selected_img = st.selectbox("2. Select image to annotate", img_list)
        img_path = os.path.join(IMG_DIR, selected_img)
        
        # Load image details
        img_cv = cv2.imread(img_path)
        h, w, _ = img_cv.shape

        # FIXED: Explicitly pass empty lists to prevent TypeError in zip()
        new_annotations = detection(
            image_path=img_path, 
            label_list=["product"], 
            bboxes=[], # Prevents the library from crashing on first load
            labels=[], 
            key=selected_img
        )

        if st.button("3. Submit Annotations"):
            if new_annotations:
                txt_name = os.path.splitext(selected_img)[0] + ".txt"
                with open(os.path.join(LBL_DIR, txt_name), "w") as f:
                    for ann in new_annotations:
                        bx = ann['bbox'] # [x_min, y_min, width, height]
                        # YOLO Normalization: (center_x, center_y, width, height)
                        xc = (bx[0] + bx[2]/2) / w
                        yc = (bx[1] + bx[3]/2) / h
                        nw = bx[2] / w
                        nh = bx[3] / h
                        f.write(f"0 {xc} {yc} {nw} {nh}\n")
                st.success(f"Saved {len(new_annotations)} boxes for {selected_img}")

    st.divider()
    if st.button("🚀 4. Start Training"):
        st.info("Generating data.yaml and starting YOLOv8 training...")
        # Create data.yaml automatically
        config = {'path': os.path.abspath(BASE_DIR), 'train': 'images', 'val': 'images', 'names': {0: 'product'}}
        with open(os.path.join(BASE_DIR, 'data.yaml'), 'w') as f:
            yaml.dump(config, f)
            
        model = YOLO("yolov8n.pt")
        model.train(data=os.path.join(BASE_DIR, 'data.yaml'), epochs=5, imgsz=640)

# --- PANEL 2: CLIENT (Automated Detection) ---
else:
    st.title("👤 Client Detection Panel")
    
    # Check for trained model weights
    best_model = "runs/detect/train/weights/best.pt"
    model_path = best_model if os.path.exists(best_model) else "yolov8n.pt"
    
    client_model = YOLO(model_path)
    test_file = st.file_uploader("Upload Store Image")
    
    if test_file:
        img_pil = Image.open(test_file)
        results = client_model.predict(img_pil, imgsz=640, conf=0.25)
        
        # THE COLOR FIX: YOLO.plot() returns BGR, Streamlit needs RGB
        res_bgr = results[0].plot()
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        
        st.image(res_rgb, caption="Automated Product Detection", use_container_width=True)

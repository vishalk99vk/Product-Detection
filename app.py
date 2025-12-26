import streamlit as st
import os
import cv2
import numpy as np
import yaml
from PIL import Image
from streamlit_image_annotation import detection
from ultralytics import YOLO

# --- DIRECTORY SETUP ---
BASE_DIR = "sku_data"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
for p in [IMG_DIR, LBL_DIR]:
    os.makedirs(p, exist_ok=True)

st.set_page_config(page_title="SKU AI: Train & Deploy", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'annotations' not in st.session_state:
    st.session_state['annotations'] = {}

panel = st.sidebar.selectbox("Navigation", ["🏗️ Training Panel", "👤 Client Panel"])

# --- PANEL 1: TRAINING ---
if panel == "🏗️ Training Panel":
    st.title("🏗️ Training Panel")
    st.write("Draw red boxes around products. These will be used to train your custom model.")

    uploads = st.file_uploader("Upload Training Images", accept_multiple_files=True, key="train_up")
    if uploads:
        for f in uploads:
            with open(os.path.join(IMG_DIR, f.name), "wb") as file:
                file.write(f.getbuffer())
    
    img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if img_list:
        selected_img = st.selectbox("Select image to label", img_list)
        img_path = os.path.join(IMG_DIR, selected_img)
        img_cv = cv2.imread(img_path)
        h, w, _ = img_cv.shape

        # FIXED: Explicitly define lists to prevent the TypeError in the library's zip() function
        # This ensures the component always has an iterable object to work with.
        new_anns = detection(
            image_path=img_path, 
            label_list=["product"], 
            bboxes=[], # DO NOT REMOVE: Prevents TypeError
            labels=[], # DO NOT REMOVE: Prevents TypeError
            key=selected_img
        )

        if st.button("Submit Annotations"):
            if new_anns is not None:
                # Save to session state so it persists
                st.session_state['annotations'][selected_img] = new_anns
                
                # Write YOLO Label File
                txt_name = os.path.splitext(selected_img)[0] + ".txt"
                with open(os.path.join(LBL_DIR, txt_name), "w") as f:
                    for ann in new_anns:
                        bx = ann['bbox'] # [x_min, y_min, width, height]
                        xc, yc = (bx[0] + bx[2]/2) / w, (bx[1] + bx[3]/2) / h
                        nw, nh = bx[2] / w, bx[3] / h
                        f.write(f"0 {xc} {yc} {nw} {nh}\n")
                st.success(f"Saved {len(new_anns)} boxes for {selected_img}")

    st.divider()
    if st.button("🔥 Start Training"):
        st.info("Generating configuration and starting YOLOv8...")
        config = {'path': os.path.abspath(BASE_DIR), 'train': 'images', 'val': 'images', 'names': {0: 'product'}}
        with open(os.path.join(BASE_DIR, 'data.yaml'), 'w') as f:
            yaml.dump(config, f)
            
        model = YOLO("yolov8n.pt")
        # Training is isolated to this specific block
        model.train(data=os.path.join(BASE_DIR, 'data.yaml'), epochs=5, imgsz=640)
        st.success("Training Finished!")

# --- PANEL 2: CLIENT ---
else:
    st.title("👤 Client Panel")
    
    # Check for the custom trained model
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        st.warning("Custom model not found. Using default weights.")
        model_path = "yolov8n.pt"

    client_model = YOLO(model_path)
    test_file = st.file_uploader("Upload Image for Detection", key="client_up")
    
    if test_file:
        img_pil = Image.open(test_file)
        results = client_model.predict(img_pil, conf=0.25)
        
        # COLOR FIX: YOLO plots in BGR, Streamlit displays in RGB
        res_bgr = results[0].plot()
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        
        st.image(res_rgb, caption="AI Detection Result", use_container_width=True)

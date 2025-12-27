import streamlit as st
import os
import cv2
import numpy as np
import yaml
import glob
from PIL import Image
from streamlit_image_annotation import detection
from ultralytics import YOLO

# --- CONFIGURATION & PATHS ---
BASE_DIR = "retail_project_data"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")

for path in [IMG_DIR, LBL_DIR]:
    os.makedirs(path, exist_ok=True)

st.set_page_config(page_title="SKU Training & Inference", layout="wide")
panel = st.sidebar.radio("Navigation", ["🏗️ Training Panel", "👤 Client Panel"])

# --- HELPER: Generate data.yaml ---
def update_data_yaml():
    data_config = {
        'path': os.path.abspath(BASE_DIR),
        'train': 'images',
        'val': 'images', 
        'names': {0: 'product'}
    }
    with open(os.path.join(BASE_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_config, f)

# --- PANEL 1: TRAINING ---
if panel == "🏗️ Training Panel":
    st.title("🏗️ Model Training Workshop")
    st.info("Step 1: Upload images. Step 2: Draw Red Boxes. Step 3: Hit Submit. Step 4: Run Training.")

    uploads = st.file_uploader("Upload SKU Photos", accept_multiple_files=True, key="train_loader")
    if uploads:
        for f in uploads:
            with open(os.path.join(IMG_DIR, f.name), "wb") as file:
                file.write(f.getbuffer())
    
    img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if img_list:
        selected_img = st.selectbox("Select image to annotate", img_list)
        img_full_path = os.path.join(IMG_DIR, selected_img)
        img_cv = cv2.imread(img_full_path)
        h, w, _ = img_cv.shape

        # FIX: Explicit empty lists [] prevent the "zip()" TypeError
        new_annotations = detection(
            image_path=img_full_path, 
            label_list=["product"], 
            bboxes=[], 
            labels=[], 
            key=selected_img
        )

        if st.button("Submit Annotations"):
            if new_annotations:
                txt_name = os.path.splitext(selected_img)[0] + ".txt"
                with open(os.path.join(LBL_DIR, txt_name), "w") as f:
                    for ann in new_annotations:
                        bx = ann['bbox'] # [x_min, y_min, width, height]
                        xc, yc = (bx[0] + bx[2]/2) / w, (bx[1] + bx[3]/2) / h
                        nw, nh = bx[2] / w, bx[3] / h
                        f.write(f"0 {xc} {yc} {nw} {nh}\n")
                
                update_data_yaml()
                st.success(f"Successfully saved {len(new_annotations)} boxes!")
            else:
                st.warning("Please draw at least one box before submitting.")

    st.divider()
    if st.button("🔥 Run Training"):
        with st.spinner("Training on your custom SKUs..."):
            model = YOLO("yolov8n.pt") 
            # Note: YOLO increments folders (train, train2, etc.)
            model.train(data=os.path.join(BASE_DIR, 'data.yaml'), epochs=5, imgsz=640)
            st.success("Training Finished!")

# --- PANEL 2: CLIENT ---
else:
    st.title("👤 Client Detection Panel")
    
    # FIX: Automatically find the latest trained 'best.pt' file
    model_files = glob.glob("runs/detect/train*/weights/best.pt")
    if model_files:
        model_path = max(model_files, key=os.path.getctime)
        st.success(f"Using Latest Model: {model_path}")
    else:
        st.warning("No custom model found. Using standard YOLO weights.")
        model_path = "yolov8n.pt"

    client_model = YOLO(model_path)
    test_file = st.file_uploader("Upload Image for Detection", key="client_loader")
    
    if test_file:
        img = Image.open(test_file)
        results = client_model.predict(img, imgsz=640, conf=0.25)
        
        # FIX: Convert BGR (OpenCV) to RGB (Streamlit) to fix the Blue Color
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        st.image(res_rgb, caption="AI Automated Detection", use_container_width=True)

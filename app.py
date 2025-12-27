import streamlit as st
import os
import cv2
import numpy as np
import yaml
from PIL import Image
from streamlit_image_annotation import detection
from ultralytics import YOLO

# --- 1. DIRECTORY SETUP ---
BASE_DIR = os.path.abspath("retail_data")
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
MODEL_EXPORT_DIR = os.path.join(BASE_DIR, "trained_model")

for path in [IMG_DIR, LBL_DIR, MODEL_EXPORT_DIR]:
    os.makedirs(path, exist_ok=True)

st.set_page_config(page_title="SKU Training & Inference (YOLOv12)", layout="wide")
panel = st.sidebar.radio("Navigation", ["🏗️ Training Panel", "👤 Client Panel"])

# --- 2. PANEL 1: TRAINING ---
if panel == "🏗️ Training Panel":
    st.title("🏗️ Model Training Workshop (YOLOv12)")
    
    uploads = st.file_uploader("Upload SKU Photos", accept_multiple_files=True)
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

        new_annotations = detection(
            image_path=img_full_path, 
            label_list=["product"], 
            bboxes=[], labels=[], 
            key=selected_img
        )

        if st.button("Submit Annotations"):
            if new_annotations:
                txt_name = os.path.splitext(selected_img)[0] + ".txt"
                with open(os.path.join(LBL_DIR, txt_name), "w") as f:
                    for ann in new_annotations:
                        bx = ann['bbox']
                        xc, yc = (bx[0] + bx[2]/2) / w, (bx[1] + bx[3]/2) / h
                        nw, nh = bx[2] / w, bx[3] / h
                        f.write(f"0 {xc} {yc} {nw} {nh}\n")
                
                config = {'path': BASE_DIR, 'train': 'images', 'val': 'images', 'names': {0: 'product'}}
                with open(os.path.join(BASE_DIR, 'data.yaml'), 'w') as f:
                    yaml.dump(config, f)
                st.success(f"Saved {len(new_annotations)} boxes!")

    st.divider()
    if st.button("🔥 Run Training"):
        with st.spinner("Training YOLOv12..."):
            # CHANGED: Using yolov12n.pt instead of yolov8n.pt
            model = YOLO("yolov12n.pt") 
            model.train(
                data=os.path.join(BASE_DIR, 'data.yaml'), 
                epochs=5, 
                project=MODEL_EXPORT_DIR, 
                name="run", 
                exist_ok=True 
            )
            st.success("Training Finished with YOLOv12!")

# --- 3. PANEL 2: CLIENT ---
else:
    st.title("👤 Client Detection Panel")
    
    fixed_path = os.path.join(MODEL_EXPORT_DIR, "run", "weights", "best.pt")
    
    if os.path.exists(fixed_path):
        model_path = fixed_path
        st.success(f"✅ Custom YOLOv12 Model Found")
    else:
        st.warning("⚠️ No custom model found. Using base YOLOv12n weights.")
        # CHANGED: Defaulting to yolov12n.pt
        model_path = "yolov12n.pt"

    client_model = YOLO(model_path)
    test_file = st.file_uploader("Upload Image")
    
    if test_file:
        img = Image.open(test_file)
        # YOLOv12 inference works exactly the same way
        results = client_model.predict(img, conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, use_container_width=True)

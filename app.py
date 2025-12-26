import streamlit as st
import os
import cv2
import numpy as np
import yaml
from PIL import Image
from streamlit_image_annotation import detection
from ultralytics import YOLO

# --- CONFIGURATION & PATHS ---
BASE_DIR = "retail_data"
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
        'val': 'images', # Using same images for val in this simple example
        'names': {0: 'product'}
    }
    with open(os.path.join(BASE_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_config, f)

# --- PANEL 1: TRAINING ---
if panel == "🏗️ Training Panel":
    st.title("🏗️ Model Training Workshop")
    st.info("Step 1: Upload images. Step 2: Draw Red Boxes. Step 3: Hit Submit. Step 4: Run Training.")

    uploads = st.file_uploader("Upload SKU Photos", accept_multiple_files=True)
    if uploads:
        for f in uploads:
            with open(os.path.join(IMG_DIR, f.name), "wb") as file:
                file.write(f.getbuffer())
    
    img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if img_list:
        selected_img = st.selectbox("Select image to annotate", img_list)
        img_full_path = os.path.join(IMG_DIR, selected_img)
        
        # Load image to get width/height for YOLO normalization
        img_cv = cv2.imread(img_full_path)
        h, w, _ = img_cv.shape

        # CRITICAL FIX: Ensure streamlit-image-annotation has data to zip
        # We pass empty lists if no previous labels exist to avoid the TypeError
        new_annotations = detection(
            image_path=img_full_path, 
            label_list=["product"], 
            bboxes=[], # Initial bboxes
            labels=[], # Initial labels
            key=selected_img
        )

        if st.button("Submit Annotations"):
            if new_annotations is not None and len(new_annotations) > 0:
                txt_name = os.path.splitext(selected_img)[0] + ".txt"
                with open(os.path.join(LBL_DIR, txt_name), "w") as f:
                    for ann in new_annotations:
                        # Convert [x, y, w, h] to YOLO [x_center, y_center, width, height]
                        bx = ann['bbox'] # [x_min, y_min, width, height]
                        xc = (bx[0] + bx[2]/2) / w
                        yc = (bx[1] + bx[3]/2) / h
                        nw = bx[2] / w
                        nh = bx[3] / h
                        f.write(f"0 {xc} {yc} {nw} {nh}\n")
                
                update_data_yaml()
                st.success(f"Successfully saved {len(new_annotations)} boxes for {selected_img}")
            else:
                st.warning("Please draw at least one box before submitting.")

    st.divider()
    if st.button("🔥 Run Training (Fine-Tune Model)"):
        with st.spinner("Training on your custom SKUs..."):
            model = YOLO("yolov8n.pt") # Start with a lightweight base
            model.train(data=os.path.join(BASE_DIR, 'data.yaml'), epochs=5, imgsz=640)
            st.success("Training Finished! New 'best.pt' is available in the Client Panel.")

# --- PANEL 2: CLIENT ---
else:
    st.title("👤 Client Detection Panel")
    
    # Check for trained model, fallback to base if not found
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        st.warning("No custom model found. Using standard YOLO weights.")
        model_path = "yolov8n.pt"

    client_model = YOLO(model_path)
    test_file = st.file_uploader("Upload New Image for Automated Detection")
    
    if test_file:
        img = Image.open(test_file)
        results = client_model.predict(img, imgsz=1280) # High res for accuracy
        
        # Plot and display
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="AI Automated Detection", use_container_width=True)

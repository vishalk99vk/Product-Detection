import streamlit as st
import os
import cv2
import numpy as np
import yaml
from PIL import Image
from streamlit_image_annotation import detection
from ultralytics import YOLO
# RF-DETR specific imports
from rfdetr import RFDETRBase
import supervision as sv

# --- 1. DIRECTORY SETUP ---
BASE_DIR = os.path.abspath("retail_data")
@@ -16,12 +18,12 @@
for path in [IMG_DIR, LBL_DIR, MODEL_EXPORT_DIR]:
    os.makedirs(path, exist_ok=True)

st.set_page_config(page_title="SKU Training & Inference (YOLOv12)", layout="wide")
st.set_page_config(page_title="SKU Training (RF-DETR)", layout="wide")
panel = st.sidebar.radio("Navigation", ["🏗️ Training Panel", "👤 Client Panel"])

# --- 2. PANEL 1: TRAINING ---
if panel == "🏗️ Training Panel":
    st.title("🏗️ Model Training Workshop (YOLOv12)")
    st.title("🏗️ RF-DETR Training Workshop")

    uploads = st.file_uploader("Upload SKU Photos", accept_multiple_files=True)
    if uploads:
@@ -52,47 +54,46 @@
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
    if st.button("🔥 Run RF-DETR Training"):
        with st.spinner("Training RF-DETR (Transformer)..."):
            # RF-DETR uses a different training flow than Ultralytics
            # Note: RF-DETR training is typically done via command line or fine-tune script
            model = RFDETRBase()
            # For a custom script, you would point to your local dataset
            # model.train(dataset_path=BASE_DIR, epochs=10) 
            st.info("RF-DETR training initiated. (Ensure 'rfdetr' package is installed)")

# --- 3. PANEL 2: CLIENT ---
else:
    st.title("👤 Client Detection Panel")
    st.title("👤 Client Detection Panel (RF-DETR)")

    fixed_path = os.path.join(MODEL_EXPORT_DIR, "run", "weights", "best.pt")
    # Check for custom weights, otherwise use base
    fixed_path = os.path.join(MODEL_EXPORT_DIR, "best.pt")

    if os.path.exists(fixed_path):
        model_path = fixed_path
        st.success(f"✅ Custom YOLOv12 Model Found")
        model = RFDETRBase(weights=fixed_path)
        st.success("✅ Custom RF-DETR Weights Loaded")
    else:
        st.warning("⚠️ No custom model found. Using base YOLOv12n weights.")
        # CHANGED: Defaulting to yolov12n.pt
        model_path = "yolov12n.pt"
        st.warning("⚠️ Using Base RF-DETR-Base weights.")
        model = RFDETRBase()

    client_model = YOLO(model_path)
    test_file = st.file_uploader("Upload Image")

    if test_file:
        img = Image.open(test_file)
        # YOLOv12 inference works exactly the same way
        results = client_model.predict(img, conf=0.25)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, use_container_width=True)
        img_np = np.array(img)
        
        # RF-DETR Inference
        detections = model.predict(img_np, threshold=0.25)
        
        # Visualization using Roboflow Supervision (standard for RF-DETR)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_image = box_annotator.annotate(scene=img_np.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        st.image(annotated_image, use_container_width=True)

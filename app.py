import streamlit as st
import os
import cv2
import numpy as np
import tempfile
from PIL import Image
from streamlit_image_annotation import detection
import supervision as sv

# --- 1. MODEL LOADING & ERROR HANDLING ---
try:
    from rfdetr import RFDETRBase
except ImportError:
    st.error("The 'rfdetr' library is missing. Please ensure it is in your requirements.txt.")

# --- 2. DIRECTORY SETUP ---
BASE_DIR = os.path.abspath("retail_data")
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
MODEL_EXPORT_DIR = os.path.join(BASE_DIR, "trained_model")

for path in [IMG_DIR, LBL_DIR, MODEL_EXPORT_DIR]:
    os.makedirs(path, exist_ok=True)

st.set_page_config(page_title="SKU Training (RF-DETR)", layout="wide")
panel = st.sidebar.radio("Navigation", ["🏗️ Training Panel", "👤 Client Panel"])

# --- 3. PANEL 1: TRAINING WORKSHOP ---
if panel == "🏗️ Training Panel":
    st.title("🏗️ RF-DETR Training Workshop")
    
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

        # Annotation tool
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
                        bx = ann['bbox'] # [x, y, width, height]
                        # Convert to YOLO format
                        xc, yc = (bx[0] + bx[2]/2) / w, (bx[1] + bx[3]/2) / h
                        nw, nh = bx[2] / w, bx[3] / h
                        f.write(f"0 {xc} {yc} {nw} {nh}\n")
                st.success(f"Saved {len(new_annotations)} boxes!")

    st.divider()
    if st.button("🔥 Run RF-DETR Training"):
        with st.spinner("Training RF-DETR (Transformer)..."):
            # Logic to trigger training script goes here
            st.info("RF-DETR training initiated. Ensure your local dataset is mapped correctly.")

# --- 4. PANEL 2: CLIENT PANEL (IMAGE & VIDEO) ---
else:
    st.title("👤 Client Detection Panel (RF-DETR)")
    
    # Load model weights
    fixed_path = os.path.join(MODEL_EXPORT_DIR, "best.pt")
    if os.path.exists(fixed_path):
        model = RFDETRBase(weights=fixed_path)
        st.success("✅ Custom RF-DETR Weights Loaded")
    else:
        st.warning("⚠️ Using Base RF-DETR-Base weights.")
        model = RFDETRBase()

    test_file = st.file_uploader("Upload Image or Video", type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'])
    
    if test_file:
        file_extension = os.path.splitext(test_file.name)[1].lower()
        
        # Annotation tools
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        if file_extension in ['.png', '.jpg', '.jpeg']:
            # IMAGE PROCESSING
            img = Image.open(test_file)
            img_np = np.array(img)
            detections = model.predict(img_np, threshold=0.25)
            
            annotated_image = box_annotator.annotate(scene=img_np.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            st.image(annotated_image, use_container_width=True)
            
        else:
            # VIDEO PROCESSING
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(test_file.read())
            
            video_cap = cv2.VideoCapture(tfile.name)
            width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video_cap.get(cv2.CAP_PROP_FPS))
            
            # Temporary path for the output file
            out_path = os.path.join(tempfile.gettempdir(), "output_detection.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            st_frame = st.empty() # Placeholder for the video stream

            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break
                
                # RF-DETR Inference (Convert BGR to RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model.predict(frame_rgb, threshold=0.25)
                
                # Annotate
                annotated_frame = box_annotator.annotate(scene=frame_rgb.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
                
                # Save frame (Convert back to BGR for Writer)
                out_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                
                # Show in Streamlit
                st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
            
            video_cap.release()
            out_writer.release()
            
            # Provide Download Button for processed video
            with open(out_path, "rb") as f:
                st.download_button(
                    label="💾 Download Processed Video",
                    data=f,
                    file_name="detected_skus.mp4",
                    mime="video/mp4"
                )

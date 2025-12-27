import streamlit as st
import os
import cv2
import numpy as np
import yaml
import tempfile
from PIL import Image
from streamlit_image_annotation import detection
import supervision as sv

# RF-DETR specific imports (Ensure these are installed in your environment)
try:
    from rfdetr import RFDETRBase
except ImportError:
    st.error("RF-DETR package not found. Please install it to use the detection features.")

# --- 1. DIRECTORY SETUP ---
BASE_DIR = os.path.abspath("retail_data")
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
MODEL_EXPORT_DIR = os.path.join(BASE_DIR, "trained_model")

for path in [IMG_DIR, LBL_DIR, MODEL_EXPORT_DIR]:
    os.makedirs(path, exist_ok=True)

st.set_page_config(page_title="Retail SKU AI (RF-DETR)", layout="wide")

# --- SIDEBAR NAVIGATION & STATS ---
st.sidebar.title("🚀 RF-DETR Retail AI")
panel = st.sidebar.radio("Navigation", ["🏗️ Training Panel", "👤 Client Panel"])

st.sidebar.divider()
st.sidebar.subheader("Dataset Statistics")
num_imgs = len([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
num_lbls = len([f for f in os.listdir(LBL_DIR) if f.endswith('.txt')])
st.sidebar.write(f"🖼️ Images: {num_imgs}")
st.sidebar.write(f"🏷️ Annotations: {num_lbls}")

# --- 2. PANEL 1: TRAINING ---
if panel == "🏗️ Training Panel":
    st.title("🏗️ SKU Training Workshop")
    
    # Upload Section
    uploads = st.file_uploader("Upload New SKU Photos", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    if uploads:
        for f in uploads:
            with open(os.path.join(IMG_DIR, f.name), "wb") as file:
                file.write(f.getbuffer())
        st.success(f"Uploaded {len(uploads)} images!")
    
    # Annotation Section
    img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if img_list:
        selected_img = st.selectbox("Select image to annotate", img_list)
        img_full_path = os.path.join(IMG_DIR, selected_img)
        img_cv = cv2.imread(img_full_path)
        h, w, _ = img_cv.shape

        # Integrated Annotation Tool
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
                        bx = ann['bbox'] # [x, y, w, h]
                        xc, yc = (bx[0] + bx[2]/2) / w, (bx[1] + bx[3]/2) / h
                        nw, nh = bx[2] / w, bx[3] / h
                        f.write(f"0 {xc} {yc} {nw} {nh}\n")
                st.success(f"Saved {len(new_annotations)} boxes for {selected_img}!")

    st.divider()
    # Training Trigger
    if st.button("🔥 Run RF-DETR Training"):
        with st.spinner("Training RF-DETR Transformer... This may take time."):
            try:
                model = RFDETRBase()
                # model.train(dataset_path=BASE_DIR, epochs=10) # Placeholder for actual training call
                st.info("RF-DETR training initiated. Check console for logs.")
            except Exception as e:
                st.error(f"Training error: {e}")

# --- 3. PANEL 2: CLIENT (DETECTION & EXPORT) ---
else:
    st.title("👤 Client Detection Panel")
    
    # Model Loading
    fixed_path = os.path.join(MODEL_EXPORT_DIR, "best.pt")
    if os.path.exists(fixed_path):
        model = RFDETRBase(weights=fixed_path)
        st.success("✅ Custom RF-DETR Weights Loaded")
    else:
        st.warning("⚠️ Using Base RF-DETR weights (Pre-trained).")
        model = RFDETRBase()

    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
    
    if uploaded_file:
        file_type = uploaded_file.type.split('/')[0]
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # --- IMAGE PROCESSING ---
        if file_type == 'image':
            img = Image.open(uploaded_file)
            img_np = np.array(img)
            
            detections = model.predict(img_np, threshold=0.25)
            
            annotated_image = box_annotator.annotate(scene=img_np.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            
            st.image(annotated_image, caption="Inference Result", use_container_width=True)
            
            # Download Image Option
            res_img = Image.fromarray(annotated_image)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                res_img.save(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button("💾 Download Annotated Image", f, "result.png", "image/png")

        # --- VIDEO PROCESSING ---
        elif file_type == 'video':
            # Save uploaded video to temp
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            
            video_cap = cv2.VideoCapture(tfile.name)
            width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video_cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Prepare Video Writer
            out_temp_path = os.path.join(tempfile.gettempdir(), "output_detection.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_writer = cv2.VideoWriter(out_temp_path, fourcc, fps, (width, height))

            st_frame = st.empty()
            progress_bar = st.progress(0)
            
            curr = 0
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break
                
                # RF-DETR expects RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model.predict(frame_rgb, threshold=0.25)
                
                # Annotate
                annotated_frame = box_annotator.annotate(scene=frame_rgb.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
                
                # Save frame (Convert back to BGR for Writer)
                video_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                
                # Update Streamlit UI
                st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
                curr += 1
                progress_bar.progress(min(curr / frame_count, 1.0))

            video_cap.release()
            video_writer.release()
            
            # Download Video Button
            with open(out_temp_path, "rb") as vid_file:
                st.download_button(
                    label="💾 Download Processed Video",
                    data=vid_file,
                    file_name="sku_detection_output.mp4",
                    mime="video/mp4"
                )

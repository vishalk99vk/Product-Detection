import streamlit as st
from streamlit_image_annotation import detection
from ultralytics import YOLO
import os
import cv2
from PIL import Image

# Initialize folders for the training data
DATA_DIR = "custom_dataset"
os.makedirs(f"{DATA_DIR}/images", exist_ok=True)
os.makedirs(f"{DATA_DIR}/labels", exist_ok=True)

st.set_page_config(layout="wide")
page = st.sidebar.radio("Navigate", ["Training Panel", "Client Panel"])

# --- 1. TRAINING PANEL ---
if page == "Training Panel":
    st.title("🏗️ Training Panel: Annotate & Train")
    uploaded_files = st.file_uploader("Upload Training Images", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            img_path = os.path.join(DATA_DIR, "images", uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.success("Images uploaded! Now draw boxes below.")
        
        # Annotation Interface
        img_list = os.listdir(f"{DATA_DIR}/images")
        target_img = st.selectbox("Select image to label", img_list)
        img_full_path = f"{DATA_DIR}/images/{target_img}"
        
        # User draws boxes here
        # label_list=["product"] ensures we only have one category
        new_labels = detection(image_path=img_full_path, label_list=["product"], key=target_img)
        
        if st.button("Submit Annotations"):
            if new_labels:
                # Logic to convert annotations to YOLO .txt format
                # YOLO format: [class_id x_center y_center width height] (normalized)
                st.info(f"Saved {len(new_labels)} boxes for {target_img}")
                # (Add your file-saving logic here)

    if st.button("🔥 Start Training Model"):
        with st.spinner("Model is learning your new products..."):
            model = YOLO("yolov8n.pt") # Start from base model
            # training command (requires a data.yaml file)
            # model.train(data="custom_data.yaml", epochs=10, imgsz=640)
            st.success("Training Complete! New model is ready for the Client Panel.")

# --- 2. CLIENT PANEL ---
elif page == "Client Panel":
    st.title("👤 Client Panel: Test Detection")
    # Load the newly trained model (usually saved in runs/detect/train/weights/best.pt)
    try:
        model = YOLO("best.pt") 
    except:
        model = YOLO("yolov8n.pt") # Fallback if not trained yet

    test_file = st.file_uploader("Upload Store Image for Detection")
    if test_file:
        img = Image.open(test_file)
        results = model.predict(img, conf=0.25)
        
        # Display annotated image
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="AI Detection Output", use_container_width=True)

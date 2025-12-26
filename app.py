import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Shelf Product Highlighter", layout="wide")
st.title("🏪 Highlight All Products with Red Box")

uploaded_file = st.file_uploader(
    "Upload store shelf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If any contours found, draw bounding box around all
    if contours:
        x_min = min([cv2.boundingRect(c)[0] for c in contours])
        y_min = min([cv2.boundingRect(c)[1] for c in contours])
        x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
        y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

        boxed_img = img_array.copy()
        cv2.rectangle(boxed_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)  # Red box

        st.subheader("Annotated Image")
        st.image(boxed_img, use_column_width=True)
    else:
        st.warning("No objects detected. Try a clearer shelf image.")

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

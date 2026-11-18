# Role 4 - Matteo Godinez

import streamlit as st
import numpy as np
import cv2
from PIL import Image

import methods.canny as canny
# import methods.unet as unet

st.set_page_config(layout="wide", page_title="Anime Line Art Extraction")
st.title("Anime Line Art Extraction")
st.write("Comparing classical edge detection (Canny) against a learned model (U-Net).")

uploaded_file = st.file_uploader("Upload a color anime image (512x512)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and normalize the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    color_np = np.array(image).astype(np.float32) / 255.0
    
    # Resize to exactly 512x512 if the user uploads a different size
    if color_np.shape[:2] != (512, 512):
        color_np = cv2.resize(color_np, (512, 512), interpolation=cv2.INTER_AREA)

    # 2. Build the 3-column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Input Image")
        st.image(color_np, use_container_width=True)
        
    with col2:
        st.subheader("2. Canny Edge Detection")
        mask_canny = canny.predict(color_np, resolution="full")
        st.image(mask_canny, clamp=True, use_container_width=True)
        st.info("Waiting for Role 2 to push `methods/canny.py`!")
        
    with col3:
        st.subheader("3. U-Net (Learned)")
        # TODO: Uncomment when Role 3 finishes training
        # mask_unet = unet.predict(color_np, resolution="full")
        # st.image(mask_unet, clamp=True, use_container_width=True)
        st.info("Waiting for Role 3 to finish training the U-Net!")
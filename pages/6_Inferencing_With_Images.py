import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch

st.set_page_config(page_title='cdio-sp', page_icon='./media/E-logo.png', layout='centered')
############################################################################################################################
comparison2 = st.sidebar.button("Compare weights")
pt_weight = st.sidebar.checkbox("pt weights", value=False)
ir_weight = st.sidebar.checkbox("IR weights", value=False)
filepath_img = st.sidebar.text_input("Insert test dataset", "./path/to/test/dataset")
images = st.sidebar.slider("Select Image", max_value=40, min_value=1, step=1)
############################################################################################################################
chosen = str(filepath_img) + 'Frame_' +str(images) + '.jpg'
if comparison2:
    if pt_weight:
        if chosen is not None:
            image_pt = Image.open(chosen)
            @st.cache
            def get_model():
                return torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/yolov5n.pt', force_reload=True)
            model=get_model()
            results = model(image_pt)
            rendered = np.squeeze(results.render())
            st.image(rendered)
            st.success(':white_check_mark: Inferenced (pt weights)')
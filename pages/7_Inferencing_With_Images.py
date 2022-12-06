import streamlit as st
from PIL import Image
from time import time

import numpy as np
import cv2
import torch

st.set_page_config(page_title='Inferencing with Images', page_icon='./media/vilota.jpg', layout='wide')
############################################################################################################################
comparison2 = st.sidebar.button("Compare weights")
pt_weight = st.sidebar.checkbox("pt weights", value=False)
ir_weight = st.sidebar.checkbox("IR weights", value=False)
filepath_img = st.sidebar.text_input("Insert test dataset", "./path/to/test/dataset")
images = st.sidebar.slider("Select Image", max_value=40, min_value=1, step=1)
############################################################################################################################
chosen = str(filepath_img) + 'Frame_' +str(images) + '.jpg'
if comparison2:
    kpi_1_img, kpi_2_img = st.columns(2)
    with kpi_1_img:
        if pt_weight:
            if chosen is not None:
                image_pt = Image.open(chosen)
                @st.cache
                def get_model():
                    return torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/v3/exp19_rect_clean/best.pt', force_reload=True)
                model=get_model()
                results = model(image_pt)
                rendered = np.squeeze(results.render())
                st.image(rendered)
                st.success(':white_check_mark: Inferenced (pt weights)')
    with kpi_2_img:
        if ir_weight:
            if chosen is not None:
                image_ir = Image.open(chosen)
                st.image(image_ir)
                st.success(':white_check_mark: Inferenced (IR weights)')
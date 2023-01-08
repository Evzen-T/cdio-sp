import streamlit as st
from PIL import Image
import numpy as np
import torch
from multi_infer import multi_infer
import subprocess

st.set_page_config(page_title='cdio-sp', page_icon='./media/E-logo.png', layout='centered')
############################################################################################################################
infer_method = st.sidebar.selectbox("Select method of inference", ["Single", "Multi"])
if infer_method=="Single":
    Inference = st.sidebar.button("Inference images")
    filepath_img = st.sidebar.text_input("Insert test dataset", "./test/dataset")
    images = st.sidebar.slider("Select Image", max_value=40, min_value=1, step=1)
if infer_method=="Multi":
    Inference2=st.sidebar.button("Inference images")
    filepath_img_2 = st.sidebar.text_input("Insert test dataset", "./dataset/test_dt")
    download_path = st.sidebar.text_input("Insert download path", "./output/inferenced")
    try:
        subprocess.run(['mkdir', download_path])
    except:
        st.write = download_path + " " + "exists"
############################################################################################################################

if infer_method=="Single":
    chosen = str(filepath_img) + 'Frame_' +str(images) + '.jpg'
    if Inference:
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

if infer_method=="Multi":
    if Inference2:
        kpi1, kpi2 = st.columns(2)
        @st.cache
        def get_model():
            return torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/yolov5n.pt', force_reload=True)
        model2=get_model()
        multi_infer(fpath_img=filepath_img_2, dlpath=download_path, model=model2, kpi=kpi1, kpi_num=1, num=1)
        multi_infer(fpath_img=filepath_img_2, dlpath=download_path, model=model2, kpi=kpi2, kpi_num=2, num=3)
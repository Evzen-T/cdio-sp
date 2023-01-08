import streamlit as st
import sys, os
sys.path.append(os.getcwd())
from yolo.export_yolov5 import YoloV5Exporter
from pathlib import Path

st.set_page_config(page_title='cdio-sp', page_icon='./media/E-logo.png', layout='centered')

conversion = './media/pt_2_blob.png'
st.image(conversion, width=640)
def fpchange():
    return
def whchange():
    return
col1, col2 = st.columns(2)
with col1:
    dpath = st.text_input('Insert download path', '/home/user/path')
    fpath = st.text_input('Insert file name', 'best.pt', on_change=fpchange)
with col2:
    wshape = st.number_input('Insert width', min_value=0, max_value=1280, value=640, step=32)
    hshape = st.number_input('Insert height', min_value=0, max_value=800, value=416, step=32)
if fpchange:
    convert = st.button('Convert')
    downloadpath = Path(dpath)
    filepath = fpath
    input_shape = [wshape, hshape]
    conv_id = 1
    if convert:
        exporter = YoloV5Exporter(downloadpath, filepath, input_shape, conv_id)
        try:
            exporter.export_onnx()
        except Exception as e:
            raise RuntimeError("onnx conversion failed")
        version = "v5"
        try:
            exporter.export_openvino(version)
        except Exception as e:
            raise RuntimeError("openvino conversion failed")
        try:
            exporter.export_blob()
        except Exception as e:
            raise RuntimeError("blob conversion failed")
        try:
            exporter.export_json()
        except Exception as e:
            raise RuntimeError("json export failed")
        st.success(':white_check_mark: Exported')
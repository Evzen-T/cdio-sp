import streamlit as st
import subprocess, os

st.set_page_config(page_title='cdio-sp', page_icon='./media/E-logo.png', layout='centered')

yv5_path = st.text_input("Insert path to yolov5", '/home/user/yolov5')
weights_path = st.text_input("Insert path to weights", '/home/user/best.pt')
image_width = st.number_input('Insert image width', min_value=0, max_value=1280, value=640, step=32)
image_height = st.number_input('Insert image height', min_value=0, max_value=1280, value=416, step=32)

include_IR = st.checkbox('Export Openvino weights?')
if include_IR:
    i_IR = ' --include openvino '
else:
    i_IR = ' '

go_2_path = 'cd ' + yv5_path
weights = ' --weights ' + str(weights_path)
ir = str(i_IR)
img_list = []
img_size = ' --img ' + str(image_width) + ' ' + str(image_height)

export_code = 'python3 export.py' + ' --weights ' + str(weights_path) + str(i_IR) + str(img_size)

st.code(go_2_path)
st.code(export_code)

weights_head, weights_tail = os.path.split(str(weights_path))

def subproc(clicked):
    st.subheader(clicked)
export = st.button('Export', on_click=subproc, args=("**Exporting...**",))

if export:
    subprocess.run(["sh", "export.sh", str(yv5_path), str(weights), ir, str(img_size)])
    st.info(":information_source: Results saved to " + weights_head)
    st.success(':white_check_mark: Exported IR')
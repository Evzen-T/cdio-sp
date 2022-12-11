import streamlit as st
import numpy as np
import psutil
import subprocess
import os, signal
from time import time

import cv2
import depthai as dai

st.set_page_config(page_title='Take picture', page_icon='./media/E-logo.png', layout='centered')
st.title("Taking pictures with oak camera")

#########################################################################################################################
# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
ccenter = pipeline.create(dai.node.ColorCamera)
xoutcenter = pipeline.create(dai.node.XLinkOut)
xoutcenter.setStreamName('center')

# Properties
ccenter.setPreviewSize(300, 300)
ccenter.setBoardSocket(dai.CameraBoardSocket.RGB)
ccenter.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
ccenter.setInterleaved(False)
ccenter.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
ccenter.video.link(xoutcenter.input)
#########################################################################################################################

online = st.sidebar.checkbox("Initialise camera")

if 'count' not in st.session_state:
    st.session_state.count = 0
def increment_counter():
    st.session_state.count += 1

previous = 0
stframe = st.empty()

select_folder = st.sidebar.checkbox("Select folder to use")

if online and select_folder:

    select_state_cam = st.sidebar.selectbox("Select state of cam",
                                    ['static', 'dynamic'])
    num_folder = st.sidebar.slider("Select folder number", max_value=10, min_value=1, step=1)
    folder_path = "./output"
    folder_name = folder_path + "/" + str(select_state_cam) + "_" + str(num_folder)
    try:
        subprocess.run(['mkdir', folder_name])
    except:
        st.write = folder_name + " " + "exists"

    taking_pics = st.button('Take 1 pic', on_click=increment_counter)
    print('Webcam Online ...')
    with dai.Device(pipeline, usb2Mode=True) as device:
        img_counter = 0
        # Output queues will be used to get the grayscale frames from the outputs defined above
        qcenter = device.getOutputQueue(name="center", maxSize=4, blocking=False)
        while True:
            # Cam feed
            c_cam = qcenter.get().getCvFrame()
            c_cam_rgb = cv2.cvtColor(c_cam, cv2.COLOR_BGR2RGB)
            # rescaled_c_cam = rescale_frame(c_cam, width_res=640, height_res=400)
            stframe.image(c_cam_rgb, channels = 'RGB', use_column_width=True)

            if taking_pics and st.session_state.count > previous:
                img_name = folder_name + "/Frame_{}.jpg".format(st.session_state.count)
                cv2.imwrite(img_name, c_cam)
                print("{} written!".format(img_name))
                st.success(':white_check_mark: {} written!'.format(img_name))
                previous = st.session_state.count
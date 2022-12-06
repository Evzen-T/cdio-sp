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
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

# Linking
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)
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
        qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        while True:
            # Webcam feed
            #########################################################################################################################
            # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
            inLeft = qLeft.get().getCvFrame()
            inLeft_Colour = cv2.cvtColor(inLeft, cv2.COLOR_GRAY2RGB)
            # inRight = qRight.get().getCvFrame()
            # inRight_Colour = cv2.cvtColor(inRight, cv2.COLOR_GRAY2RGB)
            #########################################################################################################################
            stframe.image(inLeft_Colour,channels = 'RGB', use_column_width=True)

            if taking_pics and st.session_state.count > previous:
                img_name = folder_name + "/Frame_{}.jpg".format(st.session_state.count)
                cv2.imwrite(img_name, inLeft_Colour)
                print("{} written!".format(img_name))
                st.success(':white_check_mark: {} written!'.format(img_name))
                previous = st.session_state.count
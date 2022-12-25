import streamlit as st
import time
import torch
import numpy as np
import cv2
import depthai as dai

# Streamlit config
st.set_page_config(page_title='cdio-sp', page_icon='./media/E-logo.png', layout='centered')

nnPath = './weights/yolov5n.blob'
labelMap = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
            "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
            "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2022_1)

# Define sources and outputs
ccenter = pipeline.create(dai.node.ColorCamera)
xoutcenter = pipeline.create(dai.node.XLinkOut)
xoutcenter.setStreamName('center')

# Properties
# ccenter.setPreviewSize(640, 416)
ccenter.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
ccenter.setInterleaved(False)
ccenter.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
ccenter.setFps(30)

# Linking
ccenter.video.link(xoutcenter.input)

st.set_option('deprecation.showfileUploaderEncoding', False)
####################################################
Inference = st.sidebar.button("Inference")
pt_weights = st.sidebar.checkbox("Pytorch weights")
# ir_weights = st.sidebar.checkbox("Openvino weights")
blob_weights = st.sidebar.checkbox("Blob weights")
####################################################
st.markdown(' ## Output')
stframe1     = st.empty()
stframe2    = st.empty()
prevTime    = 0
if Inference:
    st.markdown('---')
    kpi1, kpi2 = st.columns(2)
    with kpi1:
        st.markdown("**FPS**")
        kpi1_text = st.markdown("0")
    with kpi2:
        st.markdown("**Image Width**")
        kpi2_text = st.markdown("0")

    with dai.Device(pipeline, usb2Mode=True) as device:
        img_counter = 0
        # Output queues will be used to get the rgb frames from the outputs defined above
        qcenter = device.getOutputQueue(name="center", maxSize=4, blocking=False)

        start_time1 = time.time()
        counter1 = 0
        fps1 = 0

        start_time2 = time.time()
        counter2 = 0
        fps2 = 0
        while True:
            # Cam feed
            c_cam = qcenter.get().getCvFrame()
            c_cam_rgb = cv2.cvtColor(c_cam, cv2.COLOR_BGR2RGB)

            if pt_weights:
                @st.cache
                def get_model():
                    return torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/yolov5n.pt', force_reload=True)

                model = get_model()

                # Make decisions
                results = model(c_cam_rgb.copy())
                squeezed = np.squeeze(results.render())
                stframe1.image(squeezed,channels = 'RGB', use_column_width=True)

                #fps counter 1
                counter1 += 1
                if (time.time() - start_time1) > 1:
                    fps1 = counter1 / (time.time() - start_time1)
                    counter1 = 0
                    start_time1 = time.time()
                
                #kpi
                kpi1_text.write(f"<h1 style='text-align: left; color: red;'>{int(fps1)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: left; color: red;'>{1080}</h1>", unsafe_allow_html=True)
            # if blob_weights:

                # #fps counter 2
                # counter2 += 1
                # if (time.time() - start_time2) > 1:
                #     fps2 = counter2 / (time.time() - start_time2)
                #     counter2 = 0
                #     start_time2 = time.time()

            
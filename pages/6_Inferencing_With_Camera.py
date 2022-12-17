import streamlit as st
from time import time
import logging as log
import psutil
from PIL import Image

from openvino.inference_engine import IECore
import torch
import numpy as np
import cv2
import depthai as dai

st.set_page_config(page_title='cdio-sp', page_icon='./media/E-logo.png', layout='centered')

#IR
def letterbox(img, size=(1280, 800)):
    w, h = size
    prepimg     = img[:, :, ::-1].copy()
    prepimgr    = cv2.resize(prepimg, (w, h))
    meta        = {'original_shape': prepimg.shape,
                'resized_shape': prepimgr.shape}
    prepimg     = Image.fromarray(prepimgr)
    prepimg     = prepimg.resize((w, h), Image.ANTIALIAS) #or Image.Resampling.LANCZOS in the newer version
    img     = np.asarray(prepimgr)
    return img/255

def parse_yolo_region(input_image, outputs):

    class_ids = []
    class_labels = [
                    'drones ahead',
                    'land slowly',
                    'rotate 360deg right',
                    'stop for 5s']
    confidences = []
    boxes = []
    outputs = [outputs]

    # Rows.
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]

    # Resizing factor.
    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if (confidence >= 0.45):
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)

            if (classes_scores[class_id] > 0.5):
                confidences.append(confidence.item())
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)                
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.45)
    box = None
    for i in indices:
        i = int(i)
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        log.info("\nDetected boxes for batch {}:".format(1))
        log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ")
        cv2.rectangle(input_image, (left, top), (left + width, top + height), (178, 0, 255), 2)
        # print(i)
        label = "{}:{:.2f}".format(class_labels[class_ids[i]], confidences[i])
        log.info("{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4}".format(class_ids[i], confidences[i], left, top, width, height))        
    
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        dim, baseline = text_size[0], text_size[1]
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), (178, 0, 255), cv2.FILLED)
        cv2.putText(input_image, label, (left, top + dim[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    boxes = np.array(boxes)
    ibox = list(boxes[indices])
    return input_image, ibox 

#############################################################################
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
#############################################################################
st.set_option('deprecation.showfileUploaderEncoding', False)
####################################################
Inference = st.sidebar.button("Inference")
pt_weights = st.sidebar.checkbox("Pytorch weights")
ir_weights = st.sidebar.checkbox("Openvino weights")
blob_weights = st.sidebar.checkbox("Blob weights")
####################################################
#To rescale the frame of the video capture
def rescale_frame(frame, width_res=640, height_res=400):
    width = int(frame.shape[1] * width_res/1280)
    height = int(frame.shape[0] * height_res/ 800)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

st.markdown('---')
st.markdown(' ## Output')
stframe     = st.empty()
stframe2    = st.empty()
stframe3    = st.empty()
fps         = 0
prevTime    = 0

if Inference:
    st.sidebar.markdown('---')
    st.sidebar.markdown("**Framerates**")
    kpi1_text = st.sidebar.markdown("0")
    st.sidebar.markdown("**Image Width**")
    kpi2_text = st.sidebar.markdown("0")

    with dai.Device(pipeline, usb2Mode=True) as device:
        img_counter = 0
        # Output queues will be used to get the grayscale frames from the outputs defined above
        qcenter = device.getOutputQueue(name="center", maxSize=4, blocking=False)
        while True:
            # Cam feed
            c_cam = qcenter.get().getCvFrame()
            c_cam_rgb = cv2.cvtColor(c_cam, cv2.COLOR_BGR2RGB)
            rescaled_c_cam = rescale_frame(c_cam_rgb, width_res=640, height_res=400)

            if ir_weights:
                #VINO config
                ie                  = IECore()
                net                 = ie.read_network(model='./weights/v3/exp47_Dv2_l1-4/ir/best.xml')
                input_blob          = next(iter(net.input_info))
                dims                = net.input_info[input_blob].input_data.shape
                device              = "CPU"
                exec_net            = ie.load_network(network=net, num_requests=2, device_name=device)
                # classesFile         = './weights/v3/exp2_yv5s_32/classes.txt'
                n, c, h, w          = dims
                net.batch_size      = n
                is_async_mode       = True
                fnum = 0

                # with open(classesFile, 'rt') as f:
                #     classes = f.read().rstrip('\n').split('\n')

                fnum += 1
                in_frame        = letterbox(rescaled_c_cam.copy(), (w, h))
                in_frame        = in_frame.transpose((2, 0, 1))  
                in_frame        = in_frame.reshape((n, c, h, w))
                start_time      = time()

                if is_async_mode == True:
                    request_id = 1
                    exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame}) 
                else:
                    request_id = 0
                    exec_net.infer(inputs={input_blob: in_frame})

            if pt_weights:
                @st.cache
                def get_model():
                    return torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/v3/exp47_Dv2_l1-4/best.pt', force_reload=True)

                model = get_model()

                # Make decisions
                results = model(rescaled_c_cam.copy())
                squeezed = np.squeeze(results.render())
                
                stframe.image(squeezed,channels = 'RGB', use_column_width=True)

            if ir_weights:
                if exec_net.requests[request_id].wait(-1) == 0: 
                    output  = exec_net.requests[request_id].output_blobs 
                    for layer_name, out_blob in output.items():
                        imcv1, ibox = parse_yolo_region(rescaled_c_cam.copy(), out_blob.buffer)
                    parsing_time = time() - start_time
                            
                if ibox is not None:
                    label = "Frame = %d, Inference time = %.2f ms" % (fnum, parsing_time)
                    cv2.putText(imcv1, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    label = "Frame = %d, NO DETECTION" % fnum
                    cv2.putText(imcv1, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
                # imcv1_rz = cv2.resize(imcv1, (vk_width, vk_height))
                final = cv2.cvtColor(imcv1, cv2.COLOR_BGR2RGB)
                stframe2.image(final,channels = 'RGB')
            if blob_weights:
                stframe3.image(rescaled_c_cam,channels = 'RGB')

            # kpi1_text.write(f"<h1 style='text-align: left; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            # kpi2_text.write(f"<h1 style='text-align: left; color: red;'>{vk_width}</h1>", unsafe_allow_html=True)
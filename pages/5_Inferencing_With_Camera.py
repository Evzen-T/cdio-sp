import streamlit as st
import time
import torch
import numpy as np
import cv2

import depthai as dai
from pathlib import Path
import json
import argparse

# Streamlit config
st.set_page_config(page_title='cdio-sp', page_icon='./media/E-logo.png', layout='centered')

syncNN = True
blob_path = './weights/640x416/yolov5n.blob'
json_path = './weights/640x416/yolov5n.json'
####################################################
weighted_preferance = st.sidebar.selectbox("Select weights", ["not initialised", "pt weights", "blob weights"])
Inference = st.sidebar.button("Inference")
####################################################
if weighted_preferance=="pt weights":
    # Create pipeline
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2022_1)

    # Define sources and outputs
    ccenter = pipeline.create(dai.node.ColorCamera)
    xoutcenter = pipeline.create(dai.node.XLinkOut)
    xoutcenter.setStreamName('center')

    # Properties
    ccenter.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    ccenter.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    ccenter.setInterleaved(False)
    ccenter.setFps(30)

    # Linking
    ccenter.video.link(xoutcenter.input)

elif weighted_preferance=="blob weights":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                        default=blob_path, type=str)
    parser.add_argument("-c", "--config", help="Provide config path for inference",
                        default=json_path, type=str)
    args = parser.parse_args()

    # parse config
    configPath = Path(args.config)
    if not configPath.exists():
        raise ValueError("Path {} does not exist!".format(configPath))

    with configPath.open() as f:
        config = json.load(f)
    nnConfig = config.get("nn_config", {})

    # parse input shape
    if "input_size" in nnConfig:
        W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

    # extract metadata
    metadata = nnConfig.get("NN_specific_metadata", {})
    classes = metadata.get("classes", {})
    coordinates = metadata.get("coordinates", {})
    anchors = metadata.get("anchors", {})
    anchorMasks = metadata.get("anchor_masks", {})
    iouThreshold = metadata.get("iou_threshold", {})
    confidenceThreshold = metadata.get("confidence_threshold", {})

    # parse labels
    nnMappings = config.get("mappings", {})
    labels = nnMappings.get("labels", {})

    # Create pipeline
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2022_1)

    # Define sources and outputs
    ccenter = pipeline.create(dai.node.ColorCamera)
    xoutcenter_2 = pipeline.create(dai.node.XLinkOut)
    xoutcenter_2.setStreamName("rgb")

    yolo_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    nnOut = pipeline.create(dai.node.XLinkOut)
    nnOut.setStreamName("nn")

    # Properties
    ccenter.setPreviewSize(W, H)
    ccenter.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    ccenter.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    ccenter.setPreviewKeepAspectRatio(False)
    ccenter.setInterleaved(False)
    ccenter.setFps(30)

    # Network specific settings
    yolo_nn.setConfidenceThreshold(confidenceThreshold)
    yolo_nn.setNumClasses(classes)
    yolo_nn.setCoordinateSize(coordinates)
    yolo_nn.setAnchors(anchors)
    yolo_nn.setAnchorMasks(anchorMasks)
    yolo_nn.setIouThreshold(iouThreshold)
    yolo_nn.setBlobPath(blob_path)
    yolo_nn.setNumInferenceThreads(2)
    yolo_nn.input.setBlocking(False)

    # Linking
    ccenter.preview.link(yolo_nn.input)
    if syncNN:
        yolo_nn.passthrough.link(xoutcenter_2.input)
    else:
        ccenter.preview.link(xoutcenter_2.input)
    yolo_nn.out.link(nnOut.input)

st.markdown(' ## Output')
stframe1     = st.empty()
stframe2    = st.empty()
prevTime    = 0

if weighted_preferance=="pt weights":
    if Inference:
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.markdown("**FPS**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Image Width**")
            kpi2_text = st.markdown("0")
        
        # Counter params
        start_time1 = time.time()
        counter1 = 0
        fps1 = 0

        with dai.Device(pipeline, usb2Mode=True) as device:
            img_counter = 0

            # Depthai camera Output
            qcenter = device.getOutputQueue(name="center", maxSize=4, blocking=False)
            print("Oak camera online...")
            while True:
                # Cam feed
                c_cam = qcenter.get().getCvFrame()
                c_cam_rgb = cv2.cvtColor(c_cam, cv2.COLOR_BGR2RGB)

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

if weighted_preferance=="blob weights":
    if Inference:
        with dai.Device(pipeline, usb2Mode=True) as device:
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            qcenter = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qyolo_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            startTime = time.monotonic()
            counter = 0
            color2 = (255, 255, 255)

            # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
            def frameNorm(frame, bbox):
                normVals = np.full(len(bbox), frame.shape[0])
                normVals[::2] = frame.shape[1]
                return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

            def displayFrame(name, frame, detections):
                color = (255, 0, 0)
                for detection in detections:
                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Show the frame
                stframe2.image(frame, channels="RGB", use_column_width=True)
                
            print("Oak camera online...")
            while True:
                inCenter = qcenter.get()
                inNN = qyolo_nn.get()

                if inCenter is not None:
                    frame = inCenter.getCvFrame()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    texted_frame = cv2.putText(frame_rgb, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                                (2, frame_rgb.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

                if inNN is not None:
                    detections = inNN.detections
                    counter += 1

                if frame is not None:
                    displayFrame("rgb", texted_frame, detections)
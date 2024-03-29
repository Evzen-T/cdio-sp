# **SP CDIO Project**

![SP logo](./media/sp-logo.png)

## **What is used in this project**

- Hardware
    - Orangepi 3 lts
    - Oak d lite camera

- Software

    - <img src='./media/streamlit.jpg' width=55>
    - <img src='./media/yolov5.png'  width=50> 

- Programming language
    - Python
    - Jupyter notebook
    - Shell scripting

## **Trained models**
- Sign detection (Vilota trained model)
- everyday objects (yolov5n pretrain model)
- classroom objects (custom made trained model)

## **Object detection workflow**
1. Decide object to detect
2. Dataset Collection
3. Dataset Cleaning
4. Dataset Annotation
5. Training (Using YoloV5 to train)
6. Conversion to other file formats for inferencing
7. Inferencing (Camera/video/image)

## **Dependencies**
```
python3 -m venv <insert virtual name>
source <virtual name>/bin/activate

pip install -r requirements.txt
```

Annotation Tool (Segmentation) - label-studios ([Github repo](https://github.com/heartexlabs/label-studio))
```bash
# To install label-studio
pip3 install label-studio # Requires Python >=3.7 <=3.9

# To run label-studio 
label-studio # Will start the server at http://localhost:8080
```

For Training - YoloV5([Github repo](https://github.com/ultralytics/yolov5))
```bash
git submodule update --init --force --recursive
cd yolo/yolov5
source <virtual name>/bin/activate
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
## **How to run streamlit webui?**

```
streamlit run 1_About_App.py
```

## **Page functions**

1. **About App**
    - Demo inferencing video
    - More details on object detection workflow

2. **Take Picture**
    - Used for Dataset collection

3. **Training**
    - Use for training annotated datasets with yolov5
    - pretrained weights used
        - [yolov5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt)

4. **Blob Converter**
    - Exports **pytorch** weights to **onnx** > **IR** > **blob** 

5. **Inferencing With Camera**
    - Run inference with **oak d lite (Luxonis cameras)**
    - Inference with pytorch weights obtained from **Training**
    - Inference with blob weights obtained from **Blob Converter**

6. **Inferencing With Images**
    - Run image inference with **images**
    - Inference with pytorch weights obtained from **Training**
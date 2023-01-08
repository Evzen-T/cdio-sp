import streamlit as st
from PIL import Image
import cv2
import numpy as np

def multi_infer(fpath_img, dlpath, model, kpi, kpi_num, num):
    
    with kpi:
        st.header(f"Test {kpi_num}")
        chosen_k1 = f'{fpath_img}/Frame_{num}.jpg'
        if chosen_k1 is not None:
            image_pt1 = Image.open(chosen_k1)
            results1 = model(image_pt1)
            rendered1 = np.squeeze(results1.render())
            cv2.imwrite(f"{dlpath}/Frame_{num}.jpg",rendered1)
            st.image(rendered1)
            st.success(f':white_check_mark: Inferenced | {dlpath}/Frame_{num}.jpg')

        chosen_k2 = f'{fpath_img}/Frame_{num+1}.jpg'
        if chosen_k2 is not None:
            image_pt2 = Image.open(chosen_k2)
            results2 = model(image_pt2)
            rendered2 = np.squeeze(results2.render())
            cv2.imwrite(f"{dlpath}/Frame_{num+1}.jpg",rendered2)
            st.image(rendered2)
            st.success(f':white_check_mark: Inferenced | {dlpath}/Frame_{num+1}.jpg')
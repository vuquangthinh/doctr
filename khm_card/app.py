import os.path
import sys

os.environ["USE_TORCH"] = "1"

# # go up one directory level from this file's directory:
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# # prepend parent directory to the system path:
# sys.path.insert(0, path)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from khm_card.utils import load_predictor, extract_image

from khm_card.detector import Cambodia_FieldDetector

st.title('KhmerCard OCR')

cols = st.columns((1, 1, 1, 1))
cols[0].subheader("Input")
cols[2].subheader("OCR output")

det_arch = 'db_resnet50'
reco_arch = 'crnn_vgg16_bn'
predictor = load_predictor(
  det_arch, 
  reco_arch, 
  assume_straight_pages=False,
  straighten_pages=False,
  bin_thresh=0.5,
  box_thresh=0.5,
  device='cpu'
)

uploaded_file = st.sidebar.file_uploader("Upload files", type=["png", "jpeg", "jpg"])

if uploaded_file is not None:
  doc = DocumentFile.from_images(uploaded_file.read())

  page=doc[0]
  cols[0].image(page)
  
  st.spinner("Extract OCR")
  data = extract_image(predictor, doc)
  
  cols[1].json(data, expanded=True)
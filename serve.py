

from typing import Union, Annotated
import numpy as np
import cv2

from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()



from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import torch

# # # Load the OCR model with the specified language
# # model = ocr_predictor(pretrained=True)

# # document = DocumentFile.from_images('/Users/vuquangthinh/Documents/fencilux/eKYC/xyz/vocr/out/3.jpg')

# # # Load the document or image

# # # Perform OCR on the document
# # result = model(document)

# # # Access the extracted text
# # # text = result['text']

# # # Print the extracted text
# # print(result)

# # from doctr.models import recognition_predictor
# # predictor = recognition_predictor('crnn_vgg16_bn')
# # print(predictor.model.cfg['vocab'])


import os
os.environ['USE_TORCH'] = '1'

# import torch
from doctr.models import recognition_predictor, crnn_vgg16_bn
from doctr.datasets import VOCABS


# reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=VOCABS["khm"])
# reco_params = torch.load('/Users/vuquangthinh/Documents/fencilux/eKYC/KHM_TrainOCR/doctr/crnn_vgg16_bn_20240305-023534.pt', map_location="cpu")
# reco_model.load_state_dict(reco_params)

reco_model = crnn_vgg16_bn(vocab=VOCABS["khm"])
reco_params = torch.load('/Users/vuquangthinh/Documents/fencilux/eKYC/KHM_TrainOCR/doctr/crnn_vgg16_bn_20240305-164320.pt', map_location="cpu")
reco_model.load_state_dict(reco_params)

model = recognition_predictor(arch=reco_model)

@app.post("/idr")
async def ocr_idr(
  file: Annotated[UploadFile, File()],
):
  contents = await file.read()
  # nparr = np.fromstring(contents, np.uint8)
  # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  document = DocumentFile.from_images(contents)
  result = model(document)

  print(result, 'res')

  return result[0][0]

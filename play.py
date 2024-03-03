

from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# # Load the OCR model with the specified language
# model = ocr_predictor(pretrained=True)

# document = DocumentFile.from_images('/Users/vuquangthinh/Documents/fencilux/eKYC/xyz/vocr/out/3.jpg')

# # Load the document or image

# # Perform OCR on the document
# result = model(document)

# # Access the extracted text
# # text = result['text']

# # Print the extracted text
# print(result)

# from doctr.models import recognition_predictor
# predictor = recognition_predictor('crnn_vgg16_bn')
# print(predictor.model.cfg['vocab'])


import os
os.environ['USE_TORCH'] = '1'

import torch
from doctr.models import recognition_predictor, crnn_vgg16_bn
from doctr.datasets import VOCABS


reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=VOCABS["khm"])
reco_params = torch.load('/Users/vuquangthinh/Documents/fencilux/eKYC/KHM_TrainOCR/doctr/crnn_vgg16_bn_20240303-221351.pt', map_location="cpu")
reco_model.load_state_dict(reco_params)

model = recognition_predictor(arch=reco_model, pretrained=True)

document = DocumentFile.from_images('a.png')
result = model(document)

print(result)
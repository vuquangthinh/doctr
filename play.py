

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

document = DocumentFile.from_images('/Users/vuquangthinh/Documents/fencilux/eKYC/KHM_TrainOCR/doctr/data/val_data/e.png')
result = model(document)

print(result)

# predictor = ocr_predictor(reco_arch='crnn_vgg16_bn', det_arch="db_resnet50")
# res = predictor(document)

# print(res)

# import torch
# from doctr.models import ocr_predictor, crnn_vgg16_bn
# from doctr.datasets import VOCABS

# reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=VOCABS["khm"])
# reco_params = torch.load('./crnn_vgg16_bn_20240305-023534.pt')
# reco_model.load_state_dict(reco_params)

# predictor = ocr_predictor(det_arch='db_resnet50', reco_arch=reco_model)
# document = DocumentFile.from_images('a.png')

# res = reco_model(document)

# print(res)

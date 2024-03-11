# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# import tensorflow as tf

# pgu_devices = tf.config.experimental.list_physical_devices("GPU")
# if any(gpu_devices):
#     tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.models import kie_predictor, ocr_predictor, crnn_vgg16_bn, recognition_predictor, crnn_mobilenet_v3_small
from doctr.models import parseq

from doctr.datasets import VOCABS
import torch
import os

reco_khm_model = crnn_vgg16_bn(vocab=VOCABS["khm"])
reco_khm_model.load_state_dict(torch.load(os.path.join('/home/thanhpcc/Documents/testx/doctr/crnn_vgg16_bn_20240308-180030.pt'), map_location=torch.device('cuda:0')))

predictor = ocr_predictor(reco_arch=reco_khm_model, pretrained=True)
det_predictor = predictor.det_predictor
reco_predictor = predictor.reco_predictor

khm_predictor = recognition_predictor(
  arch=reco_khm_model,
  pretrained=False,
  pretrained_backbone=False,
)

mrz_model = crnn_vgg16_bn(vocab=VOCABS["mrz_code"])
mrz_model.load_state_dict(torch.load(os.path.join(os.path.join('/home/thanhpcc/Documents/testx/doctr/mrz.pt')), map_location=torch.device('cuda:0')))
mrz_predictor = ocr_predictor(reco_arch=mrz_model, 
                              pretrained=True,
                              straighten_pages=True, 
                              preserve_aspect_ratio=True, 
                              detect_orientation=True)

latin_ocr_predictor = ocr_predictor(pretrained=True)
latin_predictor = latin_ocr_predictor.reco_predictor


name_model = crnn_vgg16_bn(vocab=VOCABS['khm'])
name_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../../khm_name.pt'), map_location=torch.device('cuda:0')))
name_predictor = recognition_predictor(
  arch=name_model,
  pretrained=False,
  pretrained_backbone=False,
)

date_model = crnn_vgg16_bn(vocab=VOCABS['khm'])
date_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../../khm_date.pt'), map_location=torch.device('cuda:0')))
date_predictor = recognition_predictor(
  arch=date_model,
  pretrained=False,
  pretrained_backbone=False,
)


address_model = crnn_vgg16_bn(vocab=VOCABS['khm'])
address_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../../khm_address.pt'), map_location=torch.device('cuda:0')))
address_predictor = recognition_predictor(
  arch=address_model,
  pretrained=False,
  pretrained_backbone=False,
)

# reco_khm_model = crnn_vgg16_bn(vocab=VOCABS["khm"])
# reco_khm_model.load_state_dict(torch.load(os.path.join(os.path.join(os.path.dirname(__file__), 'models/crnn_vgg16_bn_20240307-185622.pt')), map_location=torch.device('cuda:0')))

kie_predictor = kie_predictor(pretrained=True, reco_arch=reco_khm_model)

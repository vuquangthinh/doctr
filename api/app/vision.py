# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.models import kie_predictor, ocr_predictor, crnn_vgg16_bn, recognition_predictor

from doctr.datasets import VOCABS
import torch
import os

reco_khm_model = crnn_vgg16_bn(vocab=VOCABS["khm"])
reco_khm_model.load_state_dict(torch.load(os.path.join(os.path.join(os.path.dirname(__file__), 'models/crnn_vgg16_bn_20240307-111230.pt')), map_location="cpu"))

predictor = ocr_predictor(reco_arch=reco_khm_model, pretrained=True)
det_predictor = predictor.det_predictor
reco_predictor = predictor.reco_predictor


khm_predictor = recognition_predictor(
  arch=reco_khm_model,
  pretrained=False,
  pretrained_backbone=False,
)

kie_predictor = kie_predictor(pretrained=True, reco_arch=reco_khm_model)

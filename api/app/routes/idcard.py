# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Dict, List

from fastapi import APIRouter, File, UploadFile, status

from io import BytesIO
import numpy as np
from PIL import Image
from app.schemas import OCROut
from doctr.io import decode_img_as_tensor
from doctr.models import kie_predictor, ocr_predictor, crnn_vgg16_bn, db_resnet50
from doctr.datasets import VOCABS

import cv2
from app.detector.Cambodia_FieldDetector import Cambodia_FieldDetector as FieldDetector
from app.detector.CardExtractor import CardExtractor
import time
import os

router = APIRouter()

# get current file directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_text(image):
  # extract card
  card = CardExtractor(os.path.join(BASE_DIR, 'models/card-detector.pt'))

  cardResult = card.predict(image)

  if cardResult is None:
    return None
  
  image = cardResult['image']

  # predict
  field = FieldDetector(os.path.join(BASE_DIR, 'models/field-detector.pt'))

  c1 = time.time()

  result = field.predict(image)

  c2 = time.time()

  return {
    "result": result,
    "time": c2 - c1
  }


@router.post("/", status_code=status.HTTP_200_OK, summary="Perform IDCard")
async def perform_card(file: UploadFile = File(...)):
    # img = decode_img_as_tensor(file.file.read())
    img = np.array(Image.open(BytesIO(file.file.read()), mode="r").convert("RGB"), np.uint8, copy=True)
    out = extract_text(img)
    
    return out

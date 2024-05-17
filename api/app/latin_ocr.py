from paddleocr import PaddleOCR
import numpy as np
import cv2
import uuid
np.int = np.int32
np.float = np.float64
np.bool = np.bool_



ocr = PaddleOCR(cls=True, lang='en', use_gpu=True, use_angle_cls=True)

def image2text(image):
  
  result = ocr.ocr(image)
  
  # cv2.imwrite('./latin' + str(uuid.uuid4()) + '.jpg', image)

  if len(result):
    return result[0][1][0]
    # return result[0][0][1][0]
  return ""
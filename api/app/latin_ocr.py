from paddleocr import PaddleOCR
import numpy as np

np.int = np.int32
np.float = np.float64
np.bool = np.bool_



ocr = PaddleOCR(cls=True, lang='en', use_gpu=True, use_angle_cls=True)

def image2text(image):
  result = ocr.ocr(image)  
  
  if len(result):
    return result[0][1][0]
  return ""
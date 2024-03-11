
from ultralytics import YOLO
import cv2
import numpy as np

class FakeDetector:
  def __init__(self, model) -> None:
    self.model = YOLO(model)

  # extract card in image
  def predict(self, image):
    results = self.model(image)
    
    print(results[0].probs.data[1])
    return results[0].probs.data[1].item()
  

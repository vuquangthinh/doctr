
import os
from os import path
from app.vision import name_predictor

images = os.listdir('./name')

import cv2
# img = cv2.imread('/home/thanhpcc/Documents/testx/doctr/api/name/0b4c0b7c-e795-42d3-8b40-0e63f72803d6.jpg')


labels = {}
for image in images:
  img = cv2.imread('./name/' + image)
  out = name_predictor([img])
  
  if out[0][1] > 0.5:
    labels[image] = out[0][0]
  else:
    labels[image] = ''
  
import json
with open('./labels.json', 'w') as file:
  json.dump(labels, file, indent=4, ensure_ascii=False)
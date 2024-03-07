


import numpy as np
import cv2
import pytesseract
from .FieldDetector import FieldDetector
from datetime import datetime

# import torch
from doctr.models import recognition_predictor, crnn_vgg16_bn
from doctr.datasets import VOCABS
from doctr.io import DocumentFile
import torch

def find_largest_image(boxes):
    largest_image = None
    max_area = 0

    for box in boxes:
        x1, y1, x2, y2 = box

        height = y2 - y1
        width = x2 - x1
        area = height * width

        if area > max_area:
            max_area = area
            largest_image = box

    return largest_image

reco_model = crnn_vgg16_bn(vocab=VOCABS["khm"])
reco_params = torch.load('/Users/vuquangthinh/Documents/fencilux/eKYC/KHM_TrainOCR/doctr-teamsoft/crnn_vgg16_bn_20240307-011748.pt')
reco_model.load_state_dict(reco_params)
model = recognition_predictor(arch=reco_model)

def call_ocr_service(image):
  
#document = DocumentFile.from_images(cv2.imencode(".jpg", image)[1])
  result = model([image])
  
  print(result)
  return result[0][0]

class Cambodia_FieldDetector(FieldDetector):
  def __init__(self, model, debug = False) -> None:
    super().__init__(model, debug)

  def boxImagesToText(self, image, boxImages, normalization = True):

    result = {
      "id": "id" in boxImages and self.extractId(image, boxImages["id"]),
      "name": "name" in boxImages and self.extractName(image, boxImages["name"]),
      "name_en": "name_en" in boxImages and self.extractLatin(image, boxImages["name_en"]),
      "birthday": "birthday" in boxImages and self.extractDate(image, boxImages["birthday"]),
      "sex": "sex" in boxImages and self.extractSex(image, boxImages["sex"]),
      "home": "home" in boxImages and self.extractText(image, boxImages["home"]),
      "address": "address" in boxImages and self.extractText(image, boxImages["address"]),
      "mrz": "mrz" in boxImages and self.extractMRZ(image, boxImages["mrz"]),

      "expired_date": "expired_date" in boxImages and self.extractDate(image, boxImages["expired_date"]),
      "start_date": "start_date" in boxImages and self.extractDate(image, boxImages["start_date"]),
      # "avatar": "avatar" in boxImages and [base64.b64encode(avatar) for avatar in boxImages["avatar"]],
    }

    # normalize MRZ to information
    if normalization:
      mrz = "mrz" in boxImages and self.extractMRZ(image, boxImages["mrz"])
      if len(mrz):
        data = self.mrzExtract(mrz[0])

        # prefer MRZ
        result["id"] = data["id"] if "id" in data else result["id"]
        result["name_en"] = data["name"] if "name" in data else result["name_en"]
        result["birthday"] = data["birthday"] if "birthday" in data else result["birthday"]
        result["sex"] = data["sex"] if "sex" in data else result["sex"]
        result["expired_date"] = data["expired_date"] if "expired_date" in data else result["expired_date"]

    
    return result
  
  def extractId(self, image, metadata):
    # find longest ...
    largest_box = find_largest_image(metadata)

    x1, y1, x2, y2 = largest_box
    fragment = image[y1:y2, x1:x2]
    
    fragment = self.preprocessing_image(fragment)
    # ID is numeric
    result = pytesseract.image_to_string(fragment, 
      config='--psm 7').split("\n")
    result = [item for item in result if item.strip()]

    return " ".join(result)
  
  def extractName(self, image, metadata):
    boxes = self.crop_and_recog(image, metadata)

    output = []
    for box in boxes:
      img = cv2.resize(box, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
      fragment = img

      # result = pytesseract.image_to_string(fragment, 
        # config='--tessdata-dir /opt/homebrew/share/tessdata --psm 13 -l khm').split("\n")
      result = call_ocr_service(fragment)
      output.append(result)

    return " ".join(output)
  
  def preprocessing_image2(self, image):
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    img = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)


    image2 = cv2.erode(img, None, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (1, 1))
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_OTSU)  # threshold
    dilated = cv2.dilate(thresh, kernel, iterations=1)  # dilate

    return dilated
  
  def extractLatin(self, image, metadata):
    boxes = self.crop_and_recog(image, metadata)

    output = []
    for box in boxes:

      fragment = self.preprocessing_image(box)
      result = pytesseract.image_to_string(fragment, 
        config='--psm 13 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').split("\n")
      
      result = [item for item in result if item.strip()]
      result = ' '.join([item.strip() for item in result])
      
      output.append(result)

    return " ".join(output)
  
  def extractMRZ(self, image, metadata):
    boxes = self.crop_and_recog(image, metadata)

    output = []
    for box in boxes:
      result = pytesseract.image_to_string(box, 
        config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ<0123456789').split("\n")  # tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ<0123456789').split("\n")
      
      result = [item.replace(" ", "") for item in result if item.strip()]
      output.append(result)

    return output
  
  def _khmerToLatin(self, text):
    khmer_to_latin = {
      '០': '0',
      '១': '1',
      '២': '2',
      '៣': '3',
      '៤': '4',
      '៥': '5',
      '៦': '6',
      '៧': '7',
      '៨': '8',
      '៩': '9',
      '.': '.'
    }

    latin_number = ''.join(khmer_to_latin.get(char, char) for char in text)

    return latin_number
  
  def extractDate(self, image, metadata):
    # find longest ...
    largest_box = find_largest_image(metadata)

    x1, y1, x2, y2 = largest_box
    fragment = image[y1:y2, x1:x2]
    
    fragment = self.preprocessing_image(fragment)
    # ID is numeric
    result = pytesseract.image_to_string(fragment, lang='khm', config='--psm 7 -c tessedit_char_whitelist="០១២៣៤៥៦៧៨៩."').split("\n")    
    result = [item for item in result if item.strip()]

    date_str = self._khmerToLatin(" ".join(result))

    try:
      date_str = re.sub('[-.:]', '', date_str)
      date_str = datetime.strptime(date_str, "%d%m%Y")
      date_str = date_str.strftime("%Y-%m-%d")

      return date_str
    except Exception as ex:
      return "invalid - " + " ".join(result)
  
  def extractSex(self, image, metadata):
    # find longest ...
    largest_box = find_largest_image(metadata)

    x1, y1, x2, y2 = largest_box
    fragment = image[y1:y2, x1:x2]
    
    fragment = self.preprocessing_image(fragment)
    # ID is numeric
    result = pytesseract.image_to_string(fragment, lang='khm', config='--psm 13 -c tessedit_char_blacklist=0123456789,;').split("\n")
    result = [item for item in result if item.strip()]

    if "ប្រុស" in result: # male
      return "M"
    
    if "ស្រី" in result: # female
      return "F"
    
    # origin
    return " ".join(result)
  
  def extractText(self, image, metadata):
    boxes = self.crop_and_recog(image, metadata)

    output = []
    for box in boxes:
      fragment = self.preprocessing_image(box)
      result = pytesseract.image_to_string(fragment, 
        config='--psm 13 -l khm -c tessedit_char_blacklist=/.').split("\n")

      result = [item for item in result if item.strip()]
      result = ' '.join([item.strip() for item in result])
      
      output.append(result)

    return " ".join(output)
  
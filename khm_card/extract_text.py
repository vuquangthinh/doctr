from detector.Cambodia_FieldDetector import Cambodia_FieldDetector as FieldDetector
from detector.CardExtractor import CardExtractor
import time
import os

def extract_text(image):
  # extract card
  card = CardExtractor(os.path.join(os.path.dirname(__file__), 'models/card-detector.pt'))

  cardResult = card.predict(image)
  
  print('ok')

  if cardResult is None:
    return None
  
  image = cardResult['image']

  # predict
  field = FieldDetector(os.path.join(os.path.dirname(__file__), 'models/field-detector.pt'), debug=True)

  c1 = time.time()

  result = field.predict(image)

  c2 = time.time()

  return {
    "result": result,
    "time": c2 - c1
  }

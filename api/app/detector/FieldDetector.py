from abc import abstractmethod, ABC
from ultralytics import YOLO
import numpy as np
import cv2
from mrz.checker.td2 import TD2CodeChecker
from datetime import datetime
import re

def kernel_psf(angle, d, size=20):
    kernel = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    size2 = size // 2                                              # Division(floor)
    A[:,2] = (size2, size2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kernel = cv2.warpAffine(kernel, A, (size, size), flags=cv2.INTER_CUBIC)   # image to specific matrix conversion
    return kernel

#wiener filter implementaion
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    copy_img = np.copy(img)
    copy_img = np.fft.fft2(copy_img)            #  2D fast fourier transform 
    kernel = np.fft.fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)     # wiener formula implementation
    copy_img = copy_img * kernel                             # conversion blurred to deblurred
    copy_img = np.abs(np.fft.ifft2(copy_img))   # 2D inverse fourier transform
    return copy_img

def process(ip_image):
    a=2.2                                                          # contrast
    ang=np.deg2rad(90)                                             # angle psf
    d=20                                                         # distance psf
    
    b, g, r = cv2.split(ip_image)

    # normalization of split images 
    img_b = np.float32(b)/255.0
    img_g = np.float32(g)/255.0
    img_r = np.float32(r)/255.0
    #psf calculation 

    psf = kernel_psf(ang, d)
    #wiener for all split images
    filtered_img_b = wiener_filter(img_b, psf, K = 0.0060)          # small value of k that is snr as if 0 filter will be inverse filter 
    filtered_img_g = wiener_filter(img_g, psf, K = 0.0060)
    filtered_img_r = wiener_filter(img_r, psf, K = 0.0060)
    #merge to form colored image
    filtered_img=cv2.merge((filtered_img_b,filtered_img_g,filtered_img_r))
    #converting float to unit 
    filtered_img=np.clip(filtered_img*255,0,255)   # clipping values between 0 and 255
    filtered_img=np.uint8(filtered_img)
    #changing contrast of the image
    filtered_img=cv2.convertScaleAbs(filtered_img,alpha=a)
    #removing gibbs phenomena or rings from the image
    filtered_img = cv2.fastNlMeansDenoisingColored(filtered_img, None, 10, 10, 7, 15) 
    filtered_img = cv2.fastNlMeansDenoisingColored(filtered_img, None, 10, 10, 7, 15) # removing left over rings in post processing again     
   
    # using unblurred image to get angle and id of aruco
    return filtered_img


class FieldDetector(ABC):
  def __init__(self, model, debug = False) -> None:
    self.model = YOLO(model)
    self.nms_threshold = 0.15
    self.debug = debug

  def predict(self, image, normalization = True):

    cv2.imwrite('platef.jpg', image)
    # increase size to better detect
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    results = self.model(image, nms=True)

    # find and crop
    boxImages = {}

    x = 0

    # image2 = image.copy()

    t = 0

    for result in results:
      for box in result.boxes:
        left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
        width = right - left
        height = bottom - top
        center = (left + int((right-left)/2), top + int((bottom-top)/2))
        label = result.names[int(box.cls)]
        confidence = float(box.conf.cpu())

        if label not in boxImages:
          boxImages[label] = []
        
        # normalize
          
        boxImages[label].append([
          max(0, int(left) - int(width*0.02)),
          max(0, int(top) - int(height*0.1)), 
          int(right) + int(width*0.02), 
          int(bottom) + int(height*0.3)
          # max(0, int(left)  , int(top), int(right), int(bottom)
        ])


        if self.debug:
          # cv2.rectangle(image2, (left, top),(right, bottom), (255, 0, 0), 2)
          # cv2.putText(image2, label,(left, bottom+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
          cv2.imwrite("./image-" + label + "--" + str(t) + ".jpg", image[top:bottom, left:right])
        t += 1
        

    # for result in results:
    #   boxes = result.boxes.cpu().numpy()

    #   for xyxy in boxes.xyxy:
    #     x1, y1, x2, y2 = xyxy

    #     # ignore confidence < 0.5

    #     # check per class
    #     for idx, clsId in enumerate(boxes.cls):
    #       print("x", idx, clsId)
    #       classId = result.names[int(clsId)]
    #       print("conf", boxes.conf[idx])
    #       #ignore conf < 0.5
    #       if (boxes.conf[idx] < min_confidence):
    #         continue

    #       if classId not in boxImages:
    #         boxImages[classId] = []

    #       boxImages[classId].append([int(x1),int(y1),int(x2),int(y2)])
        

    for field, value in boxImages.items():
      x = np.array(value)
      boxImages[field] = self.sort_each_category(x)

    return self.boxImagesToText(image, boxImages, normalization)
  
  @abstractmethod
  def boxImagesToText(self, image, boxImages, normalization = True): 
    pass


  def mrzExtract(self, texts):

    if len(texts) != 3:
      return {}

    [line1, line2, line3] = texts

    document_number = line1[5:14]

    birthday = line2[0:6]
    try:
      birthday = line2[0:6]
      date_str = datetime.strptime(birthday, "%y%m%d")
      date_str = date_str.strftime("%Y-%m-%d")
      birthday = date_str
    except Exception as ex:
      birthday = None

    sex = line2[7:8]

    try:
      expiry_date = line2[8:14]
      date_str = datetime.strptime(expiry_date, "%y%m%d")
      date_str = date_str.strftime("%Y-%m-%d")
      expiry_date = date_str
    except Exception as ex:
      expiry_date = None
    
    nationality = line2[15:18]

    surname = str(line3[:line3.find("<<<")]).replace("<", ' ').strip()
    surname = re.sub(r'\s+', ' ', surname)

    return {
       "id": document_number,
       "birthday": birthday,
       "sex": sex,
       "expired_date": expiry_date,
       "nationality": nationality,
       "name": surname
    }
  

  def crop_and_recog(self, image, boxes):
            crop = []
            if len(boxes) == 1:
                xmin, ymin, xmax, ymax = boxes[0]
                crop.append(image[ymin:ymax, xmin:xmax])
            else:
                def sort_boxes(box):
                  xmin, ymin, xmax, ymax = box
                  return ymin, xmin
                
                sorted_boxes = sorted(boxes, key=sort_boxes)
                for box in sorted_boxes:
                    xmin, ymin, xmax, ymax = box
                    crop.append(image[ymin:ymax, xmin:xmax])

                    # print("box", xmin, ymin, xmax, ymax)

            return crop
  
  def preprocessing_image(self, img):
    
    image = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # threshold
    dilated = cv2.dilate(thresh, kernel, iterations=1)  # dilate

    return dilated

    # #convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gray = cv2.multiply(gray, 1.5)
    
    # #blur remove noise
    # blured1 = cv2.medianBlur(gray,3)
    # blured2 = cv2.medianBlur(gray,81)
    # divided = np.ma.divide(blured1, blured2).data
    # normed = np.uint8(255*divided/divided.max())
    
    
    # #Threshold image
    # th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU )

    # return threshed

  def sort_each_category(self, category_text_boxes):
    def get_y1(x):
      return x[0]

    def get_x1(x):
      return x[1]
  
    min_y1 = min(category_text_boxes, key=get_y1)[0]

    mask = np.where(category_text_boxes[:, 0] < min_y1 + 10, True, False)
    line1_text_boxes = category_text_boxes[mask]
    line2_text_boxes = category_text_boxes[np.invert(mask)]

    line1_text_boxes = sorted(line1_text_boxes, key=get_x1)
    line2_text_boxes = sorted(line2_text_boxes, key=get_x1)

    if len(line2_text_boxes) != 0:
      merged_text_boxes = [*line1_text_boxes, *line2_text_boxes]
    else:
      merged_text_boxes = line1_text_boxes

    return merged_text_boxes
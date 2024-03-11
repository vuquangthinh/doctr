
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def crop_and_straighten(image, mask):
    # Find contours in the binary image

    # expand mask
    kernel = np.ones((10, 10), np.uint8)  # You can adjust the size of the kernel as per your requirements
    mask2 = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(contours, key = cv2.contourArea))

    # image2 = image.copy()
    # approximately quadrilateral from contour
    # aux = cv2.approxPolyDP(contours[0], 0.1 * cv2.arcLength(contours[0], True), True)

    # # draw aux as contour
    # cv2.drawContours(image2, [aux], -1, (0, 255, 0), 3)
    # cv2.drawContours(image2, [contours[0]], -1, (0, 0, 255), 3)

    # show_image(image2, cvt=cv2.COLOR_RGB2BGR)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # src_pts = aux.astype(np.float32)

    # print(aux)
    warped = four_point_transform(image, box)

    # # coordinate of the points in box points after the rectangle has been
    # # straightened
    WIDTH_OF_RECTANGLE = 1200
    HEIGHT_OF_RECTANGLE = 800
    # dst_pts = np.array(
    #    [
    #      [0,0],
    #      [WIDTH_OF_RECTANGLE - 1,0],
    #      [WIDTH_OF_RECTANGLE-1,HEIGHT_OF_RECTANGLE-1],
    #      [0,HEIGHT_OF_RECTANGLE - 1]], 
    #   dtype="float32")
    # M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # warped = cv2.warpPerspective(image, M, (WIDTH_OF_RECTANGLE, HEIGHT_OF_RECTANGLE))


    # # warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    # # warped = cv2.flip(warped, 1)
    warped = cv2.resize(warped, (WIDTH_OF_RECTANGLE, HEIGHT_OF_RECTANGLE))
    # # show_image(warped, cvt=cv2.COLOR_RGB2BGR)

    # osd = pytesseract.image_to_osd(warped)
    # angle = int(osd.split('\n')[1].split(':')[1])

    # # Correct the orientation
    # if angle == 90:
    #   rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # elif angle == 180:
    #   rotated = cv2.rotate(warped, cv2.ROTATE_180)
    # elif angle == 270:
    #   rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    # else:
    #   rotated = warped

    # return rotated

    return warped

    # osd = pytesseract.image_to_osd(warped)
    # angle = int(osd.split('\n')[1].split(':')[1])

    # # Correct the orientation
    # if angle == 90:
    #   rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # elif angle == 180:
    #   rotated = warped
    # elif angle == 270:
    #   rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    # else:
    #   rotated = cv2.rotate(warped, cv2.ROTATE_180)

    # return rotated

class CardExtractor:
  def __init__(self, model) -> None:
    self.model = YOLO(model)

  def flatten(self, image, masks):
    return [mask.flatten() for mask in masks]

  # extract card in image
  def predict(self, image):
    results = self.model(image)

    for result in results:
      # masks = result.masks.data
      # cls = result.boxes.cls

      # not found
      if result.masks is None:
        return None

      mask = (result.masks.data[0].cpu().numpy() * 255).astype('uint8')
      h2, w2, c2 = results[0].orig_img.shape
      maskx = cv2.resize(mask, (w2, h2))

      croped = crop_and_straighten(image, maskx)

      return {
        'image': croped,
        'mask': mask
      }

    return None
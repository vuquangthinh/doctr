
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract


# def order_points(pts):
# 	# initialzie a list of coordinates that will be ordered
# 	# such that the first entry in the list is the top-left,
# 	# the second entry is the top-right, the third is the
# 	# bottom-right, and the fourth is the bottom-left
# 	rect = np.zeros((4, 2), dtype = "float32")
# 	# the top-left point will have the smallest sum, whereas
# 	# the bottom-right point will have the largest sum
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]
# 	# now, compute the difference between the points, the
# 	# top-right point will have the smallest difference,
# 	# whereas the bottom-left will have the largest difference
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
# 	# return the ordered coordinates
# 	return rect

# def four_point_transform(image, pts):
# 	# obtain a consistent order of the points and unpack them
# 	# individually
# 	rect = order_points(pts)
# 	(tl, tr, br, bl) = rect
# 	# compute the width of the new image, which will be the
# 	# maximum distance between bottom-right and bottom-left
# 	# x-coordiates or the top-right and top-left x-coordinates
# 	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
# 	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
# 	maxWidth = max(int(widthA), int(widthB))
# 	# compute the height of the new image, which will be the
# 	# maximum distance between the top-right and bottom-right
# 	# y-coordinates or the top-left and bottom-left y-coordinates
# 	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
# 	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
# 	maxHeight = max(int(heightA), int(heightB))
# 	# now that we have the dimensions of the new image, construct
# 	# the set of destination points to obtain a "birds eye view",
# 	# (i.e. top-down view) of the image, again specifying points
# 	# in the top-left, top-right, bottom-right, and bottom-left
# 	# order
# 	dst = np.array([
# 		[0, 0],
# 		[maxWidth - 1, 0],
# 		[maxWidth - 1, maxHeight - 1],
# 		[0, maxHeight - 1]], dtype = "float32")
# 	# compute the perspective transform matrix and then apply it
# 	M = cv2.getPerspectiveTransform(rect, dst)
# 	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
# 	# return the warped image
# 	return warped

# def crop_and_straighten(image, mask):
#     # Find contours in the binary image

#     # expand mask
#     kernel = np.ones((10, 10), np.uint8)  # You can adjust the size of the kernel as per your requirements
#     mask2 = cv2.dilate(mask, kernel, iterations=2)

#     contours, _ = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     rect = cv2.minAreaRect(max(contours, key = cv2.contourArea))

#     # image2 = image.copy()
#     # approximately quadrilateral from contour
#     # aux = cv2.approxPolyDP(contours[0], 0.1 * cv2.arcLength(contours[0], True), True)

#     # # draw aux as contour
#     # cv2.drawContours(image2, [aux], -1, (0, 255, 0), 3)
#     # cv2.drawContours(image2, [contours[0]], -1, (0, 0, 255), 3)

#     # show_image(image2, cvt=cv2.COLOR_RGB2BGR)

#     box = cv2.boxPoints(rect)
#     box = np.int0(box)

#     # src_pts = aux.astype(np.float32)

#     # print(aux)
#     warped = four_point_transform(image, box)

#     # # coordinate of the points in box points after the rectangle has been
#     # # straightened
#     WIDTH_OF_RECTANGLE = 1200
#     HEIGHT_OF_RECTANGLE = 800
#     # dst_pts = np.array(
#     #    [
#     #      [0,0],
#     #      [WIDTH_OF_RECTANGLE - 1,0],
#     #      [WIDTH_OF_RECTANGLE-1,HEIGHT_OF_RECTANGLE-1],
#     #      [0,HEIGHT_OF_RECTANGLE - 1]], 
#     #   dtype="float32")
#     # M = cv2.getPerspectiveTransform(src_pts, dst_pts)

#     # warped = cv2.warpPerspective(image, M, (WIDTH_OF_RECTANGLE, HEIGHT_OF_RECTANGLE))


#     # # warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
#     # # warped = cv2.flip(warped, 1)
#     warped = cv2.resize(warped, (WIDTH_OF_RECTANGLE, HEIGHT_OF_RECTANGLE))
#     # # show_image(warped, cvt=cv2.COLOR_RGB2BGR)

#     # osd = pytesseract.image_to_osd(warped)
#     # angle = int(osd.split('\n')[1].split(':')[1])

#     # # Correct the orientation
#     # if angle == 90:
#     #   rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     # elif angle == 180:
#     #   rotated = cv2.rotate(warped, cv2.ROTATE_180)
#     # elif angle == 270:
#     #   rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
#     # else:
#     #   rotated = warped

#     # return rotated

#     return warped

#     # osd = pytesseract.image_to_osd(warped)
#     # angle = int(osd.split('\n')[1].split(':')[1])

#     # # Correct the orientation
#     # if angle == 90:
#     #   rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     # elif angle == 180:
#     #   rotated = warped
#     # elif angle == 270:
#     #   rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
#     # else:
#     #   rotated = cv2.rotate(warped, cv2.ROTATE_180)

#     # return rotated



# def flatten_mask_to_rectangle(image, mask):
#     # Find contours of the mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter contours to keep the largest one
#     largest_contour = max(contours, key=cv2.contourArea)

#     # Approximate the contour to a polygon
#     epsilon = 0.02 * cv2.arcLength(largest_contour, True)
#     approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

#     # Sort the polygon points
#     sorted_points = sort_quadrilateral_points(approx_polygon.reshape(-1, 2))

#     # Calculate the width and height of the bounding rectangle
#     width = max(np.linalg.norm(sorted_points[0] - sorted_points[1]), np.linalg.norm(sorted_points[2] - sorted_points[3]))
#     height = max(np.linalg.norm(sorted_points[0] - sorted_points[3]), np.linalg.norm(sorted_points[1] - sorted_points[2]))

#     # Create a transformation matrix
#     target_rectangle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
#     transformation_matrix = cv2.getPerspectiveTransform(sorted_points.astype(np.float32), target_rectangle)

#     # Apply the perspective transformation to the mask
#     flattened_mask = cv2.warpPerspective(mask, transformation_matrix, (int(width), int(height)))

#     return flattened_mask

# def sort_quadrilateral_points(points):
#     # Sort the points based on their x+y coordinates
#     sorted_points = sorted(points, key=lambda x: x[0] + x[1])
#     top_left = sorted_points[0]
#     bottom_right = sorted_points[-1]

#     # Sort the remaining points to get top-right and bottom-left
#     remaining_points = sorted_points[1:-1]
#     if remaining_points[0][1] > remaining_points[1][1]:
#         top_right, bottom_left = remaining_points
#     else:
#         bottom_left, top_right = remaining_points

#     return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)






def four_point_transform(image, approx_polygon):
    # Reshape the approx_polygon to have shape (4, 2)
    approx_polygon = approx_polygon.reshape(4, 2)

    # Reorder the points of the polygon to ensure consistent order
    rect = np.zeros((4, 2), dtype="float32")
    s = approx_polygon.sum(axis=1)
    rect[0] = approx_polygon[np.argmin(s)]
    rect[2] = approx_polygon[np.argmax(s)]
    diff = np.diff(approx_polygon, axis=1)
    rect[1] = approx_polygon[np.argmin(diff)]
    rect[3] = approx_polygon[np.argmax(diff)]

    # Calculate the width and height of the new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct the destination points
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply the perspective transformation to the image
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def expand_polygon(approx_polygon, scale_factor):
    # Calculate the centroid of the polygon
    centroid = np.mean(approx_polygon, axis=0)

    # Calculate the vectors from centroid to each vertex
    vectors = approx_polygon - centroid

    # Scale the vectors to expand the polygon
    scaled_vectors = vectors * scale_factor

    # Add the scaled vectors to the centroid to obtain expanded vertices
    expanded_polygon = centroid + scaled_vectors

    # Convert the expanded polygon to integer coordinates
    expanded_polygon = expanded_polygon.astype(int)

    return expanded_polygon

def draw_quadrilateral_on_image(image, mask):
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to keep the largest one
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the convex hull of the largest contour
    hull = cv2.convexHull(largest_contour)

    # Approximate the convex hull to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx_polygon = cv2.approxPolyDP(hull, epsilon, True)
    
    # approx_polygon = expand_polygon(approx_polygon, 1.02)
    # print(approx_polygon)

    # Draw the quadrilateral contour on the image
    # cv2.drawContours(image, [approx_polygon], -1, (0, 255, 0), 2)
    
    new_image = four_point_transform(image, approx_polygon)
    # show_image(new_image)

    return new_image


class CardExtractor:
  def __init__(self, model) -> None:
    self.model = YOLO(model).to('cuda')

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

      croped = draw_quadrilateral_on_image(image, maskx)
      cv2.imwrite('./xyz.crop.jpg', croped)
      cv2.imwrite('./xyz.jpg', image)
      cv2.imwrite('./xyz-mask.jpg', maskx)

      return {
        'image': croped,
        'mask': mask
      }

    return None
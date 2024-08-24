import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/Users/digitcrom/Desktop/multi project /Multimedia_HW3/Shapes.jpg'
image = cv2.imread(image_path)
org_image = image.copy()
image_2 = image.copy()
cv2.imshow('Original Image', org_image)
cv2.waitKey(0) #wait until key pressed

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
blurred = cv2.GaussianBlur(gray, (5, 5), 0) # kernel size = 5 , sigma_x = 0 open cv will calculate it 
_, threshold = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def detect_shape(c):
    shape = ""
    peri = cv2.arcLength(c, True) #closed contours
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    elif len(approx) == 6:
        shape = "hexagon"
    else:
        shape = "circle"
    
    return shape, len(approx) , approx

black_background = np.zeros_like(image)

for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
    else:
        cX, cY = 0, 0
    shape, vertices , approx = detect_shape(contour)
    cv2.drawContours(org_image, [contour], -1, (0, 255, 0), 2)
    cv2.putText(org_image, shape, (cX-25, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if shape in ["square", "rectangle"]:  # Only keep squares and rectangles
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Create a mask and draw the rotated rectangle
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [box], 0, 255, -1)
        # Bitwise-and to extract the region
        rotated_cropped = cv2.bitwise_and(image_2, image_2, mask=mask)

        # Place extracted part on black background
        black_background[mask == 255] = rotated_cropped[mask == 255]




cv2.imshow('Detected Shapes', org_image)
cv2.waitKey(0)

cv2.imshow('squares and rectangles', black_background)
cv2.waitKey(0)
cv2.destroyAllWindows()
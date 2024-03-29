import cv2
import numpy as np

cimg = cv2.imread('esferas.jpg')
cimg = cv2.medianBlur(cimg, 5)
img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=50, param2=50, minRadius=50, maxRadius=100)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('detected circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

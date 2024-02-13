# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import signal
from scipy import misc

img = cv2.imread('newspaper.jpg',0)
# img = cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY)
# Im1 = cv2.cvtColor(Im1, cv2.COLOR_BGR2RGB)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,2)
ret,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(th1,cmap = 'gray')
plt.title('Thresholding'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(th2,cmap = 'gray')
plt.title('Mean thresholding'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(th3,cmap = 'gray')
plt.title('Otsu thresholding'), plt.xticks([]), plt.yticks([])
plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import signal
from scipy import misc

Im1 = cv2.imread('lena.jpg')
img = cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY)

kernel = np.ones((11, 11), np.float32) / 121
img_mean = cv2.filter2D(img, -1, kernel)

kernel = cv2.getGaussianKernel(11, -1)
img_gauss = cv2.filter2D(img, -1, kernel)

img_median = cv2.medianBlur(img, 11)

img_bilat = cv2.bilateralFilter(img, 11, 75, 75)

plt.subplot(221), plt.imshow(img_median, cmap='gray')
plt.title('mediana'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(img_mean, cmap='gray')
plt.title('Filtro promedio'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_gauss, cmap='gray')
plt.title('Filtro gaussiano'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_bilat, cmap='gray')
plt.title('Filtro bilateral'), plt.xticks([]), plt.yticks([])

plt.show()

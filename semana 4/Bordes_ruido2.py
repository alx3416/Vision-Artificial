import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import signal
from scipy import misc

Im1 = cv2.imread('peppersnoise2.png')
img = cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY)

kernel = cv2.getGaussianKernel(11, -1)
img_gauss = cv2.filter2D(img, -1, kernel)

img_bilat = cv2.bilateralFilter(img, 11, 75, 75)

# Con Convolución 2D
Kernel = np.matrix('0 1 0;1 -4 1;0 1 0')  # Kernel laplaciano
Kernel = np.double(Kernel)
Gx = np.double(img)
Gy = np.double(img)

imgLap = signal.convolve2d(img, Kernel, boundary='symm', mode='same')
Kernel_g = cv2.getGaussianKernel(3, -1)
Kernel_log = signal.convolve2d(Kernel, Kernel_g, boundary='symm', mode='same')
imgLog = signal.convolve2d(img, Kernel_log, boundary='symm', mode='same')
imgedges = np.abs(signal.convolve2d(imgLap, np.matrix('-1 1'), boundary='symm', mode='same'))
# Gy[Gy<100]=255

imgLap = signal.convolve2d(img_bilat, Kernel, boundary='symm', mode='same')
imgedges2 = np.abs(signal.convolve2d(imgLap, np.matrix('-1 1'), boundary='symm', mode='same'))

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(img_bilat, cmap='gray')
plt.title('Laplaciano'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(imgedges, cmap='gray')
plt.title('LoG'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(imgedges2, cmap='gray')
plt.title('Log zerocross'), plt.xticks([]), plt.yticks([])
plt.show()

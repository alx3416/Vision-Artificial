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
Kernel = np.matrix('1 2 1;0 0 0;-1 -2 -1')
Kernel = np.double(Kernel)
Gx = np.double(img)
Gy = np.double(img)
Gx = signal.convolve2d(img, Kernel, boundary='symm', mode='same')
Gy = signal.convolve2d(img, Kernel.transpose(), boundary='symm', mode='same')

G1 = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))  # Obtenemos magnitud
Gx = ((Gx - Gx.min()) / (Gx.max() - Gx.min())) * 255
Gx = np.uint8(Gx)
Gy = ((Gy - Gy.min()) / (Gy.max() - Gy.min())) * 255
Gy = np.uint8(Gy)
G1 = ((G1 - G1.min()) / (G1.max() - G1.min())) * 255
G1 = np.uint8(G1)

Kernel = np.matrix('1 2 1;0 0 0;-1 -2 -1')
Kernel = np.double(Kernel)
Gx = np.double(img)
Gy = np.double(img)
Gx = signal.convolve2d(img_bilat, Kernel, boundary='symm', mode='same')
Gy = signal.convolve2d(img_bilat, Kernel.transpose(), boundary='symm', mode='same')

G2 = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))  # Obtenemos magnitud
Gx = ((Gx - Gx.min()) / (Gx.max() - Gx.min())) * 255
Gx = np.uint8(Gx)
Gy = ((Gy - Gy.min()) / (Gy.max() - Gy.min())) * 255
Gy = np.uint8(Gy)
G2 = ((G2 - G2.min()) / (G2.max() - G2.min())) * 255
G2 = np.uint8(G2)

# Gy[Gy<100]=255

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(img_gauss, cmap='gray')
plt.title('filtrada'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(G1, cmap='gray')
plt.title('Gradiente sin filtro'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(G2, cmap='gray')
plt.title('Gradiente con filtro'), plt.xticks([]), plt.yticks([])
plt.show()

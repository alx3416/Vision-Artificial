import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

Im1 = cv2.imread('lena.jpg')
img = cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(img,50,50)

# Con Convolución 2D
Kernel = np.array([1, -1], ndmin=2)
Kernel = np.double(Kernel)
Gx = np.double(img)
Gy = np.double(img)
Gx = signal.convolve2d(img, Kernel, boundary='symm', mode='same')
Gy = signal.convolve2d(img, Kernel.transpose(), boundary='symm', mode='same')

G = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))  # Obtenemos magnitud
Gtheta = np.arctan(Gy / (Gx + 0.01))  # Dirección de cada vector

Gx = ((Gx - Gx.min()) / (Gx.max() - Gx.min())) * 255
Gx = np.uint8(Gx)
Gy = ((Gy - Gy.min()) / (Gy.max() - Gy.min())) * 255
Gy = np.uint8(Gy)
G = ((G - G.min()) / (G.max() - G.min())) * 255
G = np.uint8(G)
Gtheta = ((Gtheta - Gtheta.min()) / (Gtheta.max() - Gtheta.min())) * 255
Gtheta = np.uint8(Gtheta)

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(Gx, cmap='gray')
plt.title('Vertical'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(Gy, cmap='gray')
plt.title('Horizontal'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(G, cmap='gray')
plt.title('Magnitud'), plt.xticks([]), plt.yticks([])
plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import signal
from scipy import misc

Im1 = cv2.imread('lena.jpg')
img = cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY)

p = 3  # Tamaño de ventana
pixels = (p * p) - 1
p = int(np.floor(p / 2))
Size = np.shape(img)
img = np.double(img)
img1 = img + 0.001
img_out = np.zeros((Size[0], Size[1]))
img_temp = np.zeros((Size[0], Size[1]))

for x in range(-p, p):  # filas
    for y in range(-p, p):  # columnas
        if not (x == 0 and y == 0):
            pixels = pixels - 1
            img_temp = np.roll(img1, x, axis=0)
            img_temp = np.roll(img_temp, y, axis=1)
            img_temp = np.floor(img1 / img_temp)
            img_temp[img_temp >= 1] = 1
            img_temp = img_temp * np.power(2, pixels)
            img_out = img_out + img_temp

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_out, cmap='gray')
plt.title('shifted'), plt.xticks([]), plt.yticks([])
plt.show()

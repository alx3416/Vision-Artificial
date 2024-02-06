import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

Im1 = cv2.imread('data/color.jpg')

Im1hsv = cv2.cvtColor(Im1, cv2.COLOR_BGR2HSV)
Im1lab = cv2.cvtColor(Im1, cv2.COLOR_BGR2LAB)

R = Im1[:, :, 0]  # Separación de canales RGB
G = Im1[:, :, 1]
B = Im1[:, :, 2]

H = Im1hsv[:, :, 0]
S = Im1hsv[:, :, 1]
V = Im1hsv[:, :, 2]

L = Im1lab[:, :, 0]
a = Im1lab[:, :, 1]
b = Im1lab[:, :, 2]

Imtemp = Im1.copy()
Im1[:, :, 0] = Imtemp[:, :, 2]  # Cambio de BGR a RGB por cada imagen
Im1[:, :, 2] = Imtemp[:, :, 0]

plt.subplot(432), plt.imshow(Im1), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(434), plt.imshow(R, 'gray'), plt.title('Red')
plt.xticks([]), plt.yticks([])
plt.subplot(435), plt.imshow(G, 'gray'), plt.title('Green')
plt.xticks([]), plt.yticks([])
plt.subplot(436), plt.imshow(B, 'gray'), plt.title('Blue')
plt.xticks([]), plt.yticks([])

plt.subplot(437), plt.imshow(H, 'gray'), plt.title('Hue')
plt.xticks([]), plt.yticks([])
plt.subplot(438), plt.imshow(S, 'gray'), plt.title('Saturation')
plt.xticks([]), plt.yticks([])
plt.subplot(439), plt.imshow(V, 'gray'), plt.title('Vertical')
plt.xticks([]), plt.yticks([])

plt.subplot(4, 3, 10), plt.imshow(L, 'gray'), plt.title('Luminance')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 11), plt.imshow(a,'gray'), plt.title('a-channel')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 12), plt.imshow(b, 'gray'), plt.title('b-channel')
plt.xticks([]), plt.yticks([])
plt.show()

print(H.max())  # Ojo con los rangos, considerar en cálculos y normalizar
print(H.min())
print(S.max())
print(S.min())
print(L.max())
print(L.min())
print(a.max())
print(b.min())

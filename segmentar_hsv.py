import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

Im1 = cv2.imread('data/color.jpg')
Im1hsv = cv2.cvtColor(Im1, cv2.COLOR_BGR2HSV)
H = Im1hsv[:, :, 0]
S = Im1hsv[:, :, 1]  # Tomamos canal S (verdes negativo (0-127) y rojos positivo (128-25))
V = Im1hsv[:, :, 2]  # tomamos canal b (positivos amarillo, negativos azules)
H = np.double(H) # Conversión tipo de dato
S = np.double(S)
V = np.double(V)


K = 0  # Ajuste para determinar tonos de rojo detectados
z = (np.bitwise_and(H >= 11, H <= 35))

new_H = H
new_s = S
new_v = V
new_H[z == 1] = new_H[z == 1] + 100
Im1hsv[:, :, 0] = new_H
Im1hsv[:, :, 1] = new_s
Im1hsv[:, :, 2] = new_v

Im1hsv = np.uint8(Im1hsv)
Im2 = cv2.cvtColor(Im1hsv, cv2.COLOR_HSV2BGR)
Im1 = cv2.cvtColor(Im1, cv2.COLOR_BGR2RGB)
Im2 = cv2.cvtColor(Im2, cv2.COLOR_BGR2RGB)
plt.subplot(1, 3, 1), plt.imshow(Im1, ), plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(Im2, ), plt.title('Color rojo')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(z, ), plt.title('deteccion')
plt.xticks([]), plt.yticks([])
plt.show()

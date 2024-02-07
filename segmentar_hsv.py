import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

Im1 = cv2.imread('data/color.jpg')
Im1lab = cv2.cvtColor(Im1, cv2.COLOR_BGR2HSV)
H = Im1lab[:, :, 0]
S = Im1lab[:, :, 1]  # Tomamos canal S (verdes negativo (0-127) y rojos positivo (128-25))
V = Im1lab[:, :, 2]  # tomamos canal b (positivos amarillo, negativos azules)
H = np.double(H) # Conversión tipo de dato
S = np.double(S)
V = np.double(V)


K = 0  # Ajuste para determinar tonos de rojo detectados
z = (np.bitwise_and(V < 0, np.abs(V) > np.abs(S)))

new_s = S
new_v = V
new_s[z == 1] = np.abs(V[z == 1])
new_v[z == 1] = S[z == 1]
new_s[z == 1] = new_s[z == 1]
Im1lab[:, :, 1] = new_s - 128
Im1lab[:, :, 2] = new_v - 128

Im1lab = np.uint8(Im1lab)
Im2 = cv2.cvtColor(Im1lab, cv2.COLOR_LAB2BGR)
Im1 = cv2.cvtColor(Im1, cv2.COLOR_BGR2RGB)
Im2 = cv2.cvtColor(Im2, cv2.COLOR_BGR2RGB)
plt.subplot(1, 3, 1), plt.imshow(Im1, ), plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(Im2, ), plt.title('Color rojo')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(z, ), plt.title('deteccion')
plt.xticks([]), plt.yticks([])
plt.show()

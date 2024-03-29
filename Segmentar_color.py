import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

Im1 = cv2.imread('data/color.jpg')
Im1lab = cv2.cvtColor(Im1, cv2.COLOR_BGR2LAB)
L = Im1lab[:, :, 0]
a = Im1lab[:, :, 1]  # Tomamos canal S (verdes negativo (0-127) y rojos positivo (128-25))
b = Im1lab[:, :, 2]  # tomamos canal b (positivos amarillo, negativos azules)
# H = np.double(H) # Conversión tipo de dato
a = np.double(a)
b = np.double(b)
a = a - 128
b = b - 128

K = 0  # Ajuste para determinar tonos de rojo detectados
z = (np.bitwise_and(b < 0, np.abs(b) > np.abs(a)))
# z=(S>0)
new_a = a
new_b = b
new_a[z == 1] = np.abs(b[z == 1])
new_b[z == 1] = a[z == 1]
new_a[z == 1] = new_a[z == 1]
Im1lab[:, :, 1] = new_a - 128
Im1lab[:, :, 2] = new_b - 128

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

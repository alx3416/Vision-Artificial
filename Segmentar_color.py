import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

Im1 = cv2.imread('data/peppers.png')
Im1lab = cv2.cvtColor(Im1, cv2.COLOR_BGR2LAB)
L = Im1lab[:, :, 0]
a = Im1lab[:, :, 1]  # Tomamos canal a (verdes negativo (0-127) y rojos positivo (128-25))
b = Im1lab[:, :, 2]  # tomamos canal b (positivos amarillo, negativos azules)
# L = np.double(L) # Conversión tipo de dato
a = np.double(a)
b = np.double(b)
a = a - 128
b = b - 128

K = 16  # Ajuste para determinar tonos de rojo detectados
z = (np.bitwise_and(a > K, (a + K) > np.abs(b)))
# z=(a>0)
new_a = a
new_a[z == 1] = b[z == 1]
new_a[z == 1] = new_a[z == 1] * (-1)

Im1lab[:, :, 0] = L
Im1lab[:, :, 1] = new_a - 128
Im1lab[:, :, 2] = b - 128

Im1lab = np.uint8(Im1lab)
Im2 = cv2.cvtColor(Im1lab, cv2.COLOR_LAB2BGR)
Im1 = cv2.cvtColor(Im1, cv2.COLOR_BGR2RGB)
Im2 = cv2.cvtColor(Im2, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 1), plt.imshow(Im1, ), plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(Im2, ), plt.title('Color rojo')
plt.xticks([]), plt.yticks([])
plt.show()

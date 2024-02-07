import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

Im1 = cv2.imread('data/im1.png')
Im1E = cv2.imread('data/im1E.png')
Im1L = cv2.imread('data/im1L.png')

Im1hsv = cv2.cvtColor(Im1, cv2.COLOR_BGR2HSV)
Im1Ehsv = cv2.cvtColor(Im1E, cv2.COLOR_BGR2HSV)
Im1Lhsv = cv2.cvtColor(Im1L, cv2.COLOR_BGR2HSV)

Im1lab = cv2.cvtColor(Im1, cv2.COLOR_BGR2LAB)
Im1Elab = cv2.cvtColor(Im1E, cv2.COLOR_BGR2LAB)
Im1Llab = cv2.cvtColor(Im1L, cv2.COLOR_BGR2LAB)

R = Im1[:, :, 0]  # Separación de canales RGB
G = Im1[:, :, 1]
B = Im1[:, :, 2]
RE = Im1E[:, :, 0]  # Separación de canales RGB
GE = Im1E[:, :, 1]
BE = Im1E[:, :, 2]
RL = Im1L[:, :, 0]  # Separación de canales RGB
GL = Im1L[:, :, 1]
BL = Im1L[:, :, 2]

H = Im1hsv[:, :, 0]
S = Im1hsv[:, :, 1]
V = Im1hsv[:, :, 2]
HE = Im1Ehsv[:, :, 0]
SE = Im1Ehsv[:, :, 1]
VE = Im1Ehsv[:, :, 2]
HL = Im1Lhsv[:, :, 0]
SL = Im1Lhsv[:, :, 1]
VL = Im1Lhsv[:, :, 2]

L = Im1lab[:, :, 0]
a = Im1lab[:, :, 1]
b = Im1lab[:, :, 2]
LE = Im1Elab[:, :, 0]
aE = Im1Elab[:, :, 1]
bE = Im1Elab[:, :, 2]
LL = Im1Llab[:, :, 0]
aL = Im1Llab[:, :, 1]
bL = Im1Llab[:, :, 2]

Imtemp = Im1.copy()
Im1[:, :, 0] = Imtemp[:, :, 2]  # Cambio de BGR S RGB por cada imagen
Im1[:, :, 2] = Imtemp[:, :, 0]
Imtemp = Im1E.copy()
Im1E[:, :, 0] = Imtemp[:, :, 2]  # Cambio de BGR S RGB por cada imagen
Im1E[:, :, 2] = Imtemp[:, :, 0]
Imtemp = Im1L.copy()
Im1L[:, :, 0] = Imtemp[:, :, 2]  # Cambio de BGR S RGB por cada imagen
Im1L[:, :, 2] = Imtemp[:, :, 0]

f1 = plt.figure(1)
plt.subplot(431), plt.imshow(Im1), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(432), plt.imshow(Im1E), plt.title('Exposure')
plt.xticks([]), plt.yticks([])
plt.subplot(433), plt.imshow(Im1L), plt.title('Light')
plt.xticks([]), plt.yticks([])
plt.subplot(434), plt.imshow(R, 'gray'), plt.title('Red')
plt.xticks([]), plt.yticks([])
plt.subplot(437), plt.imshow(G, 'gray'), plt.title('Green')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 10), plt.imshow(B, 'gray'), plt.title('Blue')
plt.xticks([]), plt.yticks([])
plt.subplot(435), plt.imshow(RE, 'gray'), plt.title('Red')
plt.xticks([]), plt.yticks([])
plt.subplot(438), plt.imshow(GE, 'gray'), plt.title('Green')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 11), plt.imshow(BE, 'gray'), plt.title('Blue')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 6), plt.imshow(RL, 'gray'), plt.title('Red')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 9), plt.imshow(GL, 'gray'), plt.title('Green')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 12), plt.imshow(BL, 'gray'), plt.title('Blue')
plt.xticks([]), plt.yticks([])

f1 = plt.figure(2)
plt.subplot(431), plt.imshow(Im1), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(432), plt.imshow(Im1E), plt.title('Exposure')
plt.xticks([]), plt.yticks([])
plt.subplot(433), plt.imshow(Im1L), plt.title('Light')
plt.xticks([]), plt.yticks([])
plt.subplot(434), plt.imshow(H, 'gray'), plt.title('Hue')
plt.xticks([]), plt.yticks([])
plt.subplot(437), plt.imshow(S, 'gray'), plt.title('Saturation')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 10), plt.imshow(V, 'gray'), plt.title('Vertical')
plt.xticks([]), plt.yticks([])
plt.subplot(435), plt.imshow(HE, 'gray'), plt.title('Hue')
plt.xticks([]), plt.yticks([])
plt.subplot(438), plt.imshow(SE, 'gray'), plt.title('Saturation')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 11), plt.imshow(VE, 'gray'), plt.title('Vertical')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 6), plt.imshow(HL, 'gray'), plt.title('Hue')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 9), plt.imshow(SL, 'gray'), plt.title('Saturation')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 12), plt.imshow(VL, 'gray'), plt.title('Vertical')
plt.xticks([]), plt.yticks([])

f1 = plt.figure(3)
plt.subplot(431), plt.imshow(Im1), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(432), plt.imshow(Im1E), plt.title('Exposure')
plt.xticks([]), plt.yticks([])
plt.subplot(433), plt.imshow(Im1L), plt.title('Light')
plt.xticks([]), plt.yticks([])
plt.subplot(434), plt.imshow(L, 'gray'), plt.title('Luminance')
plt.xticks([]), plt.yticks([])
plt.subplot(437), plt.imshow(a, 'gray'), plt.title('S-channel')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 10), plt.imshow(b, 'gray'), plt.title('b-channel')
plt.xticks([]), plt.yticks([])
plt.subplot(435), plt.imshow(LE, 'gray'), plt.title('Luminance')
plt.xticks([]), plt.yticks([])
plt.subplot(438), plt.imshow(aE, 'gray'), plt.title('S-channel')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 11), plt.imshow(bE, 'gray'), plt.title('b-channel')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 6), plt.imshow(LL, 'gray'), plt.title('Luminance')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 9), plt.imshow(aL, 'gray'), plt.title('S-channel')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 12), plt.imshow(bL, 'gray'), plt.title('b-channel')
plt.xticks([]), plt.yticks([])
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:17:24 2021

@author: Alejandro
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Opción 0 es para leer en escala de grises
img = cv2.imread('lena.jpg',0)


dst = cv2.cornerHarris(img,3,3,0.05)
# imagen, blocksize, ksize(Sobel), k-value (0.04 a 0.06)
print(dst.min())
print(dst.max())

# replicamos matriz en eje Z para que sea de 3 canales
img_out = img[:,:,np.newaxis]
img_out=np.tile(img_out,(1,1,3))
img_out[dst>(0.01*dst.max())]=[255,0,0]
mask=img*0
mask[dst>(0.01*dst.max())]=255


plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(dst,cmap = 'gray')
plt.title('harris'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img_out,cmap = 'gray')
plt.title('Deteccion'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(np.uint8(mask),cmap = 'gray')
plt.title('Deteccion bin'), plt.xticks([]), plt.yticks([])
plt.show
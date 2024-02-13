
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

img = cv2.imread('university.png',0)
img2 = cv2.imread('lowcontrastcolor2.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#img = cv2.cvtColor(imc, cv2.COLOR_BGR2GRAY)

R=img2[:,:,0] #Separación de canales RGB
G=img2[:,:,1]
B=img2[:,:,2]

img_eq = cv2.equalizeHist(img)

R_eq = cv2.equalizeHist(R)
G_eq = cv2.equalizeHist(G)
B_eq = cv2.equalizeHist(B)
img2_eq = img2.copy()
img2_eq[:,:,0] = R_eq #Separación de canales RGB
img2_eq[:,:,1] = G_eq
img2_eq[:,:,2] = B_eq

plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original grises'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(img_eq,cmap = 'gray')
plt.title('Grises ecualizada'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img2,cmap = 'gray')
plt.title('Original color'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(img2_eq,cmap = 'gray')
plt.title('Ecualizada color'), plt.xticks([]), plt.yticks([])
plt.show()
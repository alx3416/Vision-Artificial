# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:53:16 2019

@author: alx34
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('sala2.jpg')
img2 = cv2.imread('sala1.jpg')
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
#img1g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#img2g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 5,sigma = 1.6)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# SURF, ORB
# Apply ratio test
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
        matches = np.asarray(good)

if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#print H
else:
    raise AssertionError("Can’t find enough keypoints.")

dst = cv2.warpPerspective(img1,H,(img2.shape[1] + img1.shape[1], img2.shape[0]))
plt.subplot(221),plt.imshow(img1),plt.title('Imagen 1')
plt.subplot(222),plt.imshow(img2),plt.title('Imagen 2')
plt.subplot(223),plt.imshow(dst),plt.title('Warped Image')

dst[0:img2.shape[0], 0:img2.shape[1]] = img2
plt.subplot(224),plt.imshow(dst),plt.title('Resultado')
#plt.imshow(dst)
plt.show()
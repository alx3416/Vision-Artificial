# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:57:19 2019

@author: alx34
"""

import numpy as np
import cv2


cap = cv2.VideoCapture(1)
ret, frame = cap.read()

img1 = cv2.imread('aceite.png',0) 

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 2,sigma = 1.6)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(frame,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,frame,kp2,good,None,flags=2)

    cv2.imshow('SIFT',img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:00:34 2019

@author: alx34
"""

import numpy as np
import cv2 as cv
img = cv.imread('peppers.png')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
#img=cv.drawKeypoints(gray,kp,img) # Dibujar SIFT keypoints
img=cv.drawKeypoints(gray,kp,gray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #Dibujar SIFT keypoints de acuerdo a sigma y orientación
cv.imshow('sift_keypoints',img)
cv.waitKey(0)
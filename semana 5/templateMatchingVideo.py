# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:42:36 2019

@author: alx34
"""

import numpy as np
import cv2
from scipy import signal
from scipy import misc

import time

cap = cv2.VideoCapture('futbol2.mp4')
ret, frame = cap.read()
r,h,c,w = 230,20,325,25
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
roi = frame_gray[r:r+h, c:c+w]
Im2 = cv2.imread('obj2.png')
roi2 = cv2.cvtColor(Im2, cv2.COLOR_BGR2GRAY)
w2, h2 = roi2.shape[::-1]

w1, h1 = roi.shape[::-1]

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
    res = cv2.matchTemplate(img,roi,cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w1, pt[1] + h1), (0,0,255), 2)
    
    res = cv2.matchTemplate(img,roi2,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w2, pt[1] + h2), (0,0,255), 2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:57:19 2019

@author: alx34
"""

import numpy as np
import cv2
import pywt
import pywt.data

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Size=np.shape(img)
img= np.double(img)
img1=img+0.001;
img_out = np.zeros((Size[0],Size[1]))
img_temp = np.zeros((Size[0],Size[1]))


while(True):
    # Capturas frame
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    
    LL=np.uint8(LL/2)
    LH = ((LH-LH.min())/(LH.max()-LH.min()))*255
    LH = np.uint8(LH)
    
    HL = ((HL-HL.min())/(HL.max()-HL.min()))*255
    HL = np.uint8(HL)
        
    HH = ((HH-HH.min())/(HH.max()-HH.min()))*255
    HH = np.uint8(HH)
    
    cv2.imshow('LOW-LOW',LL)
    cv2.imshow('LOW-HIGH',LH)
    cv2.imshow('HIGH-LOW',HL)
    cv2.imshow('HIGH-HIGH',HH)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
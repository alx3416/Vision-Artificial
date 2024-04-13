import numpy as np
import cv2
from Block_matching_SAD import BlockMatchingSAD

img1 = cv2.imread('Venus/frame10.png')
img2 = cv2.imread('Venus/frame11.png')

resp = BlockMatchingSAD(img1, img2, 9, 15)

cv2.imshow('flujo optico', resp)
cv2.waitKey()

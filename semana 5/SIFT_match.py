import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('PlayroomBox.png', 0)
img2 = cv2.imread('Playroom.png', 0)

# Initiate SIFT detector
sift = cv2.SIFT_create(nOctaveLayers=5, sigma=1.6)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.2 * n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# plt.imshow(img3),plt.show()
cv2.imshow('sift_keypoints', img3)
cv2.waitKey()

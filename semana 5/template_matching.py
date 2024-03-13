import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import signal
from scipy import misc

Im1 = cv2.imread('coca1.jpeg')
img = cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY)
Im2 = Im1
out1 = img[640:770, 370:570]

img = np.roll(img, 80, axis=0)
img = np.roll(img, 240, axis=1)

img = np.double(img)
out = np.double(out1)
a = (img - np.mean(img)) / (np.std(img) * np.size(out))
b = (out - np.mean(out)) / (np.std(out))
c = signal.correlate(a, b, mode='valid')
Im1 = cv2.cvtColor(Im1, cv2.COLOR_BGR2RGB)
Im2 = cv2.cvtColor(Im2, cv2.COLOR_BGR2RGB)
ind = np.unravel_index(np.argmax(c, axis=None), c.shape)  # Encuentra posición de máximo valor

cv2.rectangle(Im1, (370, 640), (570, 770), (0, 255, 0), 3)

plt.subplot(221), plt.imshow(Im1)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(out1, cmap='gray')
plt.title('template'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(c, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

cv2.rectangle(img, (ind[1], ind[0]), (ind[1] + 200, ind[0] + 130), (255, 255, 255), 3)

plt.subplot(224), plt.imshow(img, cmap='gray')
plt.title('template'), plt.xticks([]), plt.yticks([])
plt.show()

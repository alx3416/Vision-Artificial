import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(256)/255
y=np.tile(x,(256,1))
y2 = np.power(y,3)
X, Y = np.meshgrid(x, x)

plt.subplot(121),plt.imshow(y,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(y2,cmap = 'gray'),plt.title('modificada')
plt.xticks([]), plt.yticks([])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, y)
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, y2)
plt.show()
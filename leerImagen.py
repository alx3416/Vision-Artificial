import cv2
import numpy as np

colorImage = cv2.imread('data/Alabama.jpg') # formato BGR


singleValue = colorImage[119, 229, 1]
pixelValues = colorImage[119, 229, :]
print("Valor de un punto y un canal especifico: " + str(singleValue))
print("Valores de un pixel: " + str(pixelValues))

cropped_image = colorImage[119:219, 229:329, :]

cv2.imshow('Imagen', colorImage)
cv2.imshow('Recorte', cropped_image)
cv2.waitKey(0)

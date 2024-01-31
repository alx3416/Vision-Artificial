import cv2
import numpy as np

colorImage = cv2.imread('data/Alabama.jpg') # formato BGR


singleValue = colorImage[119, 229, 1]
pixelValues = colorImage[119, 229, :]
print("Valor de un punto y un canal especifico: " + str(singleValue))
print("Valores de un pixel: " + str(pixelValues))

cropped_image = colorImage[119:219, 229:329, :]
blue = colorImage.copy()
blue[:, :, 1] = blue[:, :, 1] * 0
blue[:, :, 2] = blue[:, :, 2] * 0

filas, cols, chan = colorImage.shape
red = np.zeros([filas, cols, chan], dtype=np.uint8)
red[:, :, 2] = colorImage[:, :, 2]

filas, cols, chan = colorImage.shape
yellow = np.zeros([filas, cols, chan])
yellow[:, :, 2] = colorImage[:, :, 2]
yellow = yellow / 255

cv2.imshow('Imagen', colorImage)
cv2.imshow('tonos azules', blue)
cv2.imshow('tonos rojos', red)
cv2.imshow('tonos amarillos', yellow)
cv2.imshow('Recorte', cropped_image)
cv2.waitKey(0)

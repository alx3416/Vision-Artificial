# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:53:14 2019

@author: alx3416@github.com
"""


def BlockMatchingSAD(imL, imR, w, p):
    import cv2
    import numpy as np
    from scipy import signal

    Size = np.shape(imL)
    LEFT = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    RIGHT = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
    LEFT = np.double(LEFT)
    RIGHT = np.double(RIGHT)
    # Magnitud máxima posible
    M = np.sqrt(np.power(w, 2) + np.power(w, 2))

    h = np.ones((1, p)) * -1
    v = np.ones((p, 1)) * -1

    sadLEFT = np.zeros((Size[0], Size[1], 2))
    sadRIGHT = np.zeros((Size[0], Size[1], 2))
    sadLEFT[:, :, :] = np.inf
    sadRIGHT[:, :, :] = np.inf
    ddppX = np.zeros((Size[0], Size[1]))
    ddppY = np.zeros((Size[0], Size[1]))
    D1 = np.zeros((Size[0], Size[1]))
    #
    for c in range(-w, w):  # niveles de disparidad
        for f in range(-w, w):
            SHIFTED = np.roll(RIGHT, f, axis=0)
            SHIFTED = np.roll(SHIFTED, c, axis=1)
            SHIFTED = np.abs(LEFT - SHIFTED);
            sadLEFT[:, :, 1] = signal.convolve2d(SHIFTED, h, boundary='symm', mode='same')
            sadLEFT[:, :, 1] = signal.convolve2d(sadLEFT[:, :, 1], v, boundary='symm', mode='same')
            D1 = np.argmin(sadLEFT, axis=2)  # posición mínimo
            sadLEFT[:, :, 0] = np.min(sadLEFT, axis=2)  # valor minimo

            ddppX[D1 == 1] = c
            ddppY[D1 == 1] = f

    # Obtenemos magnitud (este será canal S de HSV)
    ddppMag = np.sqrt(np.power(ddppX, 2) + np.power(ddppY, 2))
    # Ajuste de rango de valores hacia 0 a 255, sabemos que el rango va de 0 a M, ajustamos
    S = np.uint8(ddppMag / (M) * 255)

    # Obtenemos dirección, este será canal H
    ddpptheta = np.arctan(ddppY / (ddppX + 0.01))  # Dirección de cada vector
    # Ajuste de rango de valores hacia 0 a 255, sabemos que el rango va de -pi/2 a pi/2, ajustamos
    H = np.uint8(((ddpptheta - (-1.5707)) / ((1.5707) - (-1.5707))) * 180)
    # Creamos canal V
    V = np.uint8(np.zeros((Size[0], Size[1])) + 127)

    # Unimos canales (merge) para formar imagen HSV
    img_out = cv2.merge((H, S, V))
    img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2BGR)

    return img_out

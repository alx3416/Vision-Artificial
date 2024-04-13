# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:27:26 2019

@author: alx34
"""

def LBP (img, p):
    import numpy as np
    
#Matrices de prueba, debe ser 6.5 resultado
    
#    I=np.double(np.matrix('1 2 3;4 5 6;7 8 9'))
#    H=np.matrix('1 1 1;2 2 2;3 3 3')
    pixels=(p*p)-1;
    k=np.int(np.floor(p/2));
    Size=np.shape(img)
    img_out = np.zeros((Size[0],Size[1]))
    pixels=(p*p)-1;
    img = np.double(img)
    img1=img+0.001;
    img_temp = np.zeros((Size[0],Size[1]))
    for x in range(-k, k): # filas
        for y in range(-k, k): # columnas
            if not(x==0 and y==0):
                pixels = pixels-1
                img_temp = np.roll(img1, x, axis=0)
                img_temp = np.roll(img_temp, y, axis=1)
                img_temp = np.floor(img1/img_temp)
                img_temp[img_temp>=1]=1
                img_temp = img_temp*np.power(2,pixels)
                img_out = img_out+img_temp
            
    img_out = ((img_out-img_out.min())/(img_out.max()-img_out.min()))*255
    img_out = np.uint8(img_out)
    return img_out;
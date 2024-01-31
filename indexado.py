# -*- coding: utf-8 -*-
"""
@author: Victor
"""
# Librerias a utilizar
import numpy as np # Importamos numpy y la usaremos con el prefijo np
from matplotlib import pyplot as plt #Importamos desde matplotlib la lib pyplot
import cv2

# Tipos de dato
# Hay 3 tipos de datos numéricos: enteros, flotantes y complejos, usaremos numpy
# tipo de dato booleano se considera lógico (0,1-true,false)
# en numpy tenemos varios tipos de datos enteros
# int8, int16 y int32, tambien en variante de positivos unit8, unit16, unit32


a=np.float(0.5) # pfm #dicom #pgm
# El rango seria de 0 a 255, un valor mayor a 127, se vuelve a contar desde 0
# los decimales se truncan, sin redondeo

a=np.float(3.14)
# float es el equivalente a dobule, 

#Creación de matrices (Arrays) con numpy, principal es con función np.array

# Creamos una matriz o vector declarando directamente los valores
a1=np.array([1, 2, 3]) # vector

a2=np.array([[1, 2], [3, 4]]) # matriz (2 dimensiones, 2 corchetes)

# especificar número de dimensiones
a1=np.array([1, 2, 3], ndmin=2)

# especificar tipo de dato
a4=np.array([1, 2, 3], dtype=int) # en tipo de dato son los de python (bool, int, float, complex)
a4=np.uint8(a4)
# Creación de matrices sin declarar valores individualmente
# las opciones son zeros, ones, empty, full
a = np.zeros((40,80)) # filas, y columnas
b = np.zeros(5)
b = np.zeros(5,dtype=bool)
c = np.zeros((40,80,3),dtype=int)

# Creación de array con una constante
a = np.full((4,8),3.14)
a[0,0]
a[1,2]=np.inf
a = np.full((4,8),np.inf) # También hay valores como inf, -inf y nan

# Creación de array valores aleatorios entre 0 y 1
a = np.random.rand(3,2)
# Selección valores aleatorios enteros de 0 a 4
a = np.random.randint(5, size=(2, 4))

# Operadores lógicos, estos a la salida siempre darán tipo booleano (true, false)
# https://numpy.org/doc/stable/reference/routines.logic.html

#Operadores IS
# Nos entregan un array evaluando cada valor si cumple o no la condición
x=np.isinf(a)
# Si queremos una decisión de toda una matriz, podemos usar any, all

np.all(np.isinf(a)) # Si todos los elementos de a son inf, sera true

np.any(np.isinf(a)) # Si algun elemento de a es inf, sera true

# Operaciones lógicas (==, >=, <=, <,>,!=) Para valores individuales
x = 2==2
x = np.inf==-np.inf
x = np.nan!=np.nan
# Otra opción es usar logica booleana and &, or |, not, xor

# NOTA los operadores lógicos solo se emplean en variables booleanas 

np.logical_and(True, False)
True & False

# Para aplicar operaciones lógicas en un array, usamos mismos operadores
# a la salida obtendremos array booleano, esto es aplicar función element-wise
x=a2>2
# Con bitwise_and podemos aplicar operador logico a array
x=np.bitwise_and(a2>2,a2>0)


# Propiedades de arrays dentro de numpy
a.ndim # numero de dimensiones
a.size # numero de elementos
a.shape # numero de filas y columnas
f,_=a.shape #Solo filas

# Hacer vectores y matrices con rangos de valoresm secuenciales o aleatorios
# Función arange, inicio, final e incremento por default 1
np.arange(3) # final iniciando en 0
np.arange(3,10) # inicio y hasta lo mas cercano a valor final
np.arange(3,10,0.5) # Inicio, final, incremento

# función linspace, incio, final, y cantidad de elementos del vector
np.linspace(2.0, 3.0, num=7) # inicio, final, cantidad de valores (incluye los valores inicio-final)

# Función tile, para repetir un array en alguna dimensión especificada
a = np.array([0, 1, 2])
np.tile(a, 2) # replicar vector 
np.tile(a, (2, 3)) # replicar 2 veces en filas, 3 veces en columnas

# función meshgrid
# Sirve para obtener 2 arrays con rangos de valores para filas y columnas 
# El operador : indica el inicio y final
# primera posicion, rango de 0 a 5, habra 5 filas con valores de 0 a 4
# segunda posicion, rango de 0 a 3, habra 3 filas con valores de 0 a 2
np.mgrid[0:5,0:3]
# primera salida, incrementos por columna
# segunda fila incrementos por fila

# Opción mas usada es indicando los incrementos como variables
nx, ny = (640, 480) # limite en x, filas de 3, limite en y, filas de 2
x = np.linspace(0, 1, nx) #creamos vector, inicia en 0 termina en 1, 640 valores
y = np.linspace(0, 1, ny) #creamos vector, inicia en 0 termina en 1, 480 valores
xv, yv = np.meshgrid(x, y) # primera salida, incrementos derecha, segunda salida incrementos abajo

a = np.zeros((256,256))

[rows,cols]=a.shape #guardando numero de filas y columnas
c=0

for y in range(0,rows):
    for x in range(0,cols):
        a[y,x]=c
        c=c+1
    
plt.imshow(a,cmap = 'hot')
plt.show()

a=a/3199 # valores van de 0 a 1

# tipos de variables
# float64 y float 32 1x10^-31 hasta 1x10^31
# int8 2^8=256 (-127 a 127) uint8 (0 a 255) uint16 uint32

a=np.uint8(a*255)

# cv2.imshow('Imagen',a)
# cv2.waitKey(0)



def mifuncion(x1,x2):
    out=x1+x2
    return out;

y=mifuncion(9,8)
print(y)
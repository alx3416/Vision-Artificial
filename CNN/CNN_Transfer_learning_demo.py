# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:46:18 2020

@author: alx34
"""
# Objetivo es tomar modelo VGG16 y reentrenar añadiendo capas FC, con salida a 10 clases
# Pesos originales serán de ImageNet y serán fijos
# Solo se reentrenará FC con dataset CIFAR10
# Entrenamiento será condataset mas pequeño y menor número de clases
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Add, Activation, Input, concatenate
from keras import backend as K
import keras
import numpy as np

# CIFAR10 carga de dataset y sus características
batch_size = 128
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



# Opciones para iniciar modelos previamente entrenados
# Selección de salida, ultima capa CNN
base_model = VGG16(weights='imagenet', include_top=False,input_shape=(32, 32, 3)) # Hasta ultima capa CNN, entrada será de CIFAR10

# Añadimos capas a modelo a entrenar
x = base_model.output # tomamos salida de VGG16 (modelo base)
x=Flatten()(x)# Vectorizar datos
x = Dense(128, activation='relu')(x) # Añadimos 1 capa FC
predictions = Dense(10, activation='softmax')(x) # Determinamos salida a 10 clases

# Modelo a entrenar
new_model = Model(inputs=base_model.input, outputs=predictions)
# Resumen del nuevo modelo (El tamaño mínimo de imagen para VGG16 es 32x32)


# Etiquetamos como NO entrenables las capas CNN, también podemos seleccionar a mano cuales son entrenables
for layer in base_model.layers:
    layer.trainable = False
    
# Si queremos congelar solo las 10 primeras y reentrenar el resto (pesos iniciales son de ImageNet)
for layer in base_model.layers[:17]: # Se cuentan de arriba a abajo en summary, incluyendo Input y pooling
  layer.trainable = False
for layer in base_model.layers[17:]: # Entrenamiento será a ultima CNN de bloque 5 y FC
  layer.trainable = True

new_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
new_model.summary()
#
new_model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,
          verbose=1,validation_data=(x_test, y_test))
score = new_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#new_model.save('saved_model/cifar10_VGG16Transfer.h5')

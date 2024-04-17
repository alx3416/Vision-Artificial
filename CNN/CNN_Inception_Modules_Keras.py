# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:28:00 2020

@author: alx34
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Add, Activation, Input, concatenate
from keras import backend as K

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

inputTensor = Input(input_shape) #Entrada de 32x32x3
a1=Conv2D(96, kernel_size=(3, 3),activation='relu')(inputTensor) # 3x3x96 ==> Salida 30x30x96
a2=Conv2D(192, kernel_size=(3, 3),activation='relu')(a1) # 3x3x192 ==> Salida 28x28x192
# A partir de aqui definimos un modulo inception como los usados en GoogLeNet
# Input = 28x28x192
#Path 1
a2_1=Conv2D(64, kernel_size=(1, 1),activation='relu')(a2) # Conv 1x1 salida 28x28x64
#Path 2
a2_2=Conv2D(96, kernel_size=(1, 1),activation='relu')(a2) # Conv 1x1 salida 28x28x96
a2_2=Conv2D(128, kernel_size=(3, 3),activation='relu',padding='same')(a2_2) # Conv 3x3 salida 28x28x128
#Path 3
a2_3=Conv2D(16, kernel_size=(1, 1),activation='relu')(a2) # Conv 1x1 salida 28x28x16
a2_3=Conv2D(32, kernel_size=(5, 5),activation='relu',padding='same')(a2_3) # Conv 5x5 salida 28x28x32
#Path 4
a2_4=MaxPooling2D(pool_size=(3, 3),padding='same',strides=(1, 1))(a2) #Salida pooling 28x28x192
a2_4=Conv2D(32, kernel_size=(1, 1),activation='relu')(a2_4) # Conv 1x1 salida 28x28x32
#Concatenar los 4 bloques para crear un solo volumen
a3=concatenate([a2_1,a2_2,a2_3,a2_4]) # Salida 28x28x256
# Fin del módulo inception
#Output = 28x28x256
a3=Conv2D(288, kernel_size=(3, 3),activation='relu')(a3)
a3=MaxPooling2D(pool_size=(2, 2))(a3) # maxpooling 2x2
a3=Conv2D(320, kernel_size=(3, 3),activation='relu')(a3)
a3=Conv2D(352, kernel_size=(3, 3),activation='relu')(a3)
a3=Conv2D(384, kernel_size=(3, 3),activation='relu')(a3)
a3=Flatten()(a3)# Vectorizar datos
a4=Dense(128, activation='relu')(a3)
a4=Dense(64, activation='relu')(a3)
a4=Dense(32, activation='relu')(a3)
out=Dense(num_classes, activation='softmax')(a3)

inception_model = Model(inputs=inputTensor, outputs=out)

inception_model.summary()

inception_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#
inception_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = inception_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

inception_model.save('cifar10_inception')
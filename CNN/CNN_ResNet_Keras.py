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
from keras.layers import Conv2D, MaxPooling2D, Add, Activation, Input
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
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

# Debemos definir suma de activaciones, y tener en cuenta que dimensiones deben ser iguales al momento de la suma
#model = Sequential()
inputTensor = Input(input_shape)
a1=Conv2D(32, kernel_size=(3, 3),activation='relu')(inputTensor) # 3x3x32
a2=Conv2D(64, kernel_size=(3, 3),activation='relu')(a1) # 3x3x64
w3=Conv2D(64, (3, 3), activation=None,padding='same')(a2) # 3x3x64 (2 capas seguidas, esta será sin activación, tiene misma dimensión que anterior)
z3=Add()([w3, a2])
a3=Activation('relu')(z3)
a3=MaxPooling2D(pool_size=(2, 2))(a3) # maxpooling 2x2
a3=Dropout(0.25)(a3) # aleatoriamente 25% de los pesos se ajustan a cero cada update para prevenir overfit en trainning
a3=Flatten()(a3)# Vectorizar datos
a4=Dense(128, activation='relu')(a3)
out=Dense(num_classes, activation='softmax')(a4)

res_net_model = Model(inputs=inputTensor, outputs=out)
res_net_model.summary()

res_net_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#
res_net_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = res_net_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
res_net_model.save('cifar10_resnet')
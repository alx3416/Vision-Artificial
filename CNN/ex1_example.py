import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv

# Model / data parameters
num_classes = 2
input_shape = (20, 20, 6)

# Load the data and split it between train and test sets
inhabitatedData = np.load("data/sampled_inhabited.npy")
uninhabitatedData = np.load("data/sampled_uninhabited.npy")

fullData = np.concatenate((inhabitatedData, uninhabitatedData), axis=0)

maximos = np.zeros((6, 60000), dtype=np.int16)
minimos = np.zeros((6, 60000), dtype=np.int16)
for channel in range(6):
    for image in range(60000):
        maximos[channel, image] = np.max(fullData[image, :, :, channel])
        minimos[channel, image] = np.min(fullData[image, :, :, channel])

data_normalized = (fullData + 9999) / (20000 + 9999)
data_labels = np.concatenate((np.zeros(30000), np.ones(30000)), axis=0)

indices = np.random.permutation(60000)
training_idx, test_idx = indices[:int(60000 * 0.7)], indices[int(60000 * 0.7):]
x_train, x_test = fullData[training_idx, :, :, :], fullData[test_idx, :, :, :]
y_train, y_test = data_labels[training_idx], data_labels[test_idx]

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 32
epochs = 20

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[callback])

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save('model.keras')

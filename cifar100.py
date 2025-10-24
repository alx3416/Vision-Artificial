import keras
from keras import layers, models, ops

# Cargar CIFAR-100
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# Normalizar imágenes
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Definir modelo usando API funcional
inputs = layers.Input(shape=(32, 32, 3))

# Capa CNN 1
x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Capa CNN 2
x = layers.Conv2D(128, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Clasificador
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(100, activation='softmax')(x)

# Crear modelo
model = models.Model(inputs=inputs, outputs=outputs)

# Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_split=0.1
)

# Evaluar en test
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_accuracy:.4f}')
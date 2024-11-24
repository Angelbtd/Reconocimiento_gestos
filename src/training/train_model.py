import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Cargar los archivos .npy
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Verificar las formas de los datos cargados
print("Shape de X_train:", X_train.shape)
print("Shape de X_test:", X_test.shape)
print("Shape de y_train:", y_train.shape)
print("Shape de y_test:", y_test.shape)

# Construir el modelo de red neuronal convolucional (CNN)
model = models.Sequential([
    layers.InputLayer(input_shape=(64, 64, 3)),  # Imagen de entrada de 64x64x3
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(y_train)), activation='softmax')  # Número de clases de gestos
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Crear la carpeta 'saved_model' si no existe
os.makedirs('saved_model', exist_ok=True)

# Guardar el modelo entrenado en formato SavedModel
model.save('saved_model/gesture_model.keras')


print("Modelo entrenado y guardado como 'saved_model/gesture_model'")

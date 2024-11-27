import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Ruta donde están las imágenes
data_dir = 'data/raw'

# Lista de gestos (14 gestos en total)
gestures = ["ok", "pausa", "pulgar_arriba", "saludos", "adios", "yo", "gracias", "perdon", "no", "uno", "dos", "tres", "cuatro", "cinco"]

# Inicializar listas para las imágenes y las etiquetas
images = []
labels = []

# Recorrer las carpetas de los gestos
for gesture_index, gesture in enumerate(gestures):
    gesture_path = os.path.join(data_dir, gesture)
    
    # Recorrer las imágenes dentro de cada carpeta
    for img_name in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # Redimensionar las imágenes
        images.append(img)
        labels.append(gesture_index)  # Guardar el índice del gesto

# Convertir las listas a arrays de numpy
images = np.array(images)
labels = np.array(labels)

# Normalizar las imágenes (valores entre 0 y 1)
images = images / 255.0

# Codificar las etiquetas en formato one-hot
labels = to_categorical(labels, num_classes=len(gestures))

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definir el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(gestures), activation="softmax")  # Aquí cambiamos a 14 salidas
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# Guardar el modelo entrenado
model.save('gesture_model.h5')
print("Modelo entrenado y guardado correctamente")

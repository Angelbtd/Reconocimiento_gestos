import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define los gestos que has capturado
gestures = ["ok", "pausa", "pulgar_arriba", "saludos", "adios", "yo", "gracias", "perdon", "no", "hola"]

# Ruta donde están las imágenes
data_dir = "data/raw"  # Ruta para las imágenes crudas

# Listas para almacenar las imágenes y sus etiquetas
images = []
labels = []

# Recorrer las carpetas de gestos
for label, gesture in enumerate(gestures):
    gesture_dir = os.path.join(data_dir, gesture)
    
    # Recorrer todas las imágenes en la carpeta del gesto
    for filename in os.listdir(gesture_dir):
        img_path = os.path.join(gesture_dir, filename)
        
        # Leer la imagen
        img = cv2.imread(img_path)
        
        # Redimensionar la imagen
        img_resized = cv2.resize(img, (64, 64))  # Redimensionar a 64x64 píxeles
        
        # Convertir la imagen a un array numpy y normalizarla (valores entre 0 y 1)
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Añadir la imagen a la lista de imágenes
        images.append(img_normalized)
        
        # Añadir la etiqueta correspondiente (es el índice del gesto)
        labels.append(label)

# Convertir las listas de imágenes y etiquetas a arrays numpy
images = np.array(images)
labels = np.array(labels)

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Guardar los datos procesados en archivos numpy para usar más tarde
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Preprocesamiento completo. Los datos están listos para entrenar el modelo.")


import cv2
import numpy as np
import os

# Aquí asumo que tienes la ruta a las imágenes
image_size = (64, 64)  # Tamaño de imagen deseado
gestures = ["ok", "pausa", "pulgar_arriba", "saludos", "adios", "yo", "gracias", "perdon", "no", "hola"]
data_dir = "data/raw"

data = []
labels = []

for gesture in gestures:
    gesture_path = os.path.join(data_dir, gesture)
    if not os.path.exists(gesture_path):
        continue

    for filename in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, filename)
        # Leer la imagen
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Redimensionar la imagen
        img_resized = cv2.resize(img, image_size)

        # Convertir a formato de arreglo
        data.append(img_resized)
        labels.append(gestures.index(gesture))

# Convertir a array de NumPy
data = np.array(data)
labels = np.array(labels)

# Verificar que los datos tengan la forma correcta
print("Datos procesados: ", data.shape)
print("Etiquetas procesadas: ", labels.shape)

# Guardar los datos procesados
np.save('data/processed_data.npy', data)
np.save('data/processed_labels.npy', labels)

print("Datos y etiquetas guardados exitosamente.")

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array

# Define los gestos
gestures = ["ok", "pausa", "pulgar_arriba", "saludos", "adios", "yo", "gracias", "perdon", "no", "uno", "dos", "tres", "cuatro", "cinco"]

# Directorio donde están las imágenes crudas
data_dir = "data/raw"

# Listas para las imágenes y etiquetas
images = []
labels = []

# Recorremos los gestos y cargamos las imágenes
for gesture in gestures:
    gesture_dir = os.path.join(data_dir, gesture)
    
    # Asegurarse de que la carpeta existe
    if not os.path.exists(gesture_dir):
        print(f"Advertencia: La carpeta {gesture_dir} no existe.")
        continue
    
    for img_name in os.listdir(gesture_dir):
        img_path = os.path.join(gesture_dir, img_name)
        
        # Cargar la imagen y convertirla en un array
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # Redimensionamos la imagen
        img = img_to_array(img)
        
        images.append(img)
        labels.append(gesture)

# Convertimos las imágenes y etiquetas a arrays de numpy
images = np.array(images, dtype="float32") / 255.0  # Normalización
labels = np.array(labels)

# Codificamos las etiquetas
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Guardamos los datos procesados
np.save("data/images.npy", images)
np.save("data/labels.npy", labels)
np.save("data/label_encoder.npy", label_encoder.classes_)

print("Datos procesados y guardados correctamente.")

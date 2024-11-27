import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('gesture_model.h5')

# Lista de gestos (debe ser la misma lista de gestos que usaste durante el entrenamiento)
gestures = ["ok", "pausa", "pulgar_arriba", "saludos", "adios", "yo", "gracias", "perdon", "no", "uno", "dos", "tres", "cuatro", "cinco"]

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Preprocesamiento de la imagen
    img_resized = cv2.resize(frame, (64, 64))  # Redimensiona la imagen a 64x64
    img_normalized = img_resized / 255.0  # Normaliza los valores de píxeles entre 0 y 1
    img_array = np.expand_dims(img_normalized, axis=0)  # Añade una dimensión adicional para el batch

    # Realizar la predicción
    predictions = model.predict(img_array)
    gesture_index = np.argmax(predictions)  # Obtiene el índice del gesto más probable
    predicted_gesture = gestures[gesture_index]

    # Mostrar el gesto predicho en la imagen (en letras rojas)
    cv2.putText(frame, predicted_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Texto en rojo

    # Mostrar la imagen con el gesto predicho
    cv2.imshow("Reconocimiento de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Salir con 'q'
        break

cap.release()
cv2.destroyAllWindows()

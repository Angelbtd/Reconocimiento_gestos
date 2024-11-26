import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Ruta del modelo entrenado
MODEL_PATH = 'saved_model/gesture_model.h5'

# Cargar el modelo
model = load_model(MODEL_PATH)
print("Modelo cargado exitosamente.")

# Lista de gestos entrenados (en el mismo orden que las etiquetas durante el entrenamiento)
gestures = ["ok", "pausa", "pulgar_arriba", "saludos", "adios", "yo", "gracias", "perdon", "no", "hola"]

# Función para predecir el gesto a partir de un frame
def get_prediction(frame):
    """
    Toma un frame, lo preprocesa y devuelve el gesto predicho y la confianza.
    """
    resized_frame = cv2.resize(frame, (64, 64))  # Redimensiona el frame a 64x64
    normalized_frame = resized_frame / 255.0    # Normaliza los valores entre 0 y 1
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Añade dimensión de batch
    predictions = model.predict(input_frame)    # Realiza predicción
    predicted_class = np.argmax(predictions)    # Obtiene la clase con mayor probabilidad
    confidence = predictions[0][predicted_class]  # Obtiene la confianza de la predicción
    return gestures[predicted_class], confidence

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # Obtener predicción del modelo
    gesture, confidence = get_prediction(frame)

    # Mostrar la predicción en el frame
    text = f"Gesto: {gesture} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame en la ventana
    cv2.imshow("Reconocimiento de Gestos", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()





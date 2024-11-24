import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('saved_model/gesture_model')
print("Modelo cargado exitosamente.")

# Diccionario de gestos
gestos = {
    0: "ok",
    1: "pausa",
    2: "pulgar_arriba",
    3: "saludos",
    4: "adios",
    5: "yo",
    6: "gracias",
    7: "perdon",
    8: "no",
    9: "hola"
}

# Configurar la captura de video
captura = cv2.VideoCapture(0)  # Usa 0 para la cámara principal

if not captura.isOpened():
    print("Error al abrir la cámara")
    exit()

print("Presiona 'q' para salir.")

while True:
    # Capturar frame de la cámara
    ret, frame = captura.read()
    if not ret:
        print("No se pudo capturar el frame")
        break

    # Redimensionar el frame al tamaño que espera el modelo (64x64)
    frame_procesado = cv2.resize(frame, (64, 64))  # Ajusta al tamaño de entrada
    frame_procesado = frame_procesado / 255.0      # Normalizar los valores de píxeles (0-1)
    frame_procesado = frame_procesado.reshape(1, 64, 64, 3)  # Añadir dimensión de batch

    # Predecir usando el modelo
    prediccion = model.predict(frame_procesado, verbose=0)  # Obtén las probabilidades
    clase_predicha = np.argmax(prediccion)  # Índice de la clase con mayor probabilidad

    # Obtener el nombre del gesto predicho
    texto_gesto = gestos.get(clase_predicha, "Desconocido")

    # Dibujar el texto en el frame
    cv2.putText(frame, f'Gesto: {texto_gesto}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame en una ventana
    cv2.imshow('Reconocimiento de Gestos', frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
captura.release()
cv2.destroyAllWindows()



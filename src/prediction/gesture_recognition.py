import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Cargar el modelo y el codificador de etiquetas
model = load_model("gesture_model.h5")
label_encoder = np.load("data/label_encoder.npy", allow_pickle=True)

# Definir los gestos
gestures = ["ok", "pausa", "pulgar_arriba", "saludos", "adios", "yo", "gracias", "perdon", "no", "uno", "dos", "tres", "cuatro", "cinco"]

# Inicia la cámara
cap = cv2.VideoCapture(0)

# Definir el tamaño de la ventana de la cámara
frame_width = 640
frame_height = 480
cap.set(3, frame_width)
cap.set(4, frame_height)

# Colores para la interfaz
highlight_color = (0, 0, 255)  # Color rojo para resaltar el gesto seleccionado
background_color = (255, 255, 255)  # Fondo blanco para el menú
font = cv2.FONT_HERSHEY_SIMPLEX  # Fuente amigable y moderna

# Crear un fondo para el menú
menu_width = 350  # Ancho del menú
frame_with_menu = np.zeros((frame_height, frame_width + menu_width, 3), dtype=np.uint8)

# Función para dibujar la interfaz
def draw_interface(frame, predicted_class):
    # Fondo del menú de gestos con cuadros blancos
    for i in range(0, len(gestures)):
        y_position = 80 + (i * 30)  # Ajuste para los gestos con espaciado más pequeño
        # Cuadro para cada gesto
        cv2.rectangle(frame, (frame_width + 10, y_position - 20), 
                      (frame_width + menu_width - 10, y_position + 10), background_color, -1)
        
        # Resaltar el gesto detectado con color diferente
        color = highlight_color if gestures[i] == predicted_class else (0, 0, 0)  # Rojo para el gesto detectado, negro para los demás
        # Resaltar con color más fuerte si el gesto es el detectado
        cv2.rectangle(frame, (frame_width + 10, y_position - 20), 
                      (frame_width + menu_width - 10, y_position + 10), color, -1)

        # Texto para cada gesto (ajustamos el tamaño de la fuente)
        cv2.putText(frame, gestures[i], (frame_width + 20, y_position), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Sección de gesto detectado arriba de la cámara
    label = f"Gesto Detectado: {predicted_class}"
    cv2.putText(frame, label, (10, 40), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    return frame

# Iniciar el ciclo de captura de video
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo acceder a la cámara.")
        break
    
    # Redimensionar la imagen y preprocesar
    frame_resized = cv2.resize(frame, (64, 64))
    frame_resized = frame_resized.astype("float32") / 255.0
    frame_resized = img_to_array(frame_resized)
    frame_resized = np.expand_dims(frame_resized, axis=0)
    
    # Realizar la predicción
    predictions = model.predict(frame_resized)
    predicted_class_index = np.argmax(predictions)
    predicted_class = gestures[predicted_class_index]
    
    # Crear la interfaz visual
    frame_with_menu[:, :frame_width] = frame  # Copiar la imagen de la cámara al lado izquierdo
    frame_with_menu = draw_interface(frame_with_menu, predicted_class)

    # Mostrar la imagen con el menú de gestos
    cv2.imshow("Reconocimiento de Gestos", frame_with_menu)

    # Salir si presionas la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()


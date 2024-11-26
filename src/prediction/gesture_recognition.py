import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model('models/gesture_recognition_model.keras')

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Iniciar la cámara
cap = cv2.VideoCapture(0)

gestures = ["ok", "pausa", "pulgar_arriba", "saludos", "adios", "yo", "gracias", "perdon", "no"]

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo acceder a la cámara.")
        break
    
    # Voltear la imagen para visualización
    frame = cv2.flip(frame, 1)
    
    # Convertir la imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen
    results = hands.process(rgb_frame)
    
    # Si se detecta una mano
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extraer las coordenadas de los puntos de las manos
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            # Normalizar las coordenadas de los puntos de la mano
            landmarks_array = landmarks_array.flatten()  # Flatten para convertirlo en un vector
            
            # Asegurarse de que la entrada tenga el tamaño correcto
            landmarks_array = np.expand_dims(landmarks_array, axis=0)  # Dimensiones (1, 63)
            
            # Redimensionar la imagen a 64x64 y convertirla en RGB
            image_resized = cv2.resize(frame, (64, 64))  # Redimensionar la imagen
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # Convertir a RGB
            
            # Normalizar la imagen a valores entre 0 y 1
            image_rgb = image_rgb / 255.0
            
            # Predecir el gesto
            predictions = model.predict(np.expand_dims(image_rgb, axis=0))
            print(predictions)  # Imprime las predicciones crudas
            
            # Obtener el gesto con la predicción más alta
            predicted_class = np.argmax(predictions, axis=1)[0]
            gesture_name = gestures[predicted_class]
            
            # Mostrar el gesto predicho en la pantalla
            cv2.putText(frame, f"Gesto: {gesture_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Mostrar la imagen con el gesto predicho
    cv2.imshow("Reconocimiento de Gestos", frame)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

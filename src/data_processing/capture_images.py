import cv2
import os

# Define las carpetas donde guardarás las imágenes de cada gesto
gestures = ["ok", "pause", "thumbs_up"]
data_dir = "data/raw"  # Ruta donde se guardarán las imágenes
img_size = (64, 64)  # Tamaño de las imágenes a guardar

# Crear las carpetas para cada gesto si no existen
for gesture in gestures:
    gesture_path = os.path.join(data_dir, gesture)
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

# Inicia la cámara
cap = cv2.VideoCapture(0)

# Instrucciones para el usuario
print("Presiona 'q' para salir")
print("Presiona la tecla correspondiente para capturar imágenes:")
print("1: ok")
print("2: pause")
print("3: thumbs_up")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Muestra la imagen en una ventana
    cv2.imshow("Captura de Gestos", frame)

    # Espera por una tecla presionada
    key = cv2.waitKey(1) & 0xFF

    # Detecta la tecla presionada y guarda la imagen correspondiente
    if key == ord('1'):
        gesture = "ok"
    elif key == ord('2'):
        gesture = "pause"
    elif key == ord('3'):
        gesture = "thumbs_up"
    elif key == ord('q'):
        break
    else:
        continue

    # Guarda la imagen en la carpeta correspondiente
    img_path = os.path.join(data_dir, gesture, f"{gesture}_{len(os.listdir(os.path.join(data_dir, gesture)))}.jpg")
    img_resized = cv2.resize(frame, img_size)  # Redimensiona la imagen
    cv2.imwrite(img_path, img_resized)

    print(f"Imagen de {gesture} guardada en {img_path}")

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()

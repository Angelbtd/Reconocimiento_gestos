import cv2
import os

# Define los gestos que vas a capturar
gestures = ["ok", "pausa", "pulgar_arriba", "saludos", "adios", "yo", "gracias", "perdon", "no"]

# Ruta donde se guardarán las imágenes
data_dir = "data/raw"  # Ruta para las imágenes crudas

# Asegurarse de que la ruta 'data/raw' y las subcarpetas de los gestos existan
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Crear las carpetas para cada gesto si no existen
for gesture in gestures:
    gesture_path = os.path.join(data_dir, gesture)
    # Se asegura de crear la carpeta si no existe, sin causar errores si ya existe
    os.makedirs(gesture_path, exist_ok=True)

# Inicia la cámara
cap = cv2.VideoCapture(0)

# Instrucciones para el usuario
print("Presiona 'q' para salir")
print("Presiona la tecla correspondiente para capturar imágenes:")
print("1: ok")
print("2: pausa")
print("3: pulgar_arriba")
print("4: saludos")
print("5: adios")
print("6: yo")
print("7: gracias")
print("8: perdon")
print("9: no")

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
        gesture = "pausa"
    elif key == ord('3'):
        gesture = "pulgar_arriba"
    elif key == ord('4'):
        gesture = "saludos"
    elif key == ord('5'):
        gesture = "adios"
    elif key == ord('6'):
        gesture = "yo"
    elif key == ord('7'):
        gesture = "gracias"
    elif key == ord('8'):
        gesture = "perdon"
    elif key == ord('9'):
        gesture = "no"
    elif key == ord('q'):  # Presiona 'q' para salir
        break
    else:
        continue

    # Guarda la imagen en la carpeta correspondiente
    img_path = os.path.join(data_dir, gesture, f"{gesture}_{len(os.listdir(os.path.join(data_dir, gesture)))}.jpg")
    img_resized = cv2.resize(frame, (64, 64))  # Redimensiona la imagen
    cv2.imwrite(img_path, img_resized)

    print(f"Imagen de {gesture} guardada en {img_path}")

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()


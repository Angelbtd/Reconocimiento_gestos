import os
import cv2
import numpy as np
from src.camera_feed import CameraFeed
from src.data_processing.preprocess import preprocess_image
from src.training.train_model import train_model

# Directorio donde se guardarán las imágenes
DATA_DIR = 'data/raw/'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Inicializar la cámara
camera = CameraFeed()
camera.start_feed(None)

def capture_images(class_name, num_images=100):
    """Captura imágenes desde la cámara y las guarda en el directorio de la clase especificada."""
    print(f"Capturando imágenes para la clase: {class_name}")
    
    # Crear carpeta para la clase si no existe
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    captured_images = 0

    while captured_images < num_images:
        frame = camera.get_frame()
        if frame is not None:
            # Mostrar la imagen en una ventana
            cv2.imshow("Captura de Imagen", frame)

            # Esperar por una tecla para capturar la imagen
            key = cv2.waitKey(1) & 0xFF

            # Capturar la imagen si presionas la tecla 'c'
            if key == ord('c'):
                # Preprocesar la imagen antes de guardarla
                img_name = f"{class_name}_{captured_images}.jpg"
                processed_img = preprocess_image(frame)  # Preprocesamiento de la imagen
                cv2.imwrite(os.path.join(class_dir, img_name), processed_img)
                print(f"Imagen {captured_images+1} guardada para la clase {class_name}")
                captured_images += 1

            # Salir del bucle si presionas 'q'
            if key == ord('q'):
                break

    # Detener la cámara y cerrar las ventanas
    camera.stop_feed()
    cv2.destroyAllWindows()

def main():
    """Función principal que inicia la captura de imágenes y entrena el modelo."""
    # Pedir la clase del gesto
    class_name = input("Introduce el nombre de la clase de gesto (ej. 'pulgar_arriba'): ")
    
    # Captura de imágenes
    capture_images(class_name)

    # Ahora que tenemos imágenes, podemos preprocesarlas y entrenar el modelo
    print("Entrenando el modelo...")
    train_model()

if __name__ == '__main__':
    main()

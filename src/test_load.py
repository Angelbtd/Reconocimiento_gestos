import numpy as np

# Cargar los archivos
try:
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    # Verificar las formas de los arrays
    print("Shape de X_train:", X_train.shape)
    print("Shape de X_test:", X_test.shape)
    print("Shape de y_train:", y_train.shape)
    print("Shape de y_test:", y_test.shape)
except Exception as e:
    print("Error al cargar los archivos:", e)

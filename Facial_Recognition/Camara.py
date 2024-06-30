import cv2
import os

# Inicializa la cámara (0 es el ID de la cámara por defecto)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Define la ruta de la carpeta donde se guardará la imagen
directorio_fotos = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fotos')

# Verifica si la carpeta existe, si no, la crea
if not os.path.exists(directorio_fotos):
    os.makedirs(directorio_fotos)

while True:
    # Lee un frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se pudo capturar la imagen.")
        break

    # Muestra el frame capturado en una ventana
    cv2.imshow('Captura', frame)

    # Espera a que se presione una tecla
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('p'):
        # Define la ruta completa del archivo donde se guardará la imagen
        ruta_imagen = os.path.join(directorio_fotos, 'Alvaro.png')
        
        # Guarda la imagen en el archivo
        cv2.imwrite(ruta_imagen, frame)
        print(f"Imagen guardada como {ruta_imagen}")
        break
    
    elif key == ord('q'):
        # Sale del bucle y termina el programa
        print("Saliendo del programa...")
        break

# Libera el objeto de captura y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
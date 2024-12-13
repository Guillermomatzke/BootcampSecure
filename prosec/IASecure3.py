import cv2
import requests  # Biblioteca para solicitudes HTTP
from ultralytics import YOLO
import time  # Para limitar la tasa de detección

# Cargar el modelo entrenado
model = YOLO("runs/detect/train4/weights/best.pt")

# Etiquetas que deseas ignorar
exclude_labels = ["Bombona", "Brazo Soldando", "Inclinacion"]

# Etiquetas para activar 'apagar'
alert_labels = ["No Casco", "No Delantal", "No Guantes"]

# Iniciar la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reducir resolución
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

last_detection_time = 0  # Tiempo de la última detección
detection_interval = 0.5  # Intervalo entre detecciones en segundos

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Invertir la imagen horizontalmente
    frame = cv2.flip(frame, 1)

    # Procesar detecciones solo si ha pasado el intervalo
    current_time = time.time()
    if current_time - last_detection_time >= detection_interval:
        last_detection_time = current_time

        # Reducir el tamaño del frame para el procesamiento
        small_frame = cv2.resize(frame, (320, 180))  # Escalar para detecciones más rápidas

        # Realizar detecciones
        results = model(small_frame)
        detections = results[0]

        # Filtrar etiquetas no deseadas
        alert_detected = False
        for box in detections.boxes:
            class_id = int(box.cls[0])  # ID de la clase detectada
            class_name = model.names[class_id]  # Nombre de la clase detectada
            if class_name in alert_labels:
                alert_detected = True
                break

        # Enviar solicitud HTTP dependiendo de las etiquetas detectadas
        try:
            if alert_detected:
                requests.get("http://10.10.2.66/apagar")  # Solicitud para apagar
                print("Etiqueta de alerta detectada. Enviando solicitud para apagar.")
            else:
                requests.get("http://10.10.2.66/encender")  # Solicitud para encender
                print("Sin etiquetas de alerta. Enviando solicitud para encender.")
        except requests.exceptions.RequestException as e:
            print(f"Error al enviar la solicitud HTTP: {e}")

    # Mostrar el frame original (sin procesar detecciones en todos los frames)
    cv2.imshow("Detecciones YOLOv8", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        requests.get("http://10.10.2.66/encender")  # Solicitud para encender
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO("runs/detect/train4/weights/best.pt")  # Reemplaza "best.pt" con la ruta a tu modelo
# Etiquetas que deseas ignorar
exclude_labels = ["Bombona", "Brazo Soldando", "Inclinacion"]  # Cambia las etiquetas que quieres excluir


# Iniciar la cámara
cap = cv2.VideoCapture(0)  # 0 representa la cámara por defecto de tu notebook
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)


while True:
    ret, frame = cap.read()  # Leer un frame de la cámara
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Invertir la imagen horizontalmente
    frame = cv2.flip(frame, 1)

    # Realizar detecciones
    results = model(frame)
    detections = results[0]

    # Filtrar etiquetas no deseadas
    filtered_boxes = []
    for box in detections.boxes:
        class_id = int(box.cls[0])  # ID de la clase detectada
        class_name = model.names[class_id]  # Nombre de la clase detectada
        if class_name not in exclude_labels:
            filtered_boxes.append(box)

    # Crear una nueva imagen con solo las detecciones deseadas
    detections.boxes = filtered_boxes
    annotated_frame = detections.plot()

    # Mostrar el frame con las detecciones
    cv2.imshow("Detecciones YOLOv8", annotated_frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

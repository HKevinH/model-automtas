from ultralytics import YOLO
import cv2

def main():
    model_path = "best.pt"
    model = YOLO(model_path)

    class_names = model.names
    confidence_threshold = 0.5
    print(f"Clases detectadas: {class_names}")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    cap_count = 0
    glasses_count = 0

    print("Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame. Saliendo...")
            break

        results = model(frame, conf=confidence_threshold)

        frame_cap_count = 0
        frame_glasses_count = 0

        for r in results[0].boxes.data:
            x1, y1, x2, y2 = map(int, r[:4])
            confidence = float(r[4])
            label_index = int(r[5])

            label_name = class_names[label_index] if label_index < len(class_names) else "Unknown"

            if label_name == "cap":
                cap_count += 1
                frame_cap_count += 1
            elif label_name == "glasses":
                glasses_count += 1
                frame_glasses_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, f"Personas con gorra (frame): {frame_cap_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Personas con gafas (frame): {frame_glasses_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Conteo total de personas con gorra: {cap_count}")
    print(f"Conteo total de personas con gafas: {glasses_count}")

if __name__ == "__main__":
    main()

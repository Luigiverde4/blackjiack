from ultralytics import YOLO
import cv2

# === COLORES ===
COLOR_DEALER = (255, 0, 0)  # azul
COLOR_JUGADOR = (0, 0, 255)  # rojo
COLOR_CAJA = (0, 255, 0)  # verde

def draw_zones(frame, alpha=0.3):
    # dibuja zonas izquierda (dealer) y derecha (jugador) con transparencia
    overlay = frame.copy()
    height, width = frame.shape[:2]

    # zona izquierda -> DEALER
    cv2.rectangle(overlay, (0, 0), (width // 2, height), COLOR_DEALER, -1)
    # zona derecha -> JUGADOR
    cv2.rectangle(overlay, (width // 2, 0), (width, height), COLOR_JUGADOR, -1)

    # aplicar mezcla de transparencia
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # texto para marcar zonas
    cv2.putText(frame, "DEALER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_DEALER, 2)
    cv2.putText(frame, "JUGADOR", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_JUGADOR, 2)

def recorta16_9(frame):
    """
    Recorta el frame centrado a proporcion 16:9.

    frame (np.ndarray): imagen de entrada (H x W x C)
    """
    # obtener dimensiones actuales del frame
    height, width = frame.shape[:2]

    # calcular altura objetivo manteniendo ancho original
    target_width = width
    target_height = int(width * 9 / 16)

    # si no cabe, recalcular usando la altura como base
    if target_height > height:
        target_height = height
        target_width = int(height * 16 / 9)

    # calcular coordenadas de inicio para recorte centrado
    y_start = (height - target_height) // 2
    x_start = (width - target_width) // 2

    # devolver recorte centrado
    return frame[y_start:y_start+target_height, x_start:x_start+target_width]


def main():
    # cargar modelo entrenado
    model = YOLO("runs/detect/train18/weights/best.pt")

    # abrir webcam
    cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


    if not cap.isOpened():
        print("No se pudo abrir la camara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = recorta16_9(frame)  # Forzamos relación 16:9

        height, width = frame.shape[:2]
        # dibujar zonas (izq dealer, der jugador)
        draw_zones(frame)

        # detectar cartas en la imagen
        results = model.predict(source=frame, conf=0.5, verbose=False)[0]

        # listas para guardar cartas detectadas
        dealer_hand = []
        player_hand = []

        for box in results.boxes:
            # obtener coords de la caja
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            # usar centro de la caja para decidir a quien pertenece
            center_x = (x1 + x2) // 2
            owner = "DEALER" if center_x < width // 2 else "JUGADOR"

            # nombre de la carta
            card_name = model.names[class_id]

            # guardar en la mano correspondiente
            (dealer_hand if owner == "DEALER" else player_hand).append(card_name)

            # dibujar caja y texto
            label = f"{owner} {card_name}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CAJA, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAJA, 2)

        # mostrar manos por consola
        if dealer_hand:
            print("🂠 Dealer:", dealer_hand)
        if player_hand:
            print("🂡 Jugador:", player_hand)

        # mostrar ventana
        cv2.imshow("Deteccion de cartas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

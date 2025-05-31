from ultralytics import YOLO
import cv2

# === COLORES PARA VISUALIZACIÓN ===
COLOR_DEALER  = (255, 0, 0)   # azul: zona del dealer
COLOR_JUGADOR = (0, 0, 255)   # rojo: zona del jugador
COLOR_CAJA    = (0, 255, 0)   # verde: bounding box de la carta detectada


def draw_zones(frame, alpha=0.2):
    """
    Dibuja dos zonas en el frame: arriba para el dealer y abajo para el jugador.
    Aplica una superposición semitransparente y etiqueta ambas zonas.

    Parámetros:
    - frame: imagen de entrada (frame de la cámara)
    - alpha: opacidad de la superposición de color
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Zona superior para el dealer
    cv2.rectangle(overlay, (0, 0), (w, h//2), COLOR_DEALER, -1)
    # Zona inferior para el jugador
    cv2.rectangle(overlay, (0, h//2), (w, h), COLOR_JUGADOR, -1)

    # Mezclar overlay con el frame original
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Etiquetas de zona
    cv2.putText(frame, "DEALER",  (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_DEALER, 2)
    cv2.putText(frame, "JUGADOR", (10, h//2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_JUGADOR, 2)

def find_cards_roi(zone, max_cards=7):
    """
    Busca los contornos más grandes en una zona de la imagen, suponiendo que representan cartas.
    Devuelve una lista con hasta `max_cards` bounding boxes (x, y, w, h), ordenadas por área descendente.

    Parámetros:
    - zone: imagen recortada (dealer o jugador)
    - max_cards: número máximo de contornos/cajas a devolver (por defecto: 7)

    Retorna:
    - Lista de bounding boxes [(x, y, w, h), ...] o lista vacía si no hay contornos válidos
    """
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return []

    bboxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area >= 0.05 * zone.shape[0] * zone.shape[1]:  # filtro de tamaño
            bboxes.append((x, y, w, h, area))

    # Ordenar por área descendente y quedarnos con los max_cards más grandes
    bboxes = sorted(bboxes, key=lambda b: b[4], reverse=True)[:max_cards]

    return [(x, y, w, h) for (x, y, w, h, _) in bboxes]

def detect_in_zone(model, frame_orig, x0, y0, w0, h0, conf):
    """
    Recorta una zona del frame original (dealer o jugador),
    localiza hasta 7 cartas por contornos grandes, hace zoom en cada una,
    y ejecuta detección con YOLOv8.

    Parámetros:
    - model: modelo YOLOv8 cargado
    - frame_orig: imagen completa
    - x0, y0: coordenadas de inicio del recorte
    - w0, h0: dimensiones del recorte
    - conf: umbral de confianza mínimo para la detección

    Retorna:
    - Lista de detecciones [(nombre_clase, x1, y1, x2, y2), ...]
    """
    zone = frame_orig[y0:y0+h0, x0:x0+w0]  # recorte de la zona (superior o inferior)
    roi_boxes = find_cards_roi(zone)  # buscamos hasta 7 contornos grandes

    if not roi_boxes:
        return []

    detections = []
    for roi in roi_boxes:
        x, y, w, h = roi
        card = zone[y:y+h, x:x+w]

        # Escalar la carta individual a tamaño completo del frame
        zoomed = cv2.resize(card, (frame_orig.shape[1], frame_orig.shape[0]),
                            interpolation=cv2.INTER_LINEAR)

        results = model.predict(source=zoomed, conf=conf, verbose=False)[0]

        for box in results.boxes:
            x1z, y1z, x2z, y2z = map(int, box.xyxy[0])

            # Convertimos del frame escalado al ROI original
            nx1 = x + x1z * w / frame_orig.shape[1]
            nx2 = x + x2z * w / frame_orig.shape[1]
            ny1 = y + y1z * h / frame_orig.shape[0]
            ny2 = y + y2z * h / frame_orig.shape[0]

            # Ajustamos a coordenadas absolutas dentro del frame completo
            X1 = int(x0 + nx1)
            X2 = int(x0 + nx2)
            Y1 = int(y0 + ny1)
            Y2 = int(y0 + ny2)

            name = model.names[int(box.cls[0])]
            detections.append((name, X1, Y1, X2, Y2))

    return detections

def debug_find_card_roi(zone):
    """
    Muestra visualmente los pasos intermedios de find_card_roi.
    Útil para entender qué contornos se detectan y qué ROI se selecciona.
    """
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = zone.copy()
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)  # dibuja todos los contornos

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > 0.05 * zone.shape[0] * zone.shape[1]:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, "ROI", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar ventanas (puedes cerrar con 'q')
    cv2.imshow("Gray", gray)
    cv2.imshow("Edges", edges)
    cv2.imshow("Contornos y ROI", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Función principal que ejecuta el flujo de detección en tiempo real:
    - Carga el modelo YOLO
    - Abre la cámara
    - Dibuja zonas de dealer y jugador
    - Detecta cartas en ambas zonas
    - Muestra resultados en pantalla
    """
    model = YOLO("runs/detect/train33/weights/best.pt")  # cargar modelo entrenado

    # Configurar captura de vídeo
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    # Analizamos frames
    while True:
        ret, frame = cap.read() # Frame es lo que vamos a visualizar en pantalla con texto y tal
        if not ret:
            break
        
        # Datos
        h, w = frame.shape[:2]
        frame_clean = frame.copy() # Copia limpia del frame

        # Dibujajmos sobre el frame que vamos a mostrar
        draw_zones(frame)

        # Detecta cartas en la zona superior (dealer), de (0, 0) a (w, h//2), con confianza ≥ 0.3
        dealer_dets = detect_in_zone(model, frame, 0, 0, w, h//2, conf=0.3)
        # Detectar carta en la zona inferior (jugador) de (0, h//2) a (w, h), con confianza ≥ 0.3
        player_dets = detect_in_zone(model, frame, 0, h//2, w, h//2, conf=0.3)

        
        # Dibujar resultados en la imagen
        
        # Resultados DEALER
        for name, x1, y1, x2, y2 in dealer_dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CAJA, 2) # Caja de deteccion
            cv2.putText(frame, name, (x1, y1-10),                   # Clase identificada
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DEALER, 2)
        
        # Resultados JUGADOR
        for name, x1, y1, x2, y2 in player_dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CAJA, 2) # Caja de deteccion
            cv2.putText(frame, name, (x1, y1-10),                   # Clase identificada
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_JUGADOR, 2)
        
        # Mostrar nombres de cartas detectadas 
        dealer_cards = [d[0] for d in dealer_dets]
        player_cards = [d[0] for d in player_dets]

        cv2.putText(frame, "Dealer: " + ", ".join(dealer_cards),
                    (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DEALER, 2)
        cv2.putText(frame, "Jugador: " + ", ".join(player_cards),
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_JUGADOR, 2)

        # Mostrar ventana
        cv2.imshow("Deteccion de cartas", frame)
        key = cv2.waitKey(1) & 0xFF 
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_find_card_roi(frame_clean[0:h//2, 0:w])     # dealer (mitad de arriba)
            debug_find_card_roi(frame_clean[h//2:h, 0:w])     # jugador (mitad de abajo)

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

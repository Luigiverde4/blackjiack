from ultralytics import YOLO
import cv2
import os
from mejor_jugada import *

# === COLORES PARA VISUALIZACIÓN ===
COLOR_DEALER  = (255, 0, 0)   # azul: zona del dealer
COLOR_JUGADOR = (0, 0, 255)   # rojo: zona del jugador
COLOR_CAJA    = (0, 255, 0)   # verde: bounding box de la carta detectada

# CHRISTIAN
def cargar_imagenes_cartas(ruta="Minicartas", tamaño=(80, 120)):
    """
    Carga las imagenes que se muestran como iconos al detectar una carta
    Cargarlas ahora ahorra espacio y tiempo luego

    Parámetros:
        - ruta: carpeta a leer
        - tamaño: tamaño de la imagen

    Devuele:
        imagenes: Diccionario con las imagenes atribuidas a cada clase
    """
    
    imagenes = {}
    # Pasa por la carpeta indicada para leer todas las imagenes de las cartas
    for archivo in os.listdir(ruta):
        nombre = os.path.splitext(archivo)[0]
        path = os.path.join(ruta, archivo)
        imagen = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # Nombre el mismo que la clase (Y la carta), path la ruta a esta para que lo lea cv2
        
        if imagen is not None:
            imagen = cv2.resize(imagen, tamaño) #Cambia el tamaño de la imagen
            imagenes[nombre] = imagen
            # Lo guarda en un diccionario con el estilo de {"Ace of Clubs": imagen, "Ace of Hearts": imagen, ...}
    return imagenes

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

def dibujar_cartas_en_esquina(frame, cartas, imagenes_cartas, side = "player", margen=10):
    """
    Dibuja miniaturas de cartas detectadas en la esquina inferior si es del jugador y superior si es del dealer
    """
    
    h_frame, w_frame = frame.shape[:2]
    x = w_frame - margen
    y = margen if side == "dealer" else h_frame - margen # Colocamos la posición de las primeras cartas
    # Como esta función se ejecuta 2 veces (Una por lado de la mesa), no hay problemas

    for nombre_carta in cartas:
        img_carta = imagenes_cartas.get(nombre_carta)
        if img_carta is None:
            continue  # saltar si no se encuentra imagen sobre dicha carta, cosa que no debería de pasar

        h, w = img_carta.shape[:2]

        # calcular posicion destino
        x_offset = x - w
        y_offset = y if side == "dealer" else y - h

        # agregar imagen
        if img_carta.shape[2] == 4:  # si tiene canal alfa
            alpha = img_carta[:, :, 3] / 255.0
            for c in range(3):
                frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = ( # Colocamos los valores de la imagen en el frame
                alpha * img_carta[:, :, c] + (1 - alpha) * frame[y_offset:y_offset+h, x_offset:x_offset+w, c])
        else:
            frame[y_offset:y_offset+h, x_offset:x_offset+w] = img_carta

        # ajustar siguiente posición
        x -= w + margen


def iou(box1, box2):
    """Calcula el IoU entre dos cajas (x1, y1, x2, y2)"""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def eliminar_detecciones_duplicadas(detections, umbral_iou=0.5):
    """Filtra detecciones solapadas (misma carta física)"""
    final = []
    for det in detections:
        name, x1, y1, x2, y2 = det
        box = (x1, y1, x2, y2)
        if not any(iou(box, (d[1], d[2], d[3], d[4])) > umbral_iou for d in final):
            final.append(det)
    return final

def interfaz(player_cards,dealer_card,cards_dict,frame,w,h,val):
    
    cv2.putText(frame, "Dealer: " + ", ".join(dealer_card),
                    (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DEALER, 2)
    cv2.putText(frame, "Jugador: " + ", ".join(player_cards),
                (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_JUGADOR, 2)
    cv2.putText(frame, "Introduce tu apuesta:{0}".format(val) , 
                    ((w//2)-100 , (h//2)-325), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    
    if len(player_cards) >= 2 and len(dealer_card) >=1:
        jugada, num_jugador, num_dealer = mejor_jugada(player_cards,dealer_card)

        cv2.putText(frame, str(np.sum(num_jugador)) , 
            ((w//2) , (h//2)+50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_JUGADOR, 2)
        
        cv2.putText(frame, str(np.sum(num_dealer)) , 
            ((w//2) , (h//2)-50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_DEALER, 2)

        if jugada == "Q":
            cv2.putText(frame, "Quedarse" , 
                    ((w//2) , h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        elif jugada == "P":
            cv2.putText(frame, "Pedir" , 
                    ((w//2) , h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        elif jugada == "D":
            cv2.putText(frame, "Pedir y doblar apuesta" , 
                    ((w//2), h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        else:
            cv2.putText(frame, "Has perdido" , 
                    ((w//2) , h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            val = ""
    else:
        cv2.putText(frame, "Esperando a que se pongan cartas" , 
                    ((w//2)-200 , h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    # mostrar miniaturas en esquinas
    dibujar_cartas_en_esquina(frame, dealer_card, cards_dict, side = "dealer")
    dibujar_cartas_en_esquina(frame, player_cards, cards_dict, side = "player")

    # Mostrar ventana
    cv2.imshow("Deteccion de cartas", frame)

def main():
    """
    Función principal que ejecuta el flujo de detección en tiempo real:
    - Carga el modelo YOLO
    - Abre la cámara
    - Dibuja zonas de dealer y jugador
    - Detecta cartas en ambas zonas
    - Muestra resultados en pantalla
    """
    cards_dict = cargar_imagenes_cartas("Minicartas", (40, 60))  # Cargamos las imagenes de icono | CHRISTIAN
    model = YOLO("best33.pt")  # cargar modelo entrenado
    val = ""

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

        
        dealer_dets = eliminar_detecciones_duplicadas(dealer_dets)
        player_dets = eliminar_detecciones_duplicadas(player_dets)


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


        key = cv2.waitKey(1) & 0xFF 
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_find_card_roi(frame_clean[0:h//2, 0:w])     # dealer (mitad de arriba)
            debug_find_card_roi(frame_clean[h//2:h, 0:w])
        elif key != ord("f") and chr(key).isdigit(): # jugador (mitad de abajo)
            val+=chr(key)

        interfaz(player_cards,dealer_cards,cards_dict,frame,w,h,val)
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

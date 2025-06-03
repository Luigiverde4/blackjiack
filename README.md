# 🃏 BlackjIAck

Proyecto final de la asignatura **Imagen y Vídeo Digital**  
*Autores: Christian Bertomeu Rodríguez, Jesús Navarro Prieto, Ricardo Román Martínez*

---

## 🎯 Objetivos

1. Detectar cartas de poker en imágenes/vídeo.
2. Reconocer jugadas en tiempo real en partidas de Blackjack.
3. Recomendar la mejor jugada (`Pedir`, `Quedarse`, `Doblar`) según la estrategia óptima del juego.

---

## 📦 Dataset

Dataset: [(Link)](https://www.kaggle.com/datasets/hosseinah1/poker-game-dataset)  
- 7624 imágenes de entrenamiento  
- 265 de validación  
- 53 clases distintas  
- Formato: JPG, 224x224x3  

Para adaptarlo a YOLOv8, se usó un script personalizado de conversión al formato YOLO (imágenes + archivos `.txt` con etiquetas).

---

## 🧠 Modelo

Se utilizó **YOLOv8s** por su buen balance entre velocidad y precisión.  
Entrenamiento final:

Train 33:
- Modelo: `yolov8s.pt`
- Resolución: `640x640`
- Epochs: `20`
- Batch size: `16`
- Aumentos de datos

---

## 🔧 Data Augmentation

Probamos a usar data augmentation mejorar la detección de cartas pequeñas o solapadas:
- Escalado y rotación aleatoria de cartas.
- Inserción sobre fondos con textura/ruido.
- Distribución aleatoria en zonas de dealer y jugador.
- Se generaron nuevos batches con este método.

Finalmente no se utilizó debido a sus malos resultados

---

## ⚙️ Funcionamiento del sistema

1. **Captura de vídeo** desde la webcam (OpenCV).
2. **División en zonas**:
   - Parte superior: cartas del dealer.
   - Parte inferior: cartas del jugador.
3. **Detección de contornos** con Canny y filtrado por área para localizar posibles cartas.
4. **Zoom a cada ROI** y clasificación con YOLOv8.
5. **Conversión de etiquetas a valores enteros** y paso a la lógica del Blackjack.
6. **Determinación de jugada óptima**.

---

## 💡 Lógica de decisión

El archivo `mejor_jugada.py` implementa una estrategia óptima en base a:
- Tipos de mano: `dura`, `blanda` o `pares`.
- Tabla de decisiones basada en el valor del dealer.

Valores:
- `P`: Pedir
- `Q`: Quedarse
- `D`: Doblar
---

## 📂 Descripción de archivos

| Archivo                   | Descripción |
|--------------------------|-------------|
| `play.py`                | Script principal que lanza la detección de cartas en tiempo real desde la webcam, visualiza resultados y calcula la mejor jugada con interfaz gráfica. |
| `eval.py`                | Evalúa el modelo YOLOv8 entrenado sobre el conjunto de validación. Muestra métricas como precisión, recall y mAP. |
| `mejor_jugada.py`        | Contiene la lógica de decisión basada en estrategia óptima de Blackjack. Convierte las cartas detectadas a valores numéricos y devuelve la mejor acción recomendada. |
| `train.ipynb`            | Cuaderno Jupyter utilizado para entrenar los modelos YOLOv8 con distintas configuraciones y datasets aumentados. |
| `convert_to_yolo.ipynb`  | Cuaderno para convertir el dataset original de Kaggle al formato compatible con YOLOv8 (imágenes + etiquetas). |
| `cards.yaml`             | Configuración del dataset: nombres de clases y rutas a los conjuntos de entrenamiento/validación. |
| `kaggle.json`            | Credenciales necesarias para descargar el dataset desde Kaggle. |
| `runs/`                  | Carpeta generada automáticamente por YOLO que contiene modelos entrenados y resultados de validación. |


### 📚 Librerías necesarias (Python 3.8+)
Para reproducir correctamente este proyecto, asegúrate de tener instalado lo siguiente:

```bash
pip install opencv-python ultralytics numpy
```
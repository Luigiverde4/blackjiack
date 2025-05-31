# 🃏 Blackjack Card Detector

Este proyecto utiliza **YOLOv8** junto con procesamiento de imagen para detectar cartas de Blackjack en vídeo en tiempo real (por ejemplo, desde una webcam), identificar las cartas visibles del **dealer** (zona superior) y del **jugador** (zona inferior), y sugerir automáticamente la **mejor jugada** según la estrategia básica del Blackjack.

---

## 🎯 Objetivo

Detectar las cartas visibles en una partida de Blackjack y ofrecer al jugador una recomendación óptima sobre qué acción tomar (`Pedir`, `Quedarse`, `Doblar`, `Separar`), combinando visión por computador con reglas expertas del juego.

---

## 🧠 Cómo funciona

El sistema sigue el siguiente flujo:

1. **Captura de vídeo** con OpenCV.
2. **Delimitación de zonas**:
   - Zona superior: cartas del dealer.
   - Zona inferior: cartas del jugador.
3. **Localización de cartas** mediante análisis de contornos:
   - Aplicamos una **conversión a escala de grises**, seguida de **detección de bordes con el operador Canny**.
   - Se seleccionan los contornos más grandes como posibles cartas usando un filtro de área.
4. **Redimensionado y clasificación**:
   - Cada carta localizada se **reescala** y se pasa por un modelo **YOLOv8** entrenado para identificar su valor.
5. **Evaluación de la jugada**:
   - Las cartas detectadas se pasan a un sistema de reglas que determina la **mejor jugada posible** en función de la estrategia óptima de Blackjack.

---

## 📁 Estructura del proyecto

```
.
├── cards.yaml             # Configuración del dataset
├── kaggle.json            # Credenciales para acceder al dataset en Kaggle
├── play.py                # Detección en tiempo real desde webcam
├── eval.py                # Evaluación del modelo entrenado
├── mejor_jugada.py        # Lógica de estrategia óptima de Blackjack
├── convert_to_yolo.ipynb  # Conversión de anotaciones a formato YOLO
├── negrojack.ipynb        # Cuaderno de desarrollo y pruebas
└── runs/                  # Directorio con los modelos entrenados
```

---

## 🛠️ Requisitos

- Python 3.8+
- OpenCV
- Ultralytics
- NumPy

Instalación rápida:

```bash
pip install -r requirements.txt
```

---

## 🚀 Cómo usar

### Detección en vivo
```bash
python play.py
```
- Inicia la webcam y muestra detecciones en tiempo real.
- Pulsa `q` para salir, `d` para ver el debug visual de contornos.

### Evaluar el modelo
```bash
python eval.py
```
- Muestra métricas como mAP50, precisión y recall sobre el conjunto de validación.

---

## 🧠 Lógica de decisión

La estrategia se basa en un diccionario experto que clasifica la mano en:

- `dura`: sin As o As contando como 1.
- `blanda`: con As que puede contar como 11.
- `pares`: dos cartas iguales.

Y sugiere una jugada:

- `P` = Pedir
- `Q` = Quedarse
- `D` = Doblar
- `S` = Separar

---

## 📦 Dataset

Dataset: [cards-image-datasetclassification](https://www.kaggle.com/datasets/jezuzo/cards-image-datasetclassification)  
Incluye imágenes individuales de cartas para entrenar el modelo de detección.

---
# train_cards.py

from ultralytics import YOLO
import torch


def main():
    # Verificar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🧠 PyTorch:", torch.__version__)
    print("🖥️ CUDA disponible:", torch.cuda.is_available())

    if device == "cuda":
        print("✅ GPU detectada:", torch.cuda.get_device_name(0))
        print("🧮 Memoria total:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")

    print(f"✅ Entrenando en: {device}")

    # Cargar modelo YOLOv8
    model = YOLO("yolov8n.pt")  # Puedes cambiar a yolov8s.pt o similar

    # Entrenar modelo
    model.train(
        data="cards.yaml",
        epochs=50,
        imgsz=640,
        device=device
    )
if __name__ == "__main__":
    main()

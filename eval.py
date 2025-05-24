from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train18/weights/best.pt")
    metrics = model.val(data="cards.yaml")

    print("\n📊 Mostrando metricas del modelo sobre el conjunto de validacion:")
    print("- mAP50       -> media de aciertos con IoU >= 0.50")
    print("- mAP50-95    -> media general de aciertos (IoU de 0.50 a 0.95)")
    print("- Precision   -> proporcion de detecciones correctas")
    print("- Recall      -> proporcion de objetos detectados sobre el total real\n")

    print("✅ Evaluacion completada")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision media (mp): {metrics.box.mp:.4f}")
    print(f"Recall media (mr): {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()


import os
import shutil
from pathlib import Path

from PIL import Image

# Configuración
input_dir = "poker"  # Carpeta original con subcarpetas por clase
output_dir = "cards"   # Carpeta destino con estructura YOLOcd 
sets = ["train", "valid"]  # Conjuntos disponibles

# Obtener clases ordenadas
class_names = sorted(os.listdir(os.path.join(input_dir, "train")))
class_to_id = {name: idx for idx, name in enumerate(class_names)}

# Crear carpetas de salida
for subset in sets:
    os.makedirs(f"{output_dir}/images/{subset}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{subset}", exist_ok=True)

    for class_name in class_names:
        src_dir = os.path.join(input_dir, subset, class_name)
        if not os.path.exists(src_dir):
            continue

        for file in os.listdir(src_dir):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(src_dir, file)
            dest_img_path = os.path.join(output_dir, "images", subset, f"{class_name}_{file}")
            label_path = os.path.join(output_dir, "labels", subset, f"{class_name}_{Path(file).stem}.txt")

            # Copiar imagen
            shutil.copy(img_path, dest_img_path)

            # Leer tamaño de imagen
            with Image.open(img_path) as img:
                w, h = img.size

            # Anotación YOLO (caja que cubre toda la imagen)
            # Formato: class_id center_x center_y width height (en proporción)
            yolo_line = f"{class_to_id[class_name]} 0.5 0.5 1.0 1.0\n"

            with open(label_path, "w") as f:
                f.write(yolo_line)

print("✅ Conversión completada en formato YOLO.")

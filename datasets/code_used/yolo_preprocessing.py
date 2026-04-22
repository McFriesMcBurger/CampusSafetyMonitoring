import os
import shutil
import xml.etree.ElementTree as ET

IMAGE_DIR = os.path.join("/home/kjel/Downloads/AI Project/combined/images")
LABEL_DIR = os.path.join("/home/kjel/Downloads/AI Project/combined/labels")

OUTPUT_IMAGES = "/home/kjel/Downloads/AI Project/Poly/train/images"
OUTPUT_LABELS = "/home/kjel/Downloads/AI Project/Poly/train/labels"

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# One of the datasets used was labeled in spanish
class_map = {

    # Fire extinguisher
    "Fire_Extinguisher": 0,
    "Extintor": 0,

    # Fire blanket
    "Fire_Blanket": 1,

    # Alarm activators
    "Alarm_Activator": 2,
    "Pulsador de alarma": 2,

    # Alarms
    "Sounders": 3,
    "Flashing_Light_Orbs": 3,

    # Smoke Detector
    "White_Domes": 4,

    # Defibrilators
    "Desfibrilador": 5,
    "AED": 5,

    # Signs labeling
    "Fire_Suppression_Signage": 6,
    "Fire_Exit": 6,
    "Salida de emergencia": 6,
    "Salida": 6,
    "Prohibido fumar": 6,
    "Alto voltaje": 6,
    "Uso obligatorio de mascarilla": 6,
    "Zona videovigilada": 6,
    "Acceso Restringido": 6,
    "Baños": 6,
    "wet_floor_sign": 6

}


# Bounding box standardization
def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]
    h = box[3] - box[2]

    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh

    return x_center, y_center, w, h

# .xml file standardization
def process_xml(xml_path, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    lines = []

    for obj in root.findall("object"):
        name = obj.find("name").text

        if name not in class_map:
            continue

        class_id = class_map[name]

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        xmax = float(bnd.find("xmax").text)
        ymin = float(bnd.find("ymin").text)
        ymax = float(bnd.find("ymax").text)

        x, y, bw, bh = convert_bbox((w, h), (xmin, xmax, ymin, ymax))

        lines.append(f"{class_id} {x} {y} {bw} {bh}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

# .txt file standardization
def process_yolo(txt_path, output_path):
    lines_out = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            cls = int(parts[0])

            # already numeric → keep (or remap if needed later)
            new_cls = cls

            lines_out.append(f"{new_cls} {' '.join(parts[1:])}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines_out))



for file in os.listdir(LABEL_DIR):

    label_path = os.path.join(LABEL_DIR, file)


    if file.endswith(".xml"):
        out_name = file.replace(".xml", ".txt")

        process_xml(
            label_path,
            os.path.join(OUTPUT_LABELS, out_name)
        )

        # match image (assume jpg/png)
        img_name_jpg = file.replace(".xml", ".jpg")
        img_name_png = file.replace(".xml", ".png")

        if os.path.exists(os.path.join(IMAGE_DIR, img_name_jpg)):
            shutil.copy(
                os.path.join(IMAGE_DIR, img_name_jpg),
                OUTPUT_IMAGES
            )
        elif os.path.exists(os.path.join(IMAGE_DIR, img_name_png)):
            shutil.copy(
                os.path.join(IMAGE_DIR, img_name_png),
                OUTPUT_IMAGES
            )

    elif file.endswith(".txt"):

        img_base = file.replace(".txt", "")
        img_path_jpg = os.path.join(IMAGE_DIR, img_base + ".jpg")
        img_path_png = os.path.join(IMAGE_DIR, img_base + ".png")

        # copy image
        if os.path.exists(img_path_jpg):
            shutil.copy(img_path_jpg, OUTPUT_IMAGES)
        elif os.path.exists(img_path_png):
            shutil.copy(img_path_png, OUTPUT_IMAGES)

        # process label
        process_yolo(
            label_path,
            os.path.join(OUTPUT_LABELS, file)
        )

print("Dataset conversion from single folder complete!")
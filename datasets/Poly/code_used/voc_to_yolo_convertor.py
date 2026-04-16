import os
import xml.etree.ElementTree as ET

# Paths
xml_dir = "/home/kjel/Downloads/AI Project/Fire Signs/FireNet_ObjectDetection"
img_dir = "/home/kjel/Downloads/AI Project/Fire Signs/FireNet_Images(1)"
output_dir = "/home/kjel/Downloads/AI Project"

os.makedirs(output_dir, exist_ok=True)

# Defined classes for one of the datasets
classes = ["Alarm_Activator", "Fire_Blanket", "Fire_Exit", "Fire_Extinguisher", "Fire_Suppression_Signage", "Flashing_Light_Orbs", "Sounders", "White_Domes"]


def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0

    width = box[1] - box[0]
    height = box[3] - box[2]

    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh

    return (x_center, y_center, width, height)


for xml_file in os.listdir(xml_dir):
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    label_file = xml_file.replace(".xml", ".txt")
    label_path = os.path.join(output_dir, label_file)

    with open(label_path, "w") as f:
        for obj in root.iter("object"):
            cls = obj.find("name").text

            if cls not in classes:
                continue

            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")
            xmin = float(xmlbox.find("xmin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymin = float(xmlbox.find("ymin").text)
            ymax = float(xmlbox.find("ymax").text)

            bb = convert_box((w, h), (xmin, xmax, ymin, ymax))

            f.write(f"{cls_id} {' '.join(map(str, bb))}\n")

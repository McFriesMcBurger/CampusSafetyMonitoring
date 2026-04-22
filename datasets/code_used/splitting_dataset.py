import os
import random
import shutil

IMAGES_DIR = "/home/kjel/Downloads/AI Project/Poly/images"
LABELS_DIR = "/home/kjel/Downloads/AI Project/Poly/labels"

OUTPUT_DIR = "/home/kjel/Downloads/AI Project/Poly/dataset"

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

images = [f for f in os.listdir(IMAGES_DIR)
          if f.endswith((".jpg", ".png", ".jpeg"))]

random.shuffle(images)

# Computing exact values required for train, val, and test split (70/20/10)
train_split = int(2908 * 0.70)   # 2035
val_split = int(2908 * 0.20)     # 581

# splitting the dataset
train_files = images[:train_split]
val_files = images[train_split:train_split + val_split]
test_files = images[train_split + val_split:]

# function to copy files to target directories (with respect to split)
def move(files, split):
    for img in files:
        base = os.path.splitext(img)[0]

        img_path = os.path.join(IMAGES_DIR, img)
        label_path = os.path.join(LABELS_DIR, base + ".txt")

        if not os.path.exists(label_path):
            continue

        shutil.copy(img_path, f"{OUTPUT_DIR}/images/{split}/")
        shutil.copy(label_path, f"{OUTPUT_DIR}/labels/{split}/")

move(train_files, "train")
move(val_files, "val")
move(test_files, "test")

print("Dataset split complete!")
# Campus Safety Monitoring with YOLOv8

## Overview

This project uses a YOLOv8 object detection model trained on a custom dataset to identify safety-related equipment in images. The goal is to detect objects such as fire extinguishers, alarms, and defibrillators for monitoring or demonstration purposes.

The repository includes code for training, evaluation, and inference, along with utilities for dataset validation and testing.

---

## Classes

The model is trained on the following categories:

* Fire_Extinguisher
* Fire_Blanket
* Alarm_Activator
* Alarms
* Smoke_Detector
* AED
* Fire_Exit
* Wet_Floor_Sign
* Safety_Signs

---

## Project Structure

```
CampusSafetyMonitoring/
│
├── datasets/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── models/
│   └── best.pt
│
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── test.py
│   ├── utils.py
│   └── config.py
│   └── run_experiments.py
│
├── demo/              # optional: images for testing/demo
├── runs/              # generated during training
├── evaluation/        # generated during evaluation
│
└── README.md
```

---

## Setup

Install dependencies:

```
pip install ultralytics torch
```

If you plan to use a GPU, confirm that CUDA is available:

```python
import torch
print(torch.cuda.is_available())
```

Utilize the link below to download the ZIP file containing the images dataset:
https://drive.google.com/drive/folders/1M9X8GfTqeieVonsRrJKasLKXcEoJcLVP?usp=sharing

Insert the **images** folder into the same directory as the **labels** folder.

---

## Workflow

### Training

Run:

```
python src/train.py
```

This starts training from pretrained YOLOv8 weights (`yolov8n.pt`).
The trained model is saved to:

```
runs/detect/campus_safety_model/weights/best.pt
```

Move or copy this file to:

```
models/best.pt
```

---

### Evaluation

Run:

```
python src/evaluate.py
```

This computes standard detection metrics and saves outputs to the `evaluation/` directory, including:

* Precision and recall
* mAP@0.5 and mAP@0.5:0.95
* Confusion matrix
* Precision–recall and F1 curves

For experimentation with the three provided categories in config.py, run:

```
python src/run_experiments.py
```

This initiates the experimentation pipeline, wherein the the model will be trained under the baseline configuration, three experimental configurations:

* Regularization and early stopping strategies
* Data augmentation techniques
* Training pipeline optimization

and then a final, combined model implementing all experimental changes.

---

### Validation (Quick Check)

Run:

```
python src/test.py
```

This performs a validation pass and prints metrics to the console.

---

### Inference

#### Single Image

```
python src/inference.py demo/image.jpg
```

This runs detection on a single image and prints detected classes with confidence scores.

#### Folder of Images

Modify or call:

```python
predict("demo/")
```

to process all images in a directory.

Output images with detections are saved under:

```
runs/detect/predict/
```

---

## Dataset Requirements

The dataset must follow YOLO format:

```
datasets/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
```

Each image must have a corresponding `.txt` label file with normalized bounding box annotations.

---

## Metrics

The following metrics are used:

* Precision: proportion of correct detections
* Recall: proportion of objects successfully detected
* mAP@0.5: standard object detection accuracy
* mAP@0.5:0.95: stricter metric across IoU thresholds
* Confusion matrix: class-level performance and errors

---

## Common Issues

* Incorrect model path: ensure `models/best.pt` exists
* COCO labels appearing: indicates the pretrained model is being used instead of the trained one
* Training instability: reduce batch size or number of workers
* Poor results: check dataset quality and label consistency

---

## Cleanup

To reset the workspace before retraining, remove:

```
runs/
evaluation/
*.cache
*.npy
```

from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model("test.jpg", show=True)
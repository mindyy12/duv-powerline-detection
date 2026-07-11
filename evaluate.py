from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")
metrics = model.val(data="data.yaml", split="test")

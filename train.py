from ultralytics import YOLO

# running from last epoch 
model = YOLO("runs/detect/train/weights/last.pt")
# for my cpu capacity, batch=8 for memory
model.train(data="data.yaml", epochs=50, imgsz=416, batch=8)
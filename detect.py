from ultralytics import YOLO

model = YOLO("yolov8n.pt")
result = model("https://ultralytics.com/images/bus.jpg")[0]
result.save(filename="output.jpg")

print(f"Detected {len(result.boxes)} objects:")
for box in result.boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f" {class_name}: {confidence:.2f} confidence, box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

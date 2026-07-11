from flask import Flask, render_template, request
from ultralytics import YOLO
from collections import Counter
import datetime
import os
import shutil

app = Flask(__name__)
model = YOLO("runs/detect/train-2/weights/best.pt")

test_images_dir = "test/images"
test_images = sorted(os.listdir(test_images_dir))

SEVERITY = {
    "Broken Cable": "critical",
    "Broken Insulator": "critical",
    "Cable": "warning",
    "Insulators": "normal",
    "Tower": "warning",
    "Vegetation": "warning",
}

CATEGORY = {
    "Broken Cable": "Structural",
    "Broken Insulator": "Component",
    "Cable": "Component",
    "Insulators": "Component",
    "Tower": "Structural",
    "Vegetation": "Environmental",
}

FRAME_COORDINATES = "67.6767°N, 67.6767°E"

@app.route("/")
def index():
    frame_index = int(request.args.get("index", 0))
    frame_index = max(0, min(frame_index, len(test_images) - 1))

    image_name = test_images[frame_index]
    image_path = os.path.join(test_images_dir, image_name)
    result = model(image_path)[0]

    shutil.copy(image_path, "static/frame.jpg")

    img_height, img_width = result.orig_shape

    detections = []
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "id": f"F-{i + 1:03d}",
            "class_name": class_name,
            "confidence": float(box.conf[0]),
            "severity": SEVERITY[class_name],
            "category": CATEGORY[class_name],
            "left": x1 / img_width * 100,
            "top": y1 / img_height * 100,
            "width": (x2 - x1) / img_width * 100,
            "height": (y2 - y1) / img_height * 100,
        })
    
    severity_counts = {
        "critical": sum(1 for d in detections if d["severity"] == "critical"),
        "warning": sum(1 for d in detections if d["severity"] == "warning"),
        "normal": sum(1 for d in detections if d["severity"] == "normal"),
    }

    if detections:
        most_common_category = Counter(d["category"] for d in detections).most_common(1)[0][0]
    else:
        most_common_category = "-"

    last_scan = datetime.datetime.now().strftime("%H:%M")

    return render_template(
        "index.html",
        detections=detections,
        severity_counts=severity_counts,
        most_common_category=most_common_category,
        last_scan=last_scan,
        coordinates=FRAME_COORDINATES,
        index=frame_index,
        total=len(test_images),
        image_name=image_name,
    )

if __name__ == "__main__":
    app.run(debug=True)

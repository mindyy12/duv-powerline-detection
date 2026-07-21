from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from collections import Counter
import datetime
import os
import shutil
import requests
import csv

app = Flask(__name__)
model = YOLO("runs/detect/train-2/weights/best.pt")

test_images_dir = "test/images"
test_images = sorted(os.listdir(test_images_dir))

SEVERITY = {
    "Broken Cable": "critical",
    "Broken Insulator": "critical",
    "Cable": "normal",
    "Insulators": "normal",
    "Tower": "normal",
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

BOM_BASE = "https://api.weather.bom.gov.au/v1"
# UNSW Maccas
BASE_LAT = -33.919548
BASE_LONG = 151.227176

# Generate synthetic coords for frames
def generate_coordinates(frame_index):
    lat = BASE_LAT + (frame_index * 0.0006)
    lon = BASE_LONG + (frame_index * 0.0004)
    return lat, lon

# Format coordinates
def format_coordinates(lat, lon):
    lat_dir = "S" if lat < 0 else "N"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"

# Retrieve BOM
WEATHER_DEFAULT = {
    "level": "unknown",
    "warnings": [],
    "warning_group_type": None,
    "wind_speed": None,
    "wind_direction": None,
    "gust_speed": None,
    "max_gust_speed": None,
    "fire_danger": None,
    "fire_danger_colour": None,
}

def get_weather_data(lat, lon):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        # searches location
        search_resp = requests.get(
            f"{BOM_BASE}/locations",
            params={"search": f"{lat}, {lon}"},
            headers=headers,
            timeout=5,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("data", [])
        if not results:
            return dict(WEATHER_DEFAULT)
        
        geohash = results[0]["geohash"][:6]

        # gets climate warnings
        warnings_resp = requests.get(
            f"{BOM_BASE}/locations/{geohash}/warnings",
            headers=headers,
            timeout=5
        )
        warnings_resp.raise_for_status()
        warnings_data = warnings_resp.json().get("data", [])

        # gets wind conditions
        observations_resp = requests.get(
            f"{BOM_BASE}/locations/{geohash}/observations",
            headers=headers,
            timeout=5
        )
        observations_resp.raise_for_status()
        obs = observations_resp.json().get("data", {})

        # gets fire danger
        daily_resp = requests.get(
            f"{BOM_BASE}/locations/{geohash}/forecasts/daily",
            headers=headers,
            timeout=5
        )
        daily_resp.raise_for_status()
        daily_data = daily_resp.json().get("data", [])
        daily = daily_data[0] if daily_data else {}

        # weather dictionary
        weather = dict(WEATHER_DEFAULT)
        weather["wind_speed"] = (obs.get("wind") or {}).get("speed_kilometre")
        weather["wind_direction"] = (obs.get("wind") or {}).get("direction")
        weather["gust_speed"] = (obs.get("gust") or {}).get("speed_kilometre")
        weather["max_gust_speed"] = (obs.get("max_gust") or {}).get("speed_kilometre")
        weather["fire_danger"] = daily.get("fire_danger")
        weather["fire_danger_colour"] = (daily.get("fire_danger_category") or {}).get("dark_mode_colour")
        if warnings_data:
            first_warning = warnings_data[0]
            weather["level"] = "high"
            weather["warnings"] = [w.get("title", "Active warning") for w in warnings_data]
            weather["warning_group_type"] = first_warning.get("warning_group_type")
        else:
            weather["level"] = "low"

        return weather

    except requests.RequestException:
        return dict(WEATHER_DEFAULT)

# retrieve wind data
def load_wind_events(path="wind.csv"):
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
                gust_knots = float(row["Max Gust speed"])
            except (ValueError, KeyError):
                continue
            
            if lat == 0 and lon == 0:
                continue
            
            if gust_knots > 150:
                continue

            if gust_knots == 0:
                gust_knots = 40
            
            events.append({
                "lat": lat,
                "lon": lon,
                "gust_kmh": gust_knots * 1.852,
                "date": row["Date/Time"],
                "town": row["Nearest town"],
            })

    return events


# discretise events into a fixed grid, coloured by event count (not zoom-dependent)
def build_wind_density_grid(events, cell_size=0.25):
    import math
    from collections import defaultdict

    counts = defaultdict(int)
    for e in events:
        key = (math.floor(e["lat"] / cell_size), math.floor(e["lon"] / cell_size))
        counts[key] += 1

    cells = []
    for (lat_idx, lon_idx), count in counts.items():
        if count >= 6:
            level = "high"
        elif count >= 3:
            level = "medium"
        else:
            level = "low"

        cells.append({
            "lat_min": lat_idx * cell_size,
            "lat_max": (lat_idx + 1) * cell_size,
            "lon_min": lon_idx * cell_size,
            "lon_max": (lon_idx + 1) * cell_size,
            "count": count,
            "level": level,
        })

    return cells

# Urgency classification
def apply_storm_urgency(detections, storm_risk):
    if storm_risk["level"] != "high":
        return detections
    bump = {"normal": "warning", "warning": "critical", "critical": "critical"}
    for d in detections:
        d["severity"] = bump[d["severity"]]
    return detections

DEFAULT_THEME = "editorial"
THEMES = ["editorial", "telemetry", "daylight"]


def current_theme():
    theme = request.cookies.get("theme", DEFAULT_THEME)
    return theme if theme in THEMES else DEFAULT_THEME


@app.route("/")
def index():
    frame_index = int(request.args.get("index", 0))
    frame_index = max(0, min(frame_index, len(test_images) - 1))

    image_name = test_images[frame_index]
    image_path = os.path.join(test_images_dir, image_name)
    result = model(image_path)[0]

    shutil.copy(image_path, "static/frame.jpg")
    result.save(filename="static/export.jpg")

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
    
    lat, lon = generate_coordinates(frame_index)
    coordinates = format_coordinates(lat, lon)
    weather = get_weather_data(lat, lon)
    detections = apply_storm_urgency(detections, weather)
    
    severity_counts = {
        "critical": sum(1 for d in detections if d["severity"] == "critical"),
        "warning": sum(1 for d in detections if d["severity"] == "warning"),
        "normal": sum(1 for d in detections if d["severity"] == "normal"),
    }

    if detections:
        most_common_category = Counter(d["category"] for d in detections).most_common(1)[0][0]
    else:
        most_common_category = "-"

    last_scan = datetime.datetime.now().strftime("%A %d %B %Y %H:%M:%S")

    return render_template(
        "index.html",
        detections=detections,
        severity_counts=severity_counts,
        most_common_category=most_common_category,
        last_scan=last_scan,
        coordinates=coordinates,
        weather=weather,
        index=frame_index,
        total=len(test_images),
        image_name=image_name,
        theme=current_theme(),
        active_page="live",
    )


@app.route("/predictive")
def predictive():
    wind_events = load_wind_events()
    wind_grid = build_wind_density_grid(wind_events)
    return render_template(
        "predictive.html",
        theme=current_theme(),
        active_page="predictive",
        wind_events=wind_events,
        wind_grid=wind_grid,
    )


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        theme = request.form.get("theme", DEFAULT_THEME)
        if theme not in THEMES:
            theme = DEFAULT_THEME
        response = redirect(url_for("settings"))
        response.set_cookie("theme", theme, max_age=60 * 60 * 24 * 365)
        return response

    return render_template(
        "settings.html",
        theme=current_theme(),
        active_page="settings",
    )


if __name__ == "__main__":
    app.run(debug=True)

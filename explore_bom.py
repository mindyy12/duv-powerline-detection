import requests
import json

headers = {"User-Agent": "Mozilla/5.0"}
lat, lon = -33.919548, 151.227176

search = requests.get(
    "https://api.weather.bom.gov.au/v1/locations",
    params={"search": f"{lat}, {lon}"},
    headers=headers,
).json()
print("=== LOCATION SEARCH ===")
print(json.dumps(search, indent=2))

geohash = search["data"][0]["geohash"][:6]

warnings = requests.get(
    f"https://api.weather.bom.gov.au/v1/locations/{geohash}/warnings",
    headers=headers,
).json()
print("\n=== WARNINGS ===")
print(json.dumps(warnings, indent=2))

observations = requests.get(
    f"https://api.weather.bom.gov.au/v1/locations/{geohash}/observations",
    headers=headers,
).json()
print("\n=== OBSERVATIONS ===")
print(json.dumps(observations, indent=2))

daily = requests.get(
    f"https://api.weather.bom.gov.au/v1/locations/{geohash}/forecasts/daily",
    headers=headers,
).json()
print("\n=== DAILY FORECAST (first entry) ===")
print(json.dumps(daily["data"][0], indent=2))

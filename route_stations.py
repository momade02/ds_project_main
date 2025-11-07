"""
Description: This script uses the OpenRouteService (ORS) API to geocode two addresses,
calculate a driving route between them, and find gas stations along that route within a specified buffer distance
"""

# === Setup ===
# import libraries
import os
from pathlib import Path
import json
import requests

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except Exception:
    pass

# Theoretical user Input
# Hardcoded addresses
# Names of the variables inline with OpenRouteService API Query parameters
start_address = "Wilhelmstraße 7"
start_locality = "Tübingen"
start_country = "Germany"

end_address = "Charlottenstraße 45"
end_locality = "Reutlingen"
end_country = "Germany"

# Search parameter for the gas stations along the route
# must not be larger than 2000 meter according to ORS docs
buffer_meters = 100  # buffer width in meters


# OpenRouteService API configuration
ors_base = "https://api.openrouteservice.org"

# check if ors_api_key is set in environment variables
ors_api_key = os.getenv("ORS_API_KEY")
if not ors_api_key:
    raise SystemExit("Please set your ORS_API_KEY environment variable first!")

# === Functions ===

# Function for structured geocoding search
def ors_geocode_structured(street, city, country, api_key):
    """
    Geocode a structured address using ORS.
    
    Inputs:
        street (str): Street address
        city (str): City name
        country (str): Country name
        api_key (str): ORS API key
    Returns:
        lat (float): Latitude of the address
        lon (float): Longitude of the address
        label (str): Formatted address label, for detail see ORS docs
    """

    url = f"{ors_base}/geocode/search/structured"
    params = {
        "api_key": api_key,
        "address": street,
        "locality": city,
        "country": country,
        "size": 1
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    features = data.get("features", [])
    if not features:
        raise ValueError(f"No results for {street}, {city}, {country}")

    feature = features[0]
    lon, lat = feature["geometry"]["coordinates"]
    label = feature["properties"].get("label", "Unknown")
    print("Function ors_geocode_structured successful")
    return lat, lon, label

# Function to get route between two coordinates
def ors_route_driving_car(start_lat, start_lon, end_lat, end_lon, api_key):
    """
    Get driving route between two coordinates using ORS.

    Inputs:
        start_lat (float): Start latitude
        start_lon (float): Start longitude
        end_lat (float): End latitude
        end_lon (float): End longitude
        api_key (str): ORS API key
    Returns:
        coords_lonlat (list): List of [lon, lat] coordinates along the route
        distance_km (float): Total distance of the route in kilometers
        duration_min (float): Total duration of the route in minutes
    """
    
    url = f"{ors_base}/v2/directions/driving-car/geojson"
    body = {
        "coordinates": [[start_lon, start_lat], [end_lon, end_lat]],
        "instructions": False  # just need geometry, distance, duration
    }
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(body))
    r.raise_for_status()
    data = r.json()

    feat = data["features"][0]
    props = feat.get("properties", {})
    summary = props.get("summary", {})  # distance (m), duration (s)
    coords_lonlat = feat["geometry"]["coordinates"]  # [[lon,lat], ...]

    distance_km = (summary.get("distance", 0.0) or 0.0) / 1000.0
    duration_min = (summary.get("duration", 0.0) or 0.0) / 60.0

    print("\n Function ors_route_driving_car successful")
    return coords_lonlat, distance_km, duration_min

# Function to get fuel stations (POIs) along a route using ORS /pois endpoint
def ors_pois_fuel_along_route(route_coords_lonlat, buffer_meters, api_key):
    """
    Get fuel stations (category ID 596) along a driving route using ORS /pois endpoint.

    Inputs:
        route_coords_lonlat (list): List of [lon, lat] coordinates along the route
        buffer_meters (float): Search radius (buffer) around the route in meters
        api_key (str): ORS API key

    Returns:
        stations (list): List of fuel stations with 'name', 'lat', 'lon'
    """
    
    url = f"{ors_base}/pois"
    body = {
        "request": "pois",
        "geometry": {
            "buffer": buffer_meters,
            "geojson": {
                "type": "LineString",
                "coordinates": route_coords_lonlat  # [lon, lat] from the route
            }
        },
        "filters": {
            "category_ids": [596]  # 596 = Fuel (gas stations)
        },
        "limit": 500  # optional, max number of results
    }
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    
    # Send POST request to ORS /pois endpoint
    r = requests.post(url, headers=headers, data=json.dumps(body))

    # for debugging purposes, print status code and reason, delete later
    print("\n Delete later \n Status code:", r.status_code, "Reason:", r.reason)

    r.raise_for_status()
    pois_data = r.json()

    stations = []
    for feature in pois_data.get("features", []):
        props = feature.get("properties", {})
        tags = props.get("osm_tags", {})
        name = tags.get("name") or tags.get("brand") or "Unnamed"
        lon, lat = feature["geometry"]["coordinates"]
        stations.append({"name": name, "lat": lat, "lon": lon})
    
    print("\n Function ors_pois_fuel_along_route successful")
    return stations


# === Execution ===

# get Geocode of start and end addresses
start_lat, start_lon, start_label = ors_geocode_structured(
    start_address, start_locality, start_country, ors_api_key
)
end_lat, end_lon, end_label = ors_geocode_structured(
    end_address, end_locality, end_country, ors_api_key
)

# Check results of structured geocoding search
print("\n--- Check GEOCODING RESULTS ---")
print(f"Start: {start_label}")
print(f" → lat = {start_lat:.6f}, lon = {start_lon:.6f}")
print(f"End: {end_label}")
print(f" → lat = {end_lat:.6f}, lon = {end_lon:.6f}")

# get the route between two coordinates
route_coords_lonlat, route_km, route_min = ors_route_driving_car(
    start_lat, start_lon, end_lat, end_lon, ors_api_key
)

print("\n--- Check ROUTE ---")
print(f"Route: {route_km:.1f} km")
print(f"Duration: {route_min:.0f} min")
print(f"Number of points: {len(route_coords_lonlat)} coordinates along the route")

# get data of fuel stations along the route
stations = ors_pois_fuel_along_route(route_coords_lonlat, buffer_meters, ors_api_key)

print("\n--- Check FUEL STATIONS ---")
for s in stations[:10]:
    print(f"{s['name']} → lat={s['lat']:.5f}, lon={s['lon']:.5f}")
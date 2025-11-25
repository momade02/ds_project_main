"""
Description: This script uses the OpenRouteService (ORS) API to geocode two addresses,
calculate a driving route between them, and find gas stations along that route within a specified buffer distance.
"""

# test case long/short/switzerland
route_scenario = "short" # "long", "short" or "switzerland"

# === Setup ===
# import libraries
import os
from pathlib import Path
import json
import requests
from dotenv import load_dotenv
from shapely.geometry import LineString, Point
from shapely.ops import transform, substring
import pyproj
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timedelta
import googlemaps

# Theoretical user Input
# Hardcoded addresses
if route_scenario == "short":
    start_address = "Wilhelmstraße 7"
    start_locality = "Tübingen"
    start_country = "Germany"
    end_address = "Charlottenstraße 45"
    end_locality = "Reutlingen"
    end_country = "Germany"
elif route_scenario == "long":
    start_address = "Wilhelmstraße 7"
    start_locality = "Tübingen"
    start_country = "Germany"
    end_address = "Borsigallee 26"
    end_locality = "Frankfurt am Main"
    end_country = "Germany"
    # end_address = "Langberger Weg 4"
    # end_locality = "Flensburg"
    # end_country = "Germany"
elif route_scenario == "switzerland":
    start_address = "Zinngärten 9"
    start_locality = "Stühlingen"
    start_country = "Germany"
    end_address = "Lendenbergstrasse 32"
    end_locality = "Schleitheim"
    end_country = "Switzerland"
else:
    raise ValueError("Unknown route_scenario")

# Search parameter for the gas stations along the route
# must not be larger than 2000 meter according to ORS docs
buffer_meters = 300  # buffer width in meters

# OpenRouteService API configuration
ors_base = "https://api.openrouteservice.org"

# === helper functions ===
# # Functions that help main functions.

def environment_check():
    """
    Check if the GOOGLE_MAPS_API_KEY environment variable is set.
    Raises SystemExit if not set.

    Returns:
        google_api_key (str): The Google Maps API key from environment variable
    """
    load_dotenv()

    # check if google api key is set in environment variables
    google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not google_api_key:
        raise SystemExit("Please set your GOOGLE_MAPS_API_KEY environment variable first!")
    return google_api_key

def determine_segments(total_length_km, buffer_m):
    """
    Determine appropriate segment length in meters for route segmentation, based on ORS API and developers constraints.

    Inputs:
        total_length_km (float): Total length of the route in kilometers
        buffer_m (float): Buffer width in meters
    Returns:
        segment_length_m (float): Length of each segment in meters
    """

    max_segments = 12  # 12, so user can request a new route every 12 seconds
    max_segment_length = 240  # km, from the docs with a little safety margin
    max_area_km2 = 49         # km^2, from the docs with a little safety margin

    if buffer_m <= 0:
        raise ValueError("buffer_m must be > 0")
    elif buffer_m > 1000:
        raise ValueError("buffer_m must be <= 1000")

    for segments in range(1, max_segments + 1):
        L = total_length_km / segments  # segment length in km

        # Constraint 1: segment length
        if L > max_segment_length:
            continue

        # Constraint 2: area
        area = 2 * L * (buffer_m / 1000)
        if area > max_area_km2:
            continue

        return L * 1000 # return segment length in meters

    raise ValueError(
        f"Cannot satisfy API constraints with buffer={buffer_m}m and total route length={total_length_km} km. "
        "Try a smaller buffer or a shorter route."
    )

def simplify_route(coords_lonlat, tolerance=0.0002):
      #  tolerance ~22m
    """
    Simplify a route by reducing the number of points using the Ramer-Douglas-Peucker algorithm.

    Inputs:
        coords_lonlat (list): List of [lon, lat] coordinates along the route
        tolerance (float): Tolerance for simplification, smaller values retain more points
    Returns:
        simplified_coords (list): Simplified list of [lon, lat] coordinates
    """

    line = LineString(coords_lonlat)
    simplified = line.simplify(tolerance, preserve_topology=False)

    print(f"Simplified route from {len(coords_lonlat)} to {len(simplified.coords)} points")

    return list(simplified.coords)

def segment_route(coords_lonlat, segment_length_m):
    """
    Segment a route into smaller segments of specified length in meters.

    Inputs:
        coords_lonlat (list): List of [lon, lat] coordinates along the route
        segment_length_m (float): Length of each segment in meters
    Returns:
        segments_lonlat (list): List of segments, each segment is a list of [lon, lat] coordinates
    """

    line = LineString(coords_lonlat)

    # transormer for WGS 84 (EPSG:4326) to UTM zone 32N (EPSG:32632)
    project_to_utm = pyproj.Transformer.from_crs(crs_from =
        "EPSG:4326", crs_to = "EPSG:32632", always_xy = True
    ).transform

    # transformer back
    project_to_lonlat = pyproj.Transformer.from_crs(
        "EPSG:32632", "EPSG:4326", always_xy=True
    ).transform

    line_m = transform(project_to_utm, line)

    segments_m = []
    total_length = line_m.length

    start = 0
    while start < total_length:
        end = start + segment_length_m
        seg = substring(line_m, start, end)
        segments_m.append(seg)
        start = end
    
    segments_lonlat = []
    for seg in segments_m:
        seg_lonlat = transform(project_to_lonlat, seg)
        coords = list(seg_lonlat.coords)
        segments_lonlat.append([[lon, lat] for (lon, lat) in coords])
    
    return segments_lonlat

def estimate_arrival_times(stations, route_coords_lonlat, total_distance_km, total_duration_min, departure_time=None):
    """
    Augments each station with:
      - distance_along_m: meters from the start along the route (projected)
      - fraction_of_route: fraction of the total route (0..1)
      - eta: estimated time of arrival

    Args:
        stations (list): list of dicts containing 'lat' and 'lon'
        route_coords_lonlat (list): [[lon, lat], ...] route coordinates
        total_distance_km (float): total route length in km (from ors_route_driving_car)
        total_duration_min (float): total route duration in minutes (from ors_route_driving_car)
        departure_time (datetime|None): departure time. If None, current time is used

    Returns:
        list: copied station dicts with added fields
    """

    if departure_time is None:
        departure_time = datetime.utcnow()

    project_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform
    line = LineString(route_coords_lonlat)
    line_m = transform(project_to_utm, line)
    total_length_m = total_distance_km * 1000.0

    results = []
    for s in stations:
        pt = Point(s["lon"], s["lat"])
        pt_m = transform(project_to_utm, pt)
        dist_along = line_m.project(pt_m)
        fraction = dist_along / total_length_m
        offset_seconds = fraction * (total_duration_min * 60.0)
        eta = departure_time + timedelta(seconds=offset_seconds)

        s_copy = s.copy()
        s_copy.update({
            "distance_along_m": float(dist_along),
            "fraction_of_route": float(fraction),
            "eta": eta.isoformat()
        })
        results.append(s_copy)

    return results

# === main functions ===
# Functions to interact with OpenRouteService (ORS) API.

# Function for structured geocoding search
def ors_geocode_structured(street, city, country, api_key):
    """
    Geocode a structured address using googlemaps.Client.

    Inputs:
        street (str): Street address
        city (str): City name
        country (str): Country name
        api_key (str): ORS API key
    Returns:
        lat (float), lon (float), label (str)
    """
    client = googlemaps.Client(key=api_key)
    address = f"{street}, {city}, {country}"
    results = client.geocode(address)

    if not results:
        raise ValueError(f"No results for {street}, {city}, {country}")

    top = results[0]
    loc = top.get("geometry", {}).get("location", {})
    lat = loc.get("lat")
    lon = loc.get("lng")
    label = top.get("formatted_address", address)

    if lat is None or lon is None:
        raise ValueError(f"Geocoding returned no coordinates for {address}")

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

    distance_km = (summary.get("distance") or 0.0) / 1000.0
    duration_min = (summary.get("duration") or 0.0) / 60.0

    return coords_lonlat, distance_km, duration_min

# Function to get fuel stations (POIs) along a route segment using ORS /pois endpoint
def ors_pois_single_segment(segment_coords, buffer_meters, api_key, timeout_seconds):
    """
    Get fuel stations (category ID 596) along a driving route using ORS /pois endpoint.
    """
    
    url = f"{ors_base}/pois"
    body = {
        "request": "pois",
        "geometry": {
            "buffer": buffer_meters,
            "geojson": {
                "type": "LineString",
                "coordinates": segment_coords  # [lon, lat] from the route segment
            }
        },
        "filters": {
            "category_ids": [596]  # 596 = Fuel (gas stations)
        }
    }
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    
    try:
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout_seconds)

        # for debugging purposes, print status code and reason, delete later
        print("\n Delete later \n Status code:", r.status_code, "Reason:", r.reason)
        # print(r.text)
        r.raise_for_status()
        pois_data = r.json()

    except requests.Timeout:
          #  catches timeout exception
        raise TimeoutError(f"ORS /pois request timed out after {timeout_seconds} seconds.")

    stations = []
    for feature in pois_data.get("features", []):
        props = feature.get("properties", {})
        tags = props.get("osm_tags", {})
        name = tags.get("name") or tags.get("brand") or "Unnamed"
        lon, lat = feature["geometry"]["coordinates"]
        distance = props.get("distance") or 0.0
          #  distance in meters from the route (as the crow flies? Nothing can be found in ORS docs)
          #  can implement distance calculation with shapely if needed
        stations.append({"name": name, "lat": lat, "lon": lon, "distance": distance})

    return stations

# Function to get fuel stations (POIs) along a route using ORS /pois endpoint
def ors_pois_fuel_along_route(route_coords_lonlat, buffer_meters, api_key, route_length_km, timeout_seconds=90):
    """
    Get fuel stations (category ID 596) along a driving route using ORS /pois endpoint.
    If the route is long, it segments the route and runs multiple requests in parallel.

    Inputs:
        route_coords_lonlat (list): List of [lon, lat] coordinates along the route
        buffer_meters (float): Search radius (buffer) around the route in meters
        api_key (str): ORS API key
        timeout_seconds (int|float): max seconds to wait for the ORS server

    Returns:
        stations (list): List of fuel stations with 'name', 'lat', 'lon', 'distance'
    """

    # call helper function to reduce number of points along the route
    route_coords_lonlat = simplify_route(route_coords_lonlat)

    # call helper function to segment the route if needed
    route_segments = segment_route(route_coords_lonlat, segment_length_m=determine_segments(route_length_km, buffer_meters))

    if len(route_segments) == 1:
        return ors_pois_single_segment(
            route_segments[0], buffer_meters, api_key, timeout_seconds
        )

    print(f"Running {len(route_segments)} segment requests in parallel...")

    all_stations = []
    futures = []

    with ThreadPoolExecutor(max_workers=min(4, len(route_segments))) as executor:
        for seg in route_segments:
            print(f"Segment {route_segments.index(seg)} -> coords: {len(seg)}, approx payload bytes: {len(seg)*32}")
            futures.append(
                executor.submit(
                    ors_pois_single_segment,
                    seg, buffer_meters, api_key, timeout_seconds
                )
            )

        for f in as_completed(futures):
            all_stations.extend(f.result())

    unique = {}
    for s in all_stations:
        key = (s["lat"], s["lon"])
        if key not in unique:
            unique[key] = s

    print(f"Finished parallel POI search. Found total: {len(unique)} fuel stations.")

    return list(unique.values())


# === Execution ===
def main():
    """
    Command-line / direct-run entrypoint for testing the route pipeline.
    When imported as a module (e.g. by Streamlit), this function is NOT executed.
    """
    # get ORS API key from environment variable
    google_api_key = environment_check()

    # get Geocode of start and end addresses
    start_lat, start_lon, start_label = ors_geocode_structured(
        start_address, start_locality, start_country, google_api_key
    )
    end_lat, end_lon, end_label = ors_geocode_structured(
        end_address, end_locality, end_country, google_api_key
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
    print(f"Number of points unthinned list: {len(route_coords_lonlat)} coordinates along the route")

    # get data of fuel stations along the route
    stations = ors_pois_fuel_along_route(route_coords_lonlat, buffer_meters, ors_api_key, route_km)

    stations_with_eta = estimate_arrival_times(stations, route_coords_lonlat, route_km, route_min)

    print("\n--- Stations with ETA ---")
    for s in stations_with_eta:
        print(f"{s.get('name','Unnamed')} → lat={s.get('lat',0.0):.5f}, lon={s.get('lon',0.0):.5f}, distance={s.get('distance',0.0)} m, "
            f"along={s.get('distance_along_m',0.0):.0f} m, "
            f"share={s.get('fraction_of_route',0.0):.1%}, ETA={s.get('eta','-')}")

if __name__ == "__main__":
    main()
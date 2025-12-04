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
from shapely.geometry import LineString
from shapely.ops import transform, substring
import pyproj
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import googlemaps
import polyline
import time

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

# def validate_route_inputs(start_address, start_locality, end_address, end_locality):
#     """
#     Validate that the start and end addresses and localities are non-empty strings and not identical.
#     Returns cleaned (stripped) values as a tuple:
#         (start_address, start_locality, end_address, end_locality)
#     Raises:
#         TypeError, ValueError
#     """
#     def clean(value, name):
#         v = value.strip()
#         if not v:
#             raise ValueError(f"{name} must not be empty")
#         if len(v) < 2:
#             raise ValueError(f"{name} is too short")
#         return v
        
#     clean(start_address, "start_address")
#     clean(start_locality, "start_locality")
#     clean(end_address, "end_address")
#     clean(end_locality, "end_locality")

#     # normalize for comparison (casefold for robust case-insensitive compare)
#     def normalized(addr, loc):
#         return f"{addr}, {loc}".casefold()

#     if normalized(s_addr, s_loc) == normalized(e_addr, e_loc):
#         raise ValueError("Start and end address/locality must not be identical")

#     return s_addr, s_loc, e_addr, e_loc

# def simplify_route(coords_lonlat, tolerance=0.0002):
#       #  tolerance ~22m
#     """
#     Simplify a route by reducing the number of points using the Ramer-Douglas-Peucker algorithm.

#     Inputs:
#         coords_lonlat (list): List of [lon, lat] coordinates along the route
#         tolerance (float): Tolerance for simplification, smaller values retain more points
#     Returns:
#         simplified_coords (list): Simplified list of [lon, lat] coordinates
#     """

#     line = LineString(coords_lonlat)
#     simplified = line.simplify(tolerance, preserve_topology=False)

#     print(f"Simplified route from {len(coords_lonlat)} to {len(simplified.coords)} points")

#     return list(simplified.coords)

# def segment_route(coords_lonlat, segment_length_m):
#     """
#     Segment a route into smaller segments of specified length in meters.

#     Inputs:
#         coords_lonlat (list): List of [lon, lat] coordinates along the route
#         segment_length_m (float): Length of each segment in meters
#     Returns:
#         segments_lonlat (list): List of segments, each segment is a list of [lon, lat] coordinates
#     """

#     line = LineString(coords_lonlat)

#     # transormer for WGS 84 (EPSG:4326) to UTM zone 32N (EPSG:32632)
#     project_to_utm = pyproj.Transformer.from_crs(crs_from =
#         "EPSG:4326", crs_to = "EPSG:32632", always_xy = True
#     ).transform

#     # transformer back
#     project_to_lonlat = pyproj.Transformer.from_crs(
#         "EPSG:32632", "EPSG:4326", always_xy=True
#     ).transform

#     line_m = transform(project_to_utm, line)

#     segments_m = []
#     total_length = line_m.length

#     start = 0
#     while start < total_length:
#         end = start + segment_length_m
#         seg = substring(line_m, start, end)
#         segments_m.append(seg)
#         start = end
    
#     segments_lonlat = []
#     for seg in segments_m:
#         seg_lonlat = transform(project_to_lonlat, seg)
#         coords = list(seg_lonlat.coords)
#         segments_lonlat.append([[lon, lat] for (lon, lat) in coords])
    
#     return segments_lonlat

def parse_duration(d):
    """
    Parse a duration string in the format "{number}s" and return the number as float.
    Inputs:
        d (str): Duration string in the format "{number}s"
    Returns:
        float: Duration in seconds
    """
    if not d.endswith("s"):
        raise ValueError(f"Unexpected duration format: {d}")
    return float(d[:-1])


# === main functions ===
# Functions to interact with OpenRouteService (ORS) API.

# Function for structured geocoding search
def google_geocode_structured(street, city, country, api_key):
    """
    Geocode a structured address using googlemaps.Client.

    Inputs:
        street (str): Street address
        city (str): City name
        country (str): Country name
        api_key (str): Google Maps API key
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
def google_route_driving_car(start_lat, start_lon, end_lat, end_lon, api_key, departure_time="now"):
    """
    Get driving route between two coordinates using Google Directions API.

    Inputs:
        start_lat (float): Start latitude
        start_lon (float): Start longitude
        end_lat (float): End latitude
        end_lon (float): End longitude
        api_key (str): Google Maps API key
        departure_time (datetime|None): Desired departure time (UTC). If None, set to now.
    Returns:
        coords_lonlat (list): List of [lon, lat] coordinates along the route
        distance_km (float): Total distance of the route in kilometers
        duration_min (float): Total duration of the route in minutes
    """

    client = googlemaps.Client(key=api_key)

    directions = client.directions(
        origin=(start_lat, start_lon),
        destination=(end_lat, end_lon),
        mode="driving",
        alternatives=False,
        departure_time=departure_time,
        traffic_model="best_guess"
    )

    if not directions:
        raise ValueError("No route found by Google Directions API.")

    if departure_time == "now":
        departure_time = datetime.now(ZoneInfo("Europe/Berlin"))

    route = directions[0]

    # total distance (m) and duration (s)
    distance_m = sum(leg.get("distance", {}).get("value", 0) for leg in route.get("legs", []))
    duration_s = sum(leg.get("duration", {}).get("value", 0) for leg in route.get("legs", []))

    # prefer overview_polyline for geometry (expected to be present)
    overview = route.get("overview_polyline", {}).get("points")
    points = googlemaps.convert.decode_polyline(overview)
      #  convert to list of {"lat":..., "lng":...} for simplify_route
    coords_lonlat = [[p["lng"], p["lat"]] for p in points]

    distance_km = distance_m / 1000.0
    duration_min = duration_s / 60.0

    return coords_lonlat, distance_km, duration_min, departure_time

def google_places_fuel_along_route(segment_coords, api_key, original_distance_km, original_duration_min, departure_time, timeout_seconds = 30):
    """
    Get fuel stations along a route segment using Google Places Text Search (Search Along Route).

    Inputs:
        segment_coords (list): List of [lon, lat] coordinates for this route segment
        api_key (str): Google Maps API key
        original_distance_km (float): Original route distance in kilometers
        original_duration_min (float): Original route duration in minutes
        departure_time (datetime): Departure time (UTC)
        timeout_seconds (int|float): Timeout für den HTTP-Request

    Returns:
        stations (list): List of fuel stations with 'name', 'lat', 'lon', 'detour_distance_km', 'detour_duration_min', 'distance_along_m', 'eta'
    """

    # segment_coords = [[lon, lat], ... ] but polyline needs [(lat, lon), ...]
    latlon = [(lat, lon) for lon, lat in segment_coords]
    encoded_poly = polyline.encode(latlon, precision=5)
      # 5 is default for Google Maps API because they use 5 decimal places for lat/lon

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.displayName,places.location,"
            "routingSummaries.legs.distanceMeters,"
            "routingSummaries.legs.duration,"
            "nextPageToken"
        ),
    }
    base_body = {
        "textQuery": "gas station",
        "languageCode": "de",
        "searchAlongRouteParameters": {
            "polyline": {
                "encodedPolyline": encoded_poly
                }
            },
        "routingParameters": {
            "travelMode": "DRIVE",
            }
        }    

    # we'll collect results across pages
    stations = []
    page_token = None

    while True:
        
        body = dict(base_body)

        if page_token:
            # request the next page
            body["pageToken"] = page_token

        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_seconds)
            r.raise_for_status()
            print(f"\n Google Places Search Along Route status code: {r.status_code} Reason: {r.reason}")
            data = r.json()
        except requests.Timeout:
            raise TimeoutError(
                f"Google Places Search Along Route timed out after {timeout_seconds} seconds."
            )

        places = data.get("places", [])
        routing_summaries = data.get("routingSummaries", [])

        for place, routing in zip(places, routing_summaries):
            legs = routing.get("legs", [])

            # Leg 0: Origin -> A
            leg_OA = legs[0]
            D_OA = leg_OA["distanceMeters"]
            T_OA = parse_duration(leg_OA["duration"])

            # Leg 1: A -> Destination
            leg_AD = legs[1]
            D_AD = leg_AD["distanceMeters"]
            T_AD = parse_duration(leg_AD["duration"])

            # Origin -> A -> Destination
            D_OAD = D_OA + D_AD
            T_OAD = T_OA + T_AD

            # detour compared to original route
            detour_distance_km = D_OAD/1000.0 - original_distance_km
            detour_duration_min = T_OAD/60.0 - original_duration_min

            loc = place.get("location", {})
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            if lat is None or lon is None:
                continue

            eta = departure_time + timedelta(seconds = T_OA)

            stations.append({
                "name": place.get("displayName", {}).get("text") or "Unnamed",
                "lat": lat,
                "lon": lon,
                "detour_distance_km": detour_distance_km,
                "detour_duration_min": detour_duration_min,
                "distance_along_m": D_OA,
                "eta": eta.isoformat()
            })

        # pagination: check for nextPageToken and loop if present
        next_token = data.get("nextPageToken")
        if next_token:
            # Could maybe be deleted later
            time.sleep(0.2)
            page_token = next_token
            # continue to fetch next page
        else:
            break

    print(f"\n Found {len(stations)} stations (all pages).")

    return stations

# === Execution ===
def main():
    """
    Command-line / direct-run entrypoint for testing the route pipeline.
    When imported as a module (e.g. by Streamlit), this function is NOT executed.
    """
    # get ORS API key from environment variable
    google_api_key = environment_check()

    # get Geocode of start and end addresses
    start_lat, start_lon, start_label = google_geocode_structured(
        start_address, start_locality, start_country, google_api_key
    )
    end_lat, end_lon, end_label = google_geocode_structured(
        end_address, end_locality, end_country, google_api_key
    )

    # Check results of structured geocoding search
    print("\n--- Check GEOCODING RESULTS ---")
    print(f"Start: {start_label}")
    print(f" → lat = {start_lat:.6f}, lon = {start_lon:.6f}")
    print(f"End: {end_label}")
    print(f" → lat = {end_lat:.6f}, lon = {end_lon:.6f}")

    # get the route between two coordinates
    route_coords_lonlat, route_km, route_min, departure_time = google_route_driving_car(
        start_lat, start_lon, end_lat, end_lon, google_api_key
    )

    print("\n--- Check ROUTE ---")
    print(f"Route: {route_km:.1f} km")
    print(f"Duration: {route_min:.0f} min")
    print(f"Number of points unthinned list: {len(route_coords_lonlat)} coordinates along the route")

    # get data of fuel stations along the route
    stations_with_eta = google_places_fuel_along_route(route_coords_lonlat, google_api_key, route_km, route_min, departure_time)

    print("\n--- Stations with ETA ---")
    for s in stations_with_eta[:5]:
        print(f"{s.get('name')} → lat={s.get('lat'):.5f}, lon={s.get('lon'):.5f}, "
            f"detour_distance={s.get('detour_distance_km'):.3f} km, "
            f"detour_duration={s.get('detour_duration_min'):.2f} min, "
            f"distance from start to station={s.get('distance_along_m'):.0f} m, "
            f"ETA={s.get('eta')} \n"
              )

if __name__ == "__main__":
    main()
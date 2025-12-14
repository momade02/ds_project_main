"""
Description: This script uses the Google Maps Directions + Places APIs to:
 - geocode two addresses,
 - calculate a driving route between them,
 - and find gas stations along that route.

The functions `google_geocode_structured`, `google_route_driving_car`, and
`google_places_fuel_along_route` are imported by the main app / integration
pipeline.
"""

# test case long/short/switzerland (only used when running this file directly)
route_scenario = "short"  # "long", "short" or "switzerland"

# === Setup ===
import os
from pathlib import Path
import json
import requests
from shapely.geometry import LineString
from shapely.ops import transform, substring
import pyproj
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import googlemaps
import polyline
import time
from src.app.app_errors import ConfigError


# Hardcoded addresses for local testing only
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
elif route_scenario == "switzerland":
    start_address = "Zinngärten 9"
    start_locality = "Stühlingen"
    start_country = "Germany"
    end_address = "Lendenbergstrasse 32"
    end_locality = "Schleitheim"
    end_country = "Switzerland"
else:
    raise ValueError("Unknown route_scenario")

# Search parameter for the gas stations along the route (legacy, not used with Google)
buffer_meters = 300  # buffer width in meters


def environment_check() -> str:
    """Return the Google Maps API key from the environment.

    Side-effect free by design: no dotenv loading happens here.
    Load .env in the application entrypoint (Streamlit/CLI), not in helpers.
    """
    google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not google_api_key:
        raise ConfigError(
            user_message="Google Maps API key is not configured.",
            remediation="Set GOOGLE_MAPS_API_KEY in your environment (or load it via .env in the entrypoint) and restart.",
            details="Missing environment variable: GOOGLE_MAPS_API_KEY",
        )
    return google_api_key

def parse_duration(d: str) -> float:
    """
    Parse a duration string in the format "{number}s" and return the number as float.

    Parameters
    ----------
    d : str
        Duration string in the format "{number}s".

    Returns
    -------
    float
        Duration in seconds.
    """
    if not d.endswith("s"):
        raise ValueError(f"Unexpected duration format: {d}")
    return float(d[:-1])


# ---------------------------------------------------------------------------
# Google API helpers
# ---------------------------------------------------------------------------

def google_geocode_structured(street: str, city: str, country: str, api_key: str):
    """
    Geocode a structured address using googlemaps.Client.

    Parameters
    ----------
    street : str
        Street address.
    city : str
        City name.
    country : str
        Country name.
    api_key : str
        Google Maps API key.

    Returns
    -------
    (float, float, str)
        (lat, lon, label)
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


def google_route_driving_car(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    api_key: str,
    departure_time: str | datetime = "now",
):
    """
    Get driving route between two coordinates using Google Directions API.

    Parameters
    ----------
    start_lat, start_lon, end_lat, end_lon : float
        Start / end coordinates.
    api_key : str
        Google Maps API key.
    departure_time : "now" or datetime
        Desired departure time. If "now", the current local time in
        Europe/Berlin is used.

    Returns
    -------
    coords_lonlat : list[list[float, float]]
        Route polyline as [lon, lat] coordinates.
    distance_km : float
        Total distance of the route in kilometers.
    duration_min : float
        Total duration of the route in minutes.
    departure_time : datetime
        The (possibly normalised) departure time used for ETA calculations.
    """
    client = googlemaps.Client(key=api_key)

    directions = client.directions(
        origin=(start_lat, start_lon),
        destination=(end_lat, end_lon),
        mode="driving",
        alternatives=False,
        departure_time=departure_time,
        traffic_model="best_guess",
    )

    if not directions:
        raise ValueError("No route found by Google Directions API.")

    if departure_time == "now":
        departure_time = datetime.now(ZoneInfo("Europe/Berlin"))

    route = directions[0]

    # total distance (m) and duration (s)
    distance_m = sum(leg.get("distance", {}).get("value", 0) for leg in route.get("legs", []))
    duration_s = sum(leg.get("duration", {}).get("value", 0) for leg in route.get("legs", []))

    # prefer overview_polyline for geometry
    overview = route.get("overview_polyline", {}).get("points")
    points = googlemaps.convert.decode_polyline(overview)
    # convert to list of [lon, lat]
    coords_lonlat = [[p["lng"], p["lat"]] for p in points]

    distance_km = distance_m / 1000.0
    duration_min = duration_s / 60.0

    return coords_lonlat, distance_km, duration_min, departure_time


def google_places_fuel_along_route(
    segment_coords,
    api_key: str,
    original_distance_km: float,
    original_duration_min: float,
    departure_time: datetime,
    timeout_seconds: int | float = 30,
):
    """
    Get fuel stations along a route using Google Places "searchAlongRoute".

    Parameters
    ----------
    segment_coords : list[list[float, float]]
        Route coordinates as [lon, lat].
    api_key : str
        Google Maps API key.
    original_distance_km : float
        Original route distance in kilometers.
    original_duration_min : float
        Original route duration in minutes.
    departure_time : datetime
        Departure time (local, Europe/Berlin).
    timeout_seconds : int or float
        HTTP timeout for the Places request.

    Returns
    -------
    list[dict]
        List of fuel stations with keys:
            name, lat, lon,
            detour_distance_km, detour_duration_min,
            distance_along_m, fraction_of_route, eta
    """
    # segment_coords = [[lon, lat], ... ] but polyline needs [(lat, lon), ...]
    latlon = [(lat, lon) for lon, lat in segment_coords]
    encoded_poly = polyline.encode(latlon, precision=5)

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
            "polyline": {"encodedPolyline": encoded_poly}
        },
        "routingParameters": {
            "travelMode": "DRIVE",
        },
    }

    stations: list[dict] = []
    page_token = None

    while True:
        body = dict(base_body)
        if page_token:
            body["pageToken"] = page_token

        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout_seconds)
            r.raise_for_status()
            print(
                f"\n Google Places Search Along Route status code: "
                f"{r.status_code} Reason: {r.reason}"
            )
            data = r.json()
        except requests.Timeout:
            raise TimeoutError(
                f"Google Places Search Along Route timed out after {timeout_seconds} seconds."
            )

        places = data.get("places", [])
        routing_summaries = data.get("routingSummaries", [])

        for place, routing in zip(places, routing_summaries):
            legs = routing.get("legs", [])

            # Leg 0: Origin -> station A
            leg_OA = legs[0]
            D_OA = leg_OA["distanceMeters"]          # distance along route in metres
            T_OA = parse_duration(leg_OA["duration"])  # seconds

            # Leg 1: station A -> Destination
            leg_AD = legs[1]
            D_AD = leg_AD["distanceMeters"]
            T_AD = parse_duration(leg_AD["duration"])

            # Origin -> A -> Destination
            D_OAD = D_OA + D_AD
            T_OAD = T_OA + T_AD

            # Detour compared to original route
            detour_distance_km = D_OAD / 1000.0 - original_distance_km
            detour_duration_min = T_OAD / 60.0 - original_duration_min

            loc = place.get("location", {})
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            if lat is None or lon is None:
                continue

            eta = departure_time + timedelta(seconds=T_OA)

            # Fraction of route: position along route relative to original distance
            if original_distance_km and original_distance_km > 0:
                fraction_of_route = D_OA / (original_distance_km * 1000.0)
            else:
                fraction_of_route = None

            stations.append(
                {
                    "name": place.get("displayName", {}).get("text") or "Unnamed",
                    "lat": lat,
                    "lon": lon,
                    "detour_distance_km": detour_distance_km,
                    "detour_duration_min": detour_duration_min,
                    "distance_along_m": D_OA,
                    "fraction_of_route": fraction_of_route,
                    "eta": eta.isoformat(),
                }
            )

        # pagination
        next_token = data.get("nextPageToken")
        if next_token:
            time.sleep(0.2)  # brief pause before next page
            page_token = next_token
        else:
            break

    print(f"\n Found {len(stations)} stations (all pages).")
    return stations


# ---------------------------------------------------------------------------
# Local test entrypoint
# ---------------------------------------------------------------------------

def main():
    """
    Command-line / direct-run entrypoint for testing the route pipeline.
    When imported as a module (e.g. by Streamlit), this function is NOT executed.
    """

    # Local development convenience: load .env here (entrypoint only).
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    google_api_key = environment_check()

    # geocode start and end addresses
    start_lat, start_lon, start_label = google_geocode_structured(
        start_address, start_locality, start_country, google_api_key
    )
    end_lat, end_lon, end_label = google_geocode_structured(
        end_address, end_locality, end_country, google_api_key
    )

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
    print(
        "Number of points unthinned list: "
        f"{len(route_coords_lonlat)} coordinates along the route"
    )

    # get data of fuel stations along the route
    stations_with_eta = google_places_fuel_along_route(
        route_coords_lonlat, google_api_key, route_km, route_min, departure_time
    )

    print("\n--- Stations with ETA ---")
    for s in stations_with_eta[:5]:
        print(
            f"{s.get('name')} → lat={s.get('lat'):.5f}, lon={s.get('lon'):.5f}, "
            f"detour_distance={s.get('detour_distance_km'):.3f} km, "
            f"detour_duration={s.get('detour_duration_min'):.2f} min, "
            f"distance from start to station={s.get('distance_along_m'):.0f} m, "
            f"fraction_of_route={s.get('fraction_of_route'):.3f} "
            f"ETA={s.get('eta')}\n"
        )


if __name__ == "__main__":
    main()

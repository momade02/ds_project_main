"""
Module: Google Maps Integration for Route and Fuel Station Data.

Description:
    This module acts as the interface to the Google Maps Platform APIs (Directions,
    Geocoding, and Places). It is responsible for:
    1. Geocoding human-readable addresses into coordinates.
    2. Calculating driving routes between points to get geometry and metrics.
    3. Identifying gas stations along a specific route corridor using the
       Places API "Search Along Route" feature.
    4. calculating detour metrics (extra time and distance) required to visit
       those stations.

Usage:
    The functions defined here are intended to be imported by higher-level
    application logic (e.g., integration pipelines or UI handlers).
    Running this file directly executes a local test scenario.
"""

import os
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Any, TypeAlias, Final
from zoneinfo import ZoneInfo

import googlemaps  # type: ignore[import-untyped]
import polyline  # type: ignore[import-untyped]
import requests

# Custom internal error handling
from src.app.app_errors import ConfigError

# --- Type Definitions for Improved LLM Readability ---
# Coordinates format: [longitude, latitude] used by some GeoJSON structures
LonLatList: TypeAlias = list[float]
# Coordinates format: (latitude, longitude) used by Google Maps Client
LatLatTuple: TypeAlias = tuple[float, float]
# A list of coordinates forming a path
RoutePathLonLat: TypeAlias = list[LonLatList]

# Dictionary structure for station data returned by google_places_fuel_along_route
StationDataDict: TypeAlias = dict[str, Any]
# Dictionary structure for route-via-waypoint results
ViaRouteResultDict: TypeAlias = dict[str, Any]


# --- Configuration and Constants ---
# The timezone used for normalizing "now" in departure time calculations.
# This suggests the application is currently localized for Central Europe.
LOCAL_TIMEZONE: Final[ZoneInfo] = ZoneInfo("Europe/Berlin")

# API configuration
GOOGLE_PLACES_SEARCH_URL: Final[str] = "https://places.googleapis.com/v1/places:searchText"


# ==========================================
# Helper Functions
# ==========================================

def _is_open_at_eta(
    *,
    eta_dt: datetime,
    regular_opening_hours: dict[str, Any],
    utc_offset_minutes: int | None,
) -> bool | None:
    """
    Determine whether a place is open at a specific ETA.

    Notes:
      - We use `regularOpeningHours.periods` (weekly schedule) and (optionally) `utcOffsetMinutes`
        to evaluate openness at the station-local time.
      - If opening hours information is missing or cannot be parsed reliably, returns None.
    """
    periods = regular_opening_hours.get("periods")
    if not isinstance(periods, list) or not periods:
        return None

    # Convert ETA into place-local time if utcOffsetMinutes is available.
    try:
        eta_utc = eta_dt.astimezone(timezone.utc)
        if isinstance(utc_offset_minutes, int):
            eta_local = eta_utc + timedelta(minutes=utc_offset_minutes)
        else:
            eta_local = eta_dt  # fallback: assume ETA already in local timezone
    except Exception:
        return None

    # Google uses 0=Sunday, 1=Monday, ..., 6=Saturday in OpeningHours Points.
    py_weekday = eta_local.weekday()  # 0=Mon..6=Sun
    google_day = (py_weekday + 1) % 7
    eta_min_of_week = google_day * 24 * 60 + eta_local.hour * 60 + eta_local.minute

    def _point_to_min_of_week(point: dict[str, Any]) -> int | None:
        try:
            day = int(point.get("day"))
            hour = int(point.get("hour", 0))
            minute = int(point.get("minute", 0))
            return day * 24 * 60 + hour * 60 + minute
        except Exception:
            return None

    for period in periods:
        if not isinstance(period, dict):
            continue
        open_p = period.get("open")
        close_p = period.get("close")

        if not isinstance(open_p, dict):
            continue

        open_min = _point_to_min_of_week(open_p)
        if open_min is None:
            continue

        # 24/7 is represented as an open point at day=0,hour=0,minute=0 with no close field.
        if close_p is None:
            if open_min == 0:
                return True
            # Otherwise, treat as unknown rather than assuming always open.
            continue

        if not isinstance(close_p, dict):
            continue

        close_min = _point_to_min_of_week(close_p)
        if close_min is None:
            continue

        # Handle overnight/week-wrapping periods (close may be earlier than open).
        if close_min <= open_min:
            is_open = (eta_min_of_week >= open_min) or (eta_min_of_week < close_min)
        else:
            is_open = (open_min <= eta_min_of_week < close_min)

        if is_open:
            return True

    # If we could parse periods but ETA did not fall into any open interval, treat as closed.
    return False

def environment_check() -> str:
    """
    Retrieves and validates the Google Maps API key from the environment.

    Raises:
        ConfigError: If the GOOGLE_MAPS_API_KEY environment variable is unset.

    Returns:
        str: The valid Google Maps API key.
    """
    # NOTE: Dotenv loading happens at the application entrypoint, not here.
    google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not google_api_key:
        raise ConfigError(
            user_message="Google Maps API key is not configured.",
            remediation="Set GOOGLE_MAPS_API_KEY in environment.",
            details="Missing environment variable: GOOGLE_MAPS_API_KEY",
        )
    return google_api_key


def parse_duration(duration_str: str) -> float:
    """
    Parses a Google API duration string (e.g., "123s") into seconds.

    Google API often returns durations as strings terminated with 's'.

    Args:
        duration_str: String format "{number}s".

    Returns:
        float: Duration in seconds.

    Raises:
        ValueError: If format does not end in 's'.
    """
    if not duration_str.endswith("s"):
        raise ValueError(f"Unexpected duration format: {duration_str}")
    return float(duration_str[:-1])

def decode_route_steps_lonlat(route: dict[str, Any]) -> RoutePathLonLat:
    """
    Decodes all step polylines from a route and constructs the complete path geometry.

    Args:
        route: Dictionary containing the route response from Google Directions API,
               with 'legs' containing 'steps' with encoded polylines.

    Returns:
        RoutePathLonLat: List of [lon, lat] coordinate pairs forming the complete route.

    Raises:
        ValueError: If the route contains no decodable step polylines.
    """
    coords: RoutePathLonLat = []

    legs = route.get("legs", [])
    for leg in legs:
        steps = leg.get("steps", [])
        for step in steps:
            enc = step.get("polyline", {}).get("points")
            if not enc:
                continue

            decoded = googlemaps.convert.decode_polyline(enc)
            step_coords: RoutePathLonLat = [[p["lng"], p["lat"]] for p in decoded]

            # Dedup Übergangspunkt zwischen Steps
            if coords and step_coords and coords[-1] == step_coords[0]:
                step_coords = step_coords[1:]

            coords.extend(step_coords)

    if not coords:
        raise ValueError("No step polylines found in route response (steps-based geometry is empty).")

    return coords

def downsample_route_coords(
    coords_lonlat: RoutePathLonLat,
    max_points: int = 2500,
) -> RoutePathLonLat:
    """
    Downsample a route geometry to a fixed maximum number of points.

    Args:
        coords_lonlat: RoutePathLonLat list in [lon, lat] format.
        max_points: Maximum number of points to return.

    Returns:
        RoutePathLonLat: Downsampled list of [lon, lat] coordinate pairs.
    """
    n_points = len(coords_lonlat)

    # If the route is already small enough, return it as-is.
    if n_points <= max_points:
        return coords_lonlat

    # Generate uniformly spaced indices from 0 .. (n_points - 1).
    # The rounding ensures we get a fixed count close to max_points.
    idxs = [
        round(i * (n_points - 1) / (max_points - 1))
        for i in range(max_points)
    ]

    # Build the output list while removing any duplicate indices caused by rounding.
    downsampled: RoutePathLonLat = []
    last_idx: int | None = None
    for idx in idxs:
        if idx == last_idx:
            continue
        downsampled.append(coords_lonlat[idx])
        last_idx = idx

    return downsampled

# ==========================================
# Google Maps API Core Integration
# ==========================================

def google_geocode_structured(
    street: str, city: str, country: str, api_key: str
) -> tuple[float, float, str]:
    """
    Geocodes a structured address using the Google Maps Geocoding API.

    Args:
        street: Street address.
        city: City name.
        country: Country name.
        api_key: Valid Google Maps API key.

    Returns:
        tuple[float, float, str]: A tuple containing:
            - latitude (float)
            - longitude (float)
            - formatted_label (str): The resolved address label from Google.

    Raises:
        ValueError: If geocoding fails to yield results or coordinates.
    """
    client = googlemaps.Client(key=api_key)
    address_query = f"{street}, {city}, {country}"
    results = client.geocode(address_query)

    if not results:
        raise ValueError(f"No geocoding results for {address_query}")

    # Extract data from the top result
    top_result = results[0]
    location_data = top_result.get("geometry", {}).get("location", {})
    lat = location_data.get("lat")
    lon = location_data.get("lng")
    label = top_result.get("formatted_address", address_query)

    if lat is None or lon is None:
        raise ValueError(f"Geocoding returned missing coordinates for {address_query}")

    return lat, lon, label


def google_route_driving_car(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    api_key: str,
    departure_time: str | datetime = "now",
    region: str = "de",
) -> tuple[RoutePathLonLat, float, float, datetime]:
    """
    Calculates a driving route between origin and destination using Google Directions API.

    Used to establish the baseline route geometry and metrics before finding stations.

    Args:
        start_lat: Origin latitude.
        start_lon: Origin longitude.
        end_lat: Destination latitude.
        end_lon: Destination longitude.
        api_key: Valid Google Maps API key.
        departure_time: "now" or a specific datetime object. Defaults to "now".
        region: Two-letter country code (ccTLD) to bias results. Defaults to "de" (Germany).

    Returns:
        tuple containing:
        - RoutePathLonLat: List of [lon, lat] coordinates forming the route polyline.
        - float: Total distance in kilometers.
        - float: Total duration in minutes.
        - datetime: The actual departure time used for the calculation.

    Raises:
        ValueError: If Google Directions API finds no route.
    """
    client = googlemaps.Client(key=api_key)

    if departure_time == "now":
        departure_time_dt = datetime.now(LOCAL_TIMEZONE)
    else:
        # Assume input is already a valid datetime if not "now"
        departure_time_dt = departure_time # type: ignore

    # Call Directions API
    # Alternatives=False ensures we get a single best route.
    # traffic_model="best_guess" uses historical and live traffic data.
    directions_result = client.directions(
        origin=(start_lat, start_lon),
        destination=(end_lat, end_lon),
        mode="driving",
        alternatives=False,
        departure_time=departure_time_dt,
        traffic_model="best_guess",
        region=region,
    )

    if not directions_result:
        raise ValueError("No route found by Google Directions API.")

    route = directions_result[0]

    # Sum metrics across all legs of the route
    # Note: Google API returns distance in meters (value) and duration in seconds (value)
    distance_meters = sum(
        leg.get("distance", {}).get("value", 0) for leg in route.get("legs", [])
    )
    duration_seconds = sum(
        leg.get("duration", {}).get("value", 0) for leg in route.get("legs", [])
    )
    # validate distance and duration
    if distance_meters <= 0:
        raise ValueError("Route distance returned by Google Directions API is zero or negative.")
    if duration_seconds <= 0:
        raise ValueError("Route duration returned by Google Directions API is zero or negative.")

    coords_lonlat: RoutePathLonLat = decode_route_steps_lonlat(route)

    # Convert metrics to standard units (km and minutes)
    distance_km = distance_meters / 1000.0
    duration_min = duration_seconds / 60.0

    return coords_lonlat, distance_km, duration_min, departure_time_dt


def google_route_via_waypoint(
    start_lat: float,
    start_lon: float,
    waypoint_lat: float,
    waypoint_lon: float,
    end_lat: float,
    end_lon: float,
    api_key: str,
    departure_time: str | datetime = "now",
    region: str = "de",
) -> ViaRouteResultDict:
    """
    Calculates a driving route going specifically from Origin -> Waypoint -> Destination.

    Used to verify the exact path and metrics of stopping at a specific station.

    Args:
        start_lat, start_lon: Origin coordinates.
        waypoint_lat, waypoint_lon: Intermediate stop coordinates (e.g., a gas station).
        end_lat, end_lon: Final destination coordinates.
        api_key: Valid Google Maps API key.
        departure_time: "now" or datetime.
        region: Two-letter country code (ccTLD) to bias results. Defaults to "de" (Germany).

    Returns:
        ViaRouteResultDict: A dictionary with keys:
            - 'via_full_coords': RoutePathLonLat (polyline list)
            - 'via_distance_km': float (total distance)
            - 'via_duration_min': float (total duration)
            - 'departure_time': datetime used.
    """
    client = googlemaps.Client(key=api_key)

    if departure_time == "now":
        departure_time_dt = datetime.now(LOCAL_TIMEZONE)
    else:
        departure_time_dt = departure_time # type: ignore

    # Call Directions API with waypoints parameter
    directions_result = client.directions(
        origin=(start_lat, start_lon),
        destination=(end_lat, end_lon),
        waypoints=[(waypoint_lat, waypoint_lon)],
        mode="driving",
        alternatives=False,
        departure_time=departure_time_dt,
        traffic_model="best_guess",
        region=region,
    )

    if not directions_result:
        raise ValueError("No via-waypoint route found by Google Directions API.")

    route = directions_result[0]

    # Sum distance/duration over the two legs (Origin->Waypoint, Waypoint->Destination)
    total_distance_meters = sum(
        leg.get("distance", {}).get("value", 0) for leg in route.get("legs", [])
    )
    total_duration_seconds = sum(
        leg.get("duration", {}).get("value", 0) for leg in route.get("legs", [])
    )
    # validate distance and duration
    if total_distance_meters <= 0:
        raise ValueError("Via-waypoint route distance returned by Google Directions API is zero or negative.")
    if total_duration_seconds <= 0:
        raise ValueError("Via-waypoint route duration returned by Google Directions API is zero or negative.")
    
    full_coords_lonlat: RoutePathLonLat = decode_route_steps_lonlat(route)

    return {
        "via_full_coords": full_coords_lonlat,
        "via_distance_km": total_distance_meters / 1000.0,
        "via_duration_min": total_duration_seconds / 60.0,
        "departure_time": departure_time_dt,
    }


def google_places_fuel_along_route(
    segment_coords_lonlat: RoutePathLonLat,
    api_key: str,
    original_distance_km: float,
    original_duration_min: float,
    departure_time: datetime,
    timeout_seconds: int | float = 30,
) -> list[StationDataDict]:
    """
    Finds gas stations along a route corridor using Google Places API (New Version).

    This function uses the 'searchAlongRouteParameters' feature of the Places API v1.
    It automatically handles pagination to retrieve all results.

    Crucially, it calculates detour metrics based on the API's 'routingSummaries' response,
    which provides the time/distance required to deviate to the station and return to route.

    Args:
        segment_coords_lonlat: Input route path as list of [lon, lat].
        api_key: Google Maps API key.
        original_distance_km: Total distance of the direct route (baseline).
        original_duration_min: Total duration of the direct route (baseline).
        departure_time: Departure time used for ETA calculations.
        timeout_seconds: Max time for API requests. Defaults to 30.

    Returns:
        list[StationDataDict]: A list of dictionaries, each representing a station
        with location, detour metrics, ETA, and opening hours data.
    """

    # Downsample route geometry if too large for API limits
    segment_coords_lonlat = downsample_route_coords(segment_coords_lonlat, max_points=5000)

    # 1. Prepare Geometry: Convert [lon, lat] list to [(lat, lon)] tuple list for encoding
    latlon_tuples: list[LatLatTuple] = [
        (lat, lon) for lon, lat in segment_coords_lonlat
    ]
    # Encode geometry into polyline string format required by Places API
    encoded_poly_str = polyline.encode(latlon_tuples, precision=5)

    # 2. Prepare API Request
    # Using the new Places API v1 endpoint
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        # FieldMask dictates exactly which fields the API returns to save bandwidth.
        # We ask for display name, location, hours, routing summaries (critical for detour calc), and next page token.
        "X-Goog-FieldMask": (
            "places.displayName,places.location,"
            "places.regularOpeningHours,places.utcOffsetMinutes,"
            "routingSummaries.legs.distanceMeters,"
            "routingSummaries.legs.duration,"
            "nextPageToken"
        ),
    }

    # Base request body for searchAlongRoute
    base_request_body = {
        "textQuery": "gas station",
        "languageCode": "de",  # Prefer German results based on context
        "searchAlongRouteParameters": {
            "polyline": {"encodedPolyline": encoded_poly_str}
        },
        "routingParameters": {
            "travelMode": "DRIVE",
        },
    }

    stations_found: list[StationDataDict] = []
    page_token: str | None = None

    # 3. Execute Search with Pagination Loop
    while True:
        current_request_body = dict(base_request_body)
        if page_token:
            current_request_body["pageToken"] = page_token

        try:
            response = requests.post(
                GOOGLE_PLACES_SEARCH_URL,
                headers=headers,
                json=current_request_body,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as e:
            raise TimeoutError(
                f"Google Places Search timed out after {timeout_seconds}s"
            ) from e
        except requests.RequestException as e:
            print(f"API Request Error: {e}")
            # Break loop on non-transient errors, returning whatever was found so far
            break

        data_json = response.json()
        places_list = data_json.get("places", [])
        # routingSummaries corresponds 1-to-1 with places_list
        routing_summaries_list = data_json.get("routingSummaries", [])

        # 4. Process Results in current page
        for place, routing in zip(places_list, routing_summaries_list):
            # Extract location
            loc_data = place.get("location", {})
            lat = loc_data.get("latitude")
            lon = loc_data.get("longitude")

            # Skip if crucial coordinate data is missing
            if lat is None or lon is None:
                continue

            # Extract routing logic for detour calculation
            # The API returns 'legs' describing the detour:
            # Leg 0: Origin -> Station
            # Leg 1: Station -> Destination
            legs = routing.get("legs", [])
            if len(legs) < 2:
                continue

            leg_origin_to_station = legs[0]
            leg_station_to_dest = legs[1]

            # Metrics: Origin -> Station
            dist_meters_oa = leg_origin_to_station.get("distanceMeters", 0)
            duration_seconds_oa = parse_duration(
                leg_origin_to_station.get("duration", "0s")
            )

            # Metrics: Station -> Destination
            dist_meters_ad = leg_station_to_dest.get("distanceMeters", 0)
            duration_seconds_ad = parse_duration(
                leg_station_to_dest.get("duration", "0s")
            )

            if dist_meters_oa <= 0 or dist_meters_ad <= 0:
                raise ValueError("Invalid detour distances returned by Places API.")
            if duration_seconds_oa <= 0 or duration_seconds_ad <= 0:
                raise ValueError("Invalid detour durations returned by Places API.")

            # Total metrics for the route passing through the station (O -> A -> D)
            total_dist_meters_oad = dist_meters_oa + dist_meters_ad
            total_duration_seconds_oad = duration_seconds_oa + duration_seconds_ad

            # Calculate Detour Cost: (Route via station) - (Original baseline route)
            # Convert to km and minutes for final output.
            detour_distance_km = (
                total_dist_meters_oad / 1000.0
            ) - original_distance_km
            detour_duration_min = (
                total_duration_seconds_oad / 60.0
            ) - original_duration_min

            # Calculate Estimated Time of Arrival at the station
            eta_at_station = departure_time + timedelta(seconds=duration_seconds_oa)

            # Calculate progression along route (fraction 0.0 to 1.0) based on distance to station
            fraction_of_route = 0.0
            
            fraction_of_route = (dist_meters_oa / 1000.0) / original_distance_km

            # Extract opening hours data
            opening_hours_data = place.get("regularOpeningHours", {})
            utc_offset_minutes = place.get("utcOffsetMinutes")
            is_open_at_eta = _is_open_at_eta(
                eta_dt=eta_at_station,
                regular_opening_hours=opening_hours_data,
                utc_offset_minutes=utc_offset_minutes if isinstance(utc_offset_minutes, int) else None,
            )

            # Build final station dictionary
            station_dict: StationDataDict = {
                "name": place.get("displayName", {}).get("text") or "Unnamed",
                "lat": lat,
                "lon": lon,
                "detour_distance_km": detour_distance_km,
                "detour_duration_min": detour_duration_min,
                "distance_along_m": dist_meters_oa,
                "fraction_of_route": fraction_of_route,
                "eta": eta_at_station.isoformat(),
                "open_now": opening_hours_data.get("openNow"),
                "opening_hours": opening_hours_data.get("weekdayDescriptions", []),
                "opening_periods": opening_hours_data.get("periods", []),
                "utc_offset_minutes": utc_offset_minutes,
                "is_open_at_eta": is_open_at_eta,
            }
            stations_found.append(station_dict)

        # 5. Handle Pagination
        next_token = data_json.get("nextPageToken")
        if next_token:
            time.sleep(0.2)  # Rate limiting pause recommended by Google
            page_token = next_token
        else:
            # No more pages
            break

    print(f"Google Places: Found {len(stations_found)} stations total.")
    return stations_found


# ==========================================
# Local Test Entrypoint
# ==========================================

if __name__ == "__main__":
    # This block only runs when executing the file directly for local testing.
    # It is NOT executed when imported by other modules like the Streamlit app.

    # --- Test Configuration ---
    # Select scenario: "short", "long", or "switzerland"
    TEST_SCENARIO = "long"

    # Hardcoded test addresses based on scenario selection
    if TEST_SCENARIO == "short":
        START_ADDR = {"street": "Wilhelmstraße 7", "city": "Tübingen", "country": "Germany"}
        END_ADDR = {"street": "Charlottenstraße 45", "city": "Reutlingen", "country": "Germany"}
    elif TEST_SCENARIO == "long":
        START_ADDR = {"street": "Wilhelmstraße 7", "city": "Tübingen", "country": "Germany"}
        END_ADDR = {"street": "", "city": "Berlin", "country": "Germany"}
    elif TEST_SCENARIO == "switzerland":
        START_ADDR = {"street": "", "city": "Stühlingen", "country": "Germany"}
        END_ADDR = {"street": "", "city": "Schleitheim", "country": "Switzerland"}
    else:
        raise ValueError(f"Unknown test scenario: {TEST_SCENARIO}")

    # --- Test Execution ---
    try:
        # Attempt to load .env for local development convenience
        from dotenv import find_dotenv, load_dotenv  # type: ignore

        load_dotenv(find_dotenv(usecwd=True), override=False)
        print("Local .env loaded.")
    except ImportError:
        print("Skipping .env loading (python-dotenv not installed).")
    except Exception as e:
        print(f"Error loading .env: {e}")

    # 1. Setup & Geocoding
    try:
        google_key = environment_check()

        print("\n--- Geocoding Addresses ---")
        s_lat, s_lon, s_label = google_geocode_structured(
            START_ADDR["street"], START_ADDR["city"], START_ADDR["country"], google_key
        )
        print(f"Start: {s_label} ({s_lat:.6f}, {s_lon:.6f})")

        e_lat, e_lon, e_label = google_geocode_structured(
            END_ADDR["street"], END_ADDR["city"], END_ADDR["country"], google_key
        )
        print(f"End:   {e_label} ({e_lat:.6f}, {e_lon:.6f})")

        # 2. Get Baseline Route
        print("\n--- Calculating Baseline Route ---")
        route_coords, route_km, route_min, dept_time = google_route_driving_car(
            s_lat, s_lon, e_lat, e_lon, google_key
        )
        print(f"Route Distance: {route_km:.1f} km")
        print(f"Route Duration: {route_min:.0f} min")
        print(f"Route Geometry Points: {len(route_coords)}")

        # 3. Find Stations Along Route
        print("\n--- Searching for Stations Along Route ---")
        # Note: The timeout can be increased for very long routes if necessary.
        stations_results = google_places_fuel_along_route(
            route_coords, google_key, route_km, route_min, dept_time, timeout_seconds=45
        )

        # 4. Print Sample Results
        print("\n--- Top 5 Stations found (Sample Info) ---")
        for i, s in enumerate(stations_results[:5]):
            print(f"[{i+1}] {s.get('name')}")
            print(f"    Coords: {s.get('lat'):.5f}, {s.get('lon'):.5f}")
            print(
                f"    Detour: +{s.get('detour_distance_km'):.3f}km, "
                f"+{s.get('detour_duration_min'):.2f}min"
            )
            print(
                f"    Progress: {s.get('distance_along_m'):.0f}m along route "
                f"({s.get('fraction_of_route'):.2%})"
            )
            print(f"    ETA: {s.get('eta')}")
            print(f"    Open Now: {s.get('open_now')}")
            print("-" * 40)

    except ConfigError as ce:
        print(f"\nConfiguration Error: {ce.user_message}")
        print(f"Remediation: {ce.remediation}")
    except Exception as ex:
        print(f"\nAn unexpected error occurred during testing: {ex}")
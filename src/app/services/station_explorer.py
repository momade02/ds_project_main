# src/app/services/station_explorer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

from app_errors import ConfigError, ExternalServiceError, DataAccessError

# Reuse your existing integration utilities (stations cache + realtime price batching).
# We intentionally DO NOT modify integration code for Step 8.
from src.integration import route_tankerkoenig_integration as tkint


@dataclass(frozen=True)
class StationExplorerInputs:
    location_query: str
    fuel_code: str = "e5"              # "e5" | "e10" | "diesel"
    radius_km: float = 10.0
    limit: int = 200                   # cap results for UI
    country: str = "Germany"
    use_realtime: bool = True
    only_open: bool = False


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _geocode_location(location_query: str, country: str) -> Tuple[float, float, str]:
    """
    Geocode a free-form query using the same structured approach as your route pipeline.
    """
    if not location_query.strip():
        raise ExternalServiceError("Geocoding failed.", details="Empty location query.")

    # route_tankerkoenig_integration loads GOOGLE_API_KEY via environment_check if available
    geocode_fn = getattr(tkint, "google_geocode_structured", None)
    envcheck_fn = getattr(tkint, "environment_check", None)

    if geocode_fn is None or envcheck_fn is None:
        raise ConfigError(
            "Google geocoding is not available.",
            details="route_stations.py / Google functions not available in integration module.",
        )

    try:
        api_key = envcheck_fn()  # same as the rest of your app
    except Exception as e:
        raise ConfigError("Google API key missing or invalid.", details=str(e))

    # For structured geocode, split query into (address, locality) heuristically.
    # We keep this simple: user can provide "Street 1, City" or just "City" or "ZIP City".
    # We treat everything before the first comma as 'address-ish', remainder as locality.
    raw = location_query.strip()
    if "," in raw:
        left, right = [p.strip() for p in raw.split(",", 1)]
        address = left
        locality = right
    else:
        address = ""
        locality = raw

    try:
        lat, lon, label = geocode_fn(address, locality, country, api_key)
        return float(lat), float(lon), str(label)
    except Exception as e:
        raise ExternalServiceError("Geocoding failed.", details=str(e))


def search_stations_nearby(inputs: StationExplorerInputs) -> Dict[str, Any]:
    """
    Minimal station explorer search:
      - geocode to center,
      - load stations master list from Supabase (cached in integration module),
      - filter by radius,
      - optionally enrich with realtime prices from Tankerkönig (batched),
      - return station dicts compatible with the UI + map module.

    Returns dict with:
      - center: {lat, lon, label, radius_km}
      - stations: List[station_dict]
    """
    if inputs.radius_km <= 0:
        raise DataAccessError("Invalid radius.", details=f"radius_km={inputs.radius_km}")

    lat0, lon0, label = _geocode_location(inputs.location_query, inputs.country)

    # Load stations from Supabase (already cached in integration)
    try:
        df = tkint.load_all_stations_from_supabase()
    except Exception as e:
        raise DataAccessError("Failed to load station master data.", details=str(e))

    # Compute distances and filter
    stations: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except Exception:
            continue

        d_km = _haversine_km(lat0, lon0, lat, lon)
        if d_km <= float(inputs.radius_km):
            stations.append(
                {
                    # Keys expected by your UI/map stack
                    "station_uuid": str(row.get("uuid") or ""),
                    "tk_name": row.get("name"),
                    "brand": row.get("brand"),
                    "city": row.get("city"),
                    "lat": lat,
                    "lon": lon,
                    # Explorer-specific
                    "distance_km": float(d_km),
                    # Price placeholders (filled by realtime below if enabled)
                    "price_current_e5": None,
                    "price_current_e10": None,
                    "price_current_diesel": None,
                    "is_open": None,
                }
            )

    # Cap raw list early (avoid huge UI payload)
    stations.sort(key=lambda s: s["distance_km"])
    stations = stations[: max(1, int(inputs.limit))]

    # Realtime enrichment (optional)
    if inputs.use_realtime:
        ids = [s["station_uuid"] for s in stations if s.get("station_uuid")]
        try:
            live = tkint.get_realtime_prices_batch(ids)
        except Exception as e:
            # Non-fatal: keep stations without realtime prices
            live = {}
        for s in stations:
            uid = s.get("station_uuid")
            info = live.get(uid, {}) if uid else {}
            s["price_current_e5"] = info.get("e5")
            s["price_current_e10"] = info.get("e10")
            s["price_current_diesel"] = info.get("diesel")
            s["is_open"] = info.get("is_open")

    # Optional open-only filter
    if inputs.only_open:
        stations = [s for s in stations if s.get("is_open") is True]

    # Sorting policy:
    # - If we have current prices for chosen fuel_code, sort by price then distance.
    # - Otherwise sort by distance.
    price_key = f"price_current_{inputs.fuel_code}"
    def _sort_key(s: Dict[str, Any]) -> Tuple[int, float, float]:
        p = s.get(price_key)
        if p is None:
            return (1, 0.0, s["distance_km"])  # missing price goes last
        try:
            return (0, float(p), s["distance_km"])
        except Exception:
            return (1, 0.0, s["distance_km"])

    stations.sort(key=_sort_key)

    return {
        "center": {"lat": lat0, "lon": lon0, "label": label, "radius_km": float(inputs.radius_km)},
        "stations": stations,
    }


def pick_best_station_uuid(stations: List[Dict[str, Any]], fuel_code: str) -> Optional[str]:
    """
    Choose a 'best' station for highlighting in Explorer.
    Criterion: lowest current price for selected fuel if available, else nearest distance.
    """
    if not stations:
        return None

    price_key = f"price_current_{fuel_code}"

    priced = []
    for s in stations:
        p = s.get(price_key)
        if p is None:
            continue
        try:
            priced.append((float(p), float(s.get("distance_km", 1e9)), s))
        except Exception:
            continue

    if priced:
        priced.sort(key=lambda x: (x[0], x[1]))
        best = priced[0][2]
        return best.get("station_uuid") or best.get("tk_uuid")

    # Fallback: nearest
    stations_sorted = sorted(stations, key=lambda s: float(s.get("distance_km", 1e9)))
    best = stations_sorted[0]
    return best.get("station_uuid") or best.get("tk_uuid")

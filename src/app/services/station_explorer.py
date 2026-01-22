"""
MODULE: Station Explorer Service — Location-Centered Station Search for Page 04
------------------------------------------------------------------------------

Purpose
- Implements the data acquisition and normalization layer for the Station Explorer workflow:
  user provides a center location + radius + fuel type → service returns nearby stations with
  optional realtime prices and filter/sort semantics compatible with the UI.

What this module does
- Input contract:
  - Defines `StationExplorerInputs` capturing location query, fuel code, radius, UI result cap (`limit`),
    realtime toggle, open-only toggle, and an optional brand filter list.
- Geocoding:
  - Resolves the user’s free-form location query into a center coordinate using integration helpers
    (Google structured geocode via the integration module).
- Two retrieval strategies (same output shape):
  1) `search_stations_nearby(...)` (Supabase-based):
     - Loads a station master list (cached in integration), computes haversine distance, filters by radius,
       optionally enriches via batched realtime requests, then sorts for UI.
  2) `search_stations_nearby_list_api(...)` (Tankerkönig list.php):
     - Uses Tankerkönig’s list endpoint to fetch stations + realtime prices + isOpen in one call,
       applies filtering/sorting locally, then enforces `limit` *after* filtering/sorting.
- Filtering & sorting semantics:
  - `only_open=True` => station must be open AND have a valid current price for the selected fuel.
  - Optional brand filtering uses canonical-to-alias normalization to match known naming variants.
  - Sorting defaults to “cheapest current price for selected fuel, then distance”; missing prices go last.
- Explorer highlighting:
  - `pick_best_station_uuid(...)` selects the station to highlight on the map (cheapest price if available,
    otherwise nearest station).

Outputs
- Dict payload compatible with Page 04 session state:
  - `center`: {lat, lon, label, radius_km}
  - `stations`: List[station_dict] with normalized keys used by maps/UI (lat/lon, distance_km,
    `price_current_<fuel>`, `is_open`, identity fields).

Design constraints
- Must be best-effort and robust to partial API responses (normalize prices, tolerate missing fields).
- Must keep output key names aligned with the map and table renderers.
- External requests must be bounded with timeouts and should degrade gracefully to “no realtime” behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import requests

from app_errors import ConfigError, ExternalServiceError, DataAccessError

# Reuse existing integration utilities (stations cache + realtime price batching).
from src.integration import route_tankerkoenig_integration as tkint

@dataclass(frozen=True)
class StationExplorerInputs:
    location_query: str
    fuel_code: str = "e5"              # "e5" | "e10" | "diesel"
    radius_km: float = 10.0
    limit: int = 50                    # cap results for UI (Page 04 default)
    country: str = "Germany"
    use_realtime: bool = True
    only_open: bool = False
    brand_filter_selected: Optional[List[str]] = None


# ---------------------------------------------------------------------
# Brand filter normalization (kept aligned with Trip Planner logic)
# ---------------------------------------------------------------------

BRAND_FILTER_ALIASES: dict[str, list[str]] = {
    "ARAL": ["ARAL"],
    "AVIA": ["AVIA", "AVIA XPress", "AVIA Xpress"],
    "AGIP ENI": ["Agip", "AGIP ENI"],
    "Shell": ["Shell"],
    "Total": ["Total", "TotalEnergies"],
    "ESSO": ["ESSO"],
    "JET": ["JET"],
    "ORLEN": ["ORLEN"],
    "HEM": ["HEM"],
    "OMV": ["OMV"],
}


def _normalize_brand(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    # Case-insensitive + whitespace normalization
    return " ".join(s.upper().split())


def _allowed_brand_norm_set(selected: Optional[List[str]]) -> set[str]:
    """
    Convert a list of canonical brands into the normalized allowed set,
    including their known aliases.
    """
    allowed: set[str] = set()
    for canon in (selected or []):
        canon_s = str(canon).strip()
        if not canon_s:
            continue
        aliases = BRAND_FILTER_ALIASES.get(canon_s, [canon_s])
        for a in aliases:
            allowed.add(_normalize_brand(a))
    return allowed


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
        except Exception:
            # Best-effort: if realtime fails, still return stations (prices remain None).
            live = {}
        for s in stations:
            uid = s.get("station_uuid")
            info = live.get(uid, {}) if uid else {}
            s["price_current_e5"] = info.get("e5")
            s["price_current_e10"] = info.get("e10")
            s["price_current_diesel"] = info.get("diesel")
            s["is_open"] = info.get("is_open")

    # Sorting policy:
    # - If we have current prices for chosen fuel_code, sort by price then distance.
    # - Otherwise sort by distance.
    price_key = f"price_current_{inputs.fuel_code}"

    def _has_current_price(st: Dict[str, Any]) -> bool:
        v = (st or {}).get(price_key)
        if v is None:
            return False
        try:
            return float(v) > 0
        except (TypeError, ValueError):
            return False

    # Optional filter (Page 04 semantics):
    # "Only show open stations & stations with realtime data"
    # => must be open AND have a current price for the selected fuel.
    if inputs.only_open:
        stations = [
            s for s in stations
            if (s.get("is_open") is True) and _has_current_price(s)
        ]

    # Optional brand filter (Explorer): canonical list + alias matching
    if inputs.brand_filter_selected:
        allowed_norm = _allowed_brand_norm_set(inputs.brand_filter_selected)
        if allowed_norm:
            stations = [
                s for s in stations
                if _normalize_brand((s or {}).get("brand")) in allowed_norm
            ]

    def _sort_key(s: Dict[str, Any]) -> Tuple[int, float, float]:
        p = (s or {}).get(price_key)
        if p is None:
            return (1, 0.0, float(s.get("distance_km") or 0.0))  # missing price goes last
        try:
            return (0, float(p), float(s.get("distance_km") or 0.0))
        except (TypeError, ValueError):
            return (1, 0.0, float(s.get("distance_km") or 0.0))

    stations.sort(key=_sort_key)

    return {
        "center": {"lat": lat0, "lon": lon0, "label": label, "radius_km": float(inputs.radius_km)},
        "stations": stations,
    }


def search_stations_nearby_list_api(inputs: StationExplorerInputs) -> Dict[str, Any]:
    """
    Page-04-optimized station explorer search using Tankerkönig API Method 1 (list.php).

    Behavior:
      - Geocode the center (same as the rest of your app).
      - Call Tankerkönig list.php once to fetch stations + realtime prices + isOpen.
      - Apply filtering/sorting in Python (so "limit" is enforced AFTER filtering/sorting).
      - Return the same payload shape as search_stations_nearby() so Page 04 session state remains stable.

    Notes:
      - list.php supports a maximum radius of 25 km. If the user exceeds this, we fall back to the
        Supabase-based implementation (search_stations_nearby), which already supports larger radii.
      - If inputs.use_realtime is False, we also fall back to search_stations_nearby(inputs) to preserve
        the existing "no realtime" behavior (prices and open state remain None).
    """
    # Preserve existing behavior when realtime is disabled
    if not inputs.use_realtime:
        return search_stations_nearby(inputs)

    if inputs.radius_km <= 0:
        raise DataAccessError("Invalid radius.", details=f"radius_km={inputs.radius_km}")

    # list.php hard limit (official): max 25 km. Page 04 UI already enforces this, but keep it defensive.
    if float(inputs.radius_km) > 25.0:
        return search_stations_nearby(inputs)

    lat0, lon0, label = _geocode_location(inputs.location_query, inputs.country)

    api_key = getattr(tkint, "TANKERKOENIG_API_KEY", None)
    if not api_key:
        raise ConfigError(
            "Tankerkönig API key missing.",
            details="Set TANKERKOENIG_API_KEY in your environment (.env / hosting settings).",
        )

    url = "https://creativecommons.tankerkoenig.de/json/list.php"
    params = {
        "lat": float(lat0),
        "lng": float(lon0),
        "rad": float(inputs.radius_km),
        "type": "all",
        # For type=all, Tankerkönig sorts by distance anyway; keep it explicit.
        "sort": "dist",
        "apikey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ExternalServiceError("Tankerkönig request failed.", details=str(e))

    if not data.get("ok"):
        raise ExternalServiceError("Tankerkönig request returned ok=false.", details=str(data.get("message") or data))

    raw_stations = data.get("stations") or []
    if not isinstance(raw_stations, list):
        raise ExternalServiceError("Unexpected Tankerkönig response shape.", details="'stations' is not a list.")

    def _norm_price(v: Any) -> Optional[float]:
        """Normalize Tankerkönig price fields: False/None/non-numeric/<=0 -> None."""
        if v is None or v is False:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(fv) or fv <= 0:
            return None
        return fv

    stations: List[Dict[str, Any]] = []
    for stn in raw_stations:
        if not isinstance(stn, dict):
            continue

        # Coordinates
        try:
            lat = float(stn.get("lat"))
            lon = float(stn.get("lng"))
        except (TypeError, ValueError):
            continue

        # Required id
        uid = str(stn.get("id") or "").strip()
        if not uid:
            continue

        # Distance returned by list.php (km)
        dist_v = stn.get("dist")
        try:
            distance_km = float(dist_v) if dist_v is not None else float(_haversine_km(lat0, lon0, lat, lon))
        except (TypeError, ValueError):
            distance_km = float(_haversine_km(lat0, lon0, lat, lon))

        is_open_v = stn.get("isOpen")
        is_open: Optional[bool]
        if isinstance(is_open_v, bool):
            is_open = is_open_v
        else:
            is_open = None

        # Build station record compatible with your UI/map stack
        stations.append(
            {
                "station_uuid": uid,
                "tk_name": stn.get("name"),
                "brand": stn.get("brand"),
                "city": stn.get("place") or "",
                "place": stn.get("place") or "",
                # Address parts for popups/tooltips
                "street": stn.get("street"),
                "houseNumber": stn.get("houseNumber"),
                "postCode": stn.get("postCode"),
                "lat": lat,
                "lon": lon,
                "distance_km": float(distance_km),
                "price_current_e5": _norm_price(stn.get("e5")),
                "price_current_e10": _norm_price(stn.get("e10")),
                "price_current_diesel": _norm_price(stn.get("diesel")),
                "is_open": is_open,
            }
        )

    # Page 04 semantics (Option A):
    # If only_open is enabled => station must be open AND have a current price for selected fuel.
    price_key = f"price_current_{inputs.fuel_code}"

    def _has_current_price(st: Dict[str, Any]) -> bool:
        v = (st or {}).get(price_key)
        if v is None:
            return False
        try:
            return float(v) > 0
        except (TypeError, ValueError):
            return False

    if inputs.only_open:
        stations = [
            s for s in stations
            if (s.get("is_open") is True) and _has_current_price(s)
        ]

    # Optional brand filter (Explorer): canonical list + alias matching
    if inputs.brand_filter_selected:
        allowed_norm = _allowed_brand_norm_set(inputs.brand_filter_selected)
        if allowed_norm:
            stations = [
                s for s in stations
                if _normalize_brand((s or {}).get("brand")) in allowed_norm
            ]

    # Sort by selected fuel price (if available), then distance; missing prices go last
    def _sort_key(s: Dict[str, Any]) -> Tuple[int, float, float]:
        p = (s or {}).get(price_key)
        if p is None:
            return (1, 0.0, float(s.get("distance_km") or 0.0))
        try:
            return (0, float(p), float(s.get("distance_km") or 0.0))
        except (TypeError, ValueError):
            return (1, 0.0, float(s.get("distance_km") or 0.0))

    stations.sort(key=_sort_key)

    # Enforce limit AFTER filtering/sorting (as requested)
    if inputs.limit is not None:
        stations = stations[: max(1, int(inputs.limit))]

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

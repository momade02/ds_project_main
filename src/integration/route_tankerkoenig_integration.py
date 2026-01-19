"""
Module: Route â†” Fuel Price Integration Pipeline.

Description:
    This module bridges the Geospatial domain (Google Maps Routes) with the
    Economic domain (Tankerkönig Fuel Prices).

    It performs the following high-level transformation:
    [Route Coordinates & ETAs]
           ↔
    [Match to Tankerkönig Stations (Spatial KD-Tree)]
           ↔
    [Enrich with Historical Prices (Supabase Batch Queries)]
           ↔
    [Enrich with Real-time Prices (Optional API Call)]
           ↔
    [Feature Engineering (Time Cells, Lags)]
           ↔
    [Model-Ready Feature Vectors]

Usage:
    The primary entrypoint is `get_fuel_prices_for_route(...)`.
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Final, Set

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree  # type: ignore

# Custom Error Handling
from src.app.app_errors import (
    ConfigError,
    DataAccessError,
    DataQualityError,
    ExternalServiceError,
)

# --- Dynamic Import Setup for route_stations.py ---
# Ensures we can import the sibling module regardless of execution context
try:
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    if "src" in _PROJECT_ROOT:
        _PROJECT_ROOT = os.path.dirname(os.path.dirname(_PROJECT_ROOT))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from route_stations import (
        environment_check,
        google_geocode_structured,
        google_places_fuel_along_route,
        google_route_driving_car,
    )

    _ROUTE_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("WARNING: 'route_stations.py' not found. Integration features disabled.")
    _ROUTE_FUNCTIONS_AVAILABLE = False


# ==========================================
# Type Definitions (LLM Context Hints)
# ==========================================

# A raw station matched from Google/OSM logic
InputStationDict: TypeAlias = Dict[str, Any]

# A dictionary containing price data: {'e5': 1.70, 'e10': ..., 'diesel': ...}
FuelPriceDict: TypeAlias = Dict[str, Optional[float]]

# A nested dictionary for historical lags: {'1d': FuelPriceDict, '7d': ...}
PriceHistoryDict: TypeAlias = Dict[str, FuelPriceDict]

# The final, fully enriched dictionary ready for the prediction model
StationFeatureDict: TypeAlias = Dict[str, Any]


# ==========================================
# Configuration & Constants
# ==========================================

# Supabase Credentials
SUPABASE_URL: Final[Optional[str]] = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY: Final[Optional[str]] = os.getenv("SUPABASE_SECRET_KEY")

# Tankerkönig API (Real-time only)
TANKERKOENIG_API_KEY: Final[Optional[str]] = os.getenv("TANKERKOENIG_API_KEY")
TK_API_BATCH_SIZE: Final[int] = 10
TK_API_DELAY_SEC: Final[float] = 0.5

# Caching Configuration
CACHE_DURATION_SEC: Final[int] = 3600
_CACHED_STATIONS_DF: Optional[pd.DataFrame] = None
_CACHE_TIMESTAMP: Optional[float] = None

# Google API Key Loading
try:
    if _ROUTE_FUNCTIONS_AVAILABLE:
        GOOGLE_API_KEY = environment_check()
    else:
        GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY") or ""
except Exception:
    GOOGLE_API_KEY = ""


# ==========================================
# Section 1: Feature Engineering (Time)
# ==========================================

def calculate_time_cell(eta_string: str) -> int:
    """
    Converts an ISO timestamp into a 30-minute 'Time Cell' index (0-47).

    Logic:
        Cell 0  = 00:00 - 00:29
        Cell 1  = 00:30 - 00:59
        ...
        Cell 47 = 23:30 - 23:59

    Formula:
        cell = hour * 2 + (1 if minute >= 30 else 0)

    Args:
        eta_string: ISO 8601 string (e.g., "2025-11-20T14:35:00")

    Returns:
        int: The integer cell index.
    """
    try:
        # Robust parsing: remove timezone offsets if present to focus on local time
        clean_str = eta_string
        if "+" in clean_str:
            clean_str = clean_str.split("+")[0]
        elif clean_str.count("-") > 2:  # Negative offset case
            parts = clean_str.split("-")
            clean_str = "-".join(parts[:3])

        if "." in clean_str:
            dt = datetime.strptime(clean_str, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            dt = datetime.strptime(clean_str, "%Y-%m-%dT%H:%M:%S")

        return dt.hour * 2 + (1 if dt.minute >= 30 else 0)

    except ValueError as e:
        print(f"WARNING: Time parse error for '{eta_string}': {e}")
        return 0


def get_cell_time_range(cell: int) -> Tuple[str, str]:
    """
    Returns the (start_time, end_time) strings for a given time cell.
    Used for database queries to find prices valid within that window.

    Args:
        cell: 0-47 index.

    Returns:
        Tuple[str, str]: ("HH:MM:00", "HH:MM:59")
    """
    hour = cell // 2
    minute = 30 if cell % 2 == 1 else 0
    return (
        f"{hour:02d}:{minute:02d}:00",
        f"{hour:02d}:{minute+29:02d}:59",
    )


# ==========================================
# Section 2: Spatial Matching (KD-Tree)
# ==========================================

def match_coordinates_progressive(
    target_lat: float,
    target_lon: float,
    stations_df: pd.DataFrame,
    thresholds_meters: List[int] = [25, 50, 100],
) -> Optional[Dict[str, Any]]:
    """
    Finds the nearest Tankerkönig station using a KD-Tree spatial index.

    Optimization:
        The KD-Tree is built once and cached as an attribute `_spatial_index`
        on the DataFrame object itself. This makes lookups O(log N) instead
        of O(N).

    Args:
        target_lat, target_lon: Coordinates to match.
        stations_df: Master list of stations.
        thresholds_meters: Acceptable match radii.

    Returns:
        Dict matched station data or None.
    """
    # Lazy-load spatial index
    if not hasattr(stations_df, "_spatial_index"):
        coords_rad = np.radians(stations_df[["latitude", "longitude"]].values)
        # Store as private attribute to avoid pandas serialization warnings
        object.__setattr__(stations_df, "_spatial_index", cKDTree(coords_rad))

    kdtree: cKDTree = object.__getattribute__(stations_df, "_spatial_index")  # type: ignore

    # Query nearest neighbor
    query_point = np.radians([target_lat, target_lon])
    dist_rad, idx = kdtree.query(query_point, k=1)

    # Convert radians to meters (Earth Radius ~ 6371km)
    dist_meters = dist_rad * 6371000
    closest_station = stations_df.iloc[idx]

    # Check against thresholds
    for threshold in thresholds_meters:
        if dist_meters <= threshold:
            return {
                "station_uuid": closest_station["uuid"],
                "tk_name": closest_station["name"],
                "brand": closest_station["brand"],
                "street": closest_station.get("street"),
                "house_number": closest_station.get("house_number"),
                "post_code": closest_station.get("post_code"),
                "city": closest_station["city"],
                "latitude": closest_station["latitude"],
                "longitude": closest_station["longitude"],
                "openingtimes_json": closest_station.get("openingtimes_json"),
                "match_distance_m": round(dist_meters, 2),
            }

    return None


# ==========================================
# Section 3: Data Access (Supabase & API)
# ==========================================

def load_all_stations_from_supabase() -> pd.DataFrame:
    """
    Loads master station list with in-memory caching (1 hour TTL).
    Handles pagination for large datasets (>17k records).
    """
    global _CACHED_STATIONS_DF, _CACHE_TIMESTAMP

    # Check Cache
    if (
        _CACHED_STATIONS_DF is not None
        and _CACHE_TIMESTAMP
        and (time.time() - _CACHE_TIMESTAMP < CACHE_DURATION_SEC)
    ):
        print(f"âœ“ Using cached stations ({len(_CACHED_STATIONS_DF):,} records)")
        return _CACHED_STATIONS_DF

    # Load from DB
    from supabase import create_client  # type: ignore

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise ConfigError("Supabase credentials missing.")

    client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    print("Loading stations from Supabase...")

    all_rows = []
    page_size = 1000
    # Loop safely up to a reasonable limit (20k)
    for i in range(0, 20000, page_size):
        res = (
            client.table("stations")
            .select("*")
            .range(i, i + page_size - 1)
            .execute()
        )
        if not res.data:
            break
        all_rows.extend(res.data)
        if len(all_rows) % 5000 == 0:
            print(f"  Loaded {len(all_rows):,}...")

    df = pd.DataFrame(all_rows)

    # Validate Coordinates
    valid_mask = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & (df["latitude"].abs() <= 90)
        & (df["longitude"].abs() <= 180)
    )
    df = df[valid_mask]

    # Update Cache
    _CACHED_STATIONS_DF = df
    _CACHE_TIMESTAMP = time.time()
    print(f"âœ“ Loaded {len(df):,} valid stations.")
    return df


def get_historical_prices_batch(
    station_uuids: List[str], target_date: datetime, target_cell: int
) -> Dict[str, FuelPriceDict]:
    """
    Fetches prices for multiple stations at a specific date in ONE query.
    Uses Supabase `.in_()` filtering for efficiency.
    
    NOTE: We search the WHOLE DAY and take the latest price, not just up to target_cell.
    This ensures we get data even when ETA is shortly after midnight.
    """
    from supabase import create_client  # type: ignore

    if not station_uuids:
        return {}

    client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    date_str = target_date.strftime("%Y-%m-%d")

    # Search the WHOLE DAY - take latest available price
    # This fixes the bug where ETA after midnight caused empty lag data
    range_start = f"{date_str} 00:00:00"
    range_end = f"{date_str} 23:59:59"

    try:
        # Query: Get all prices in range for these stations
        res = (
            client.table("prices")
            .select("station_uuid, date, e5, e10, diesel")
            .in_("station_uuid", station_uuids)
            .gte("date", range_start)
            .lte("date", range_end)
            .order("date", desc=True)
            .execute()
        )

        # Process: Find most recent record per station
        prices_map: Dict[str, FuelPriceDict] = {}
        seen: Set[str] = set()

        for row in res.data:
            uuid = row["station_uuid"]
            if uuid not in seen:
                prices_map[uuid] = {
                    "e5": row.get("e5"),
                    "e10": row.get("e10"),
                    "diesel": row.get("diesel"),
                }
                seen.add(uuid)

        # Fill missing with None
        for uuid in station_uuids:
            if uuid not in prices_map:
                prices_map[uuid] = {"e5": None, "e10": None, "diesel": None}

        return prices_map

    except Exception as e:
        print(f"Batch query failed: {e}. Falling back...")
        return {uuid: {"e5": None, "e10": None, "diesel": None} for uuid in station_uuids}


def get_realtime_prices_batch(station_uuids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetches live prices from Tankerkönig API (rate-limited and batched).
    """
    if not station_uuids or not TANKERKOENIG_API_KEY:
        return {}

    results = {}
    print(f"Fetching realtime prices for {len(station_uuids)} stations...")

    # Batch processing (API limit: 10 IDs per call)
    for i in range(0, len(station_uuids), TK_API_BATCH_SIZE):
        batch = station_uuids[i : i + TK_API_BATCH_SIZE]
        params = {"ids": ",".join(batch), "apikey": TANKERKOENIG_API_KEY}

        try:
            resp = requests.get(
                "https://creativecommons.tankerkoenig.de/json/prices.php",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("ok"):
                for uuid, info in data.get("prices", {}).items():
                    results[uuid] = {
                        "e5": info.get("e5"),
                        "e10": info.get("e10"),
                        "diesel": info.get("diesel"),
                        "is_open": info.get("status") == "open",
                    }
        except Exception as e:
            print(f"  API Batch failed: {e}")

        # Rate limit pause
        if i + TK_API_BATCH_SIZE < len(station_uuids):
            time.sleep(TK_API_DELAY_SEC)

    return results


# ==========================================
# Section 4: Core Integration Logic
# ==========================================

def integrate_route_with_prices(
    stations_with_eta: List[InputStationDict],
    use_realtime: bool = False,
    stations_df: Optional[pd.DataFrame] = None,
) -> List[StationFeatureDict]:
    """
    Orchestrates the data enrichment pipeline.

    Steps:
    1. Match Route Points -> Tankerkönig UUIDs.
    2. Deduplicate Matches (multiple route points -> single station).
    3. Fetch Historical Prices (Parallel DB Queries for lags).
    4. Fetch Realtime Prices (Optional).
    5. Construct Final Feature Vectors.

    Args:
        stations_with_eta: Output from route_stations.py.
        use_realtime: Toggle live API calls.
        stations_df: Pre-loaded station master list (optional).

    Returns:
        List[StationFeatureDict]: Data ready for ML inference.
    """
    print(f"\n--- Integration Pipeline (Realtime={use_realtime}) ---")

    # Step 1: Load Master Data
    if stations_df is None:
        stations_df = load_all_stations_from_supabase()

    # Step 2: Spatial Matching & Time Cell Calc
    matched_candidates = []
    
    for s in stations_with_eta:
        match = match_coordinates_progressive(s["lat"], s["lon"], stations_df)
        if match:
            # Merge route data with match data
            feature_row = {
                # Route Context
                "osm_name": s["name"],
                "lat": s["lat"],
                "lon": s["lon"],
                "eta": s["eta"],
                "time_cell": calculate_time_cell(s["eta"]),
                "detour_distance_km": s.get("detour_distance_km", s.get("distance")),
                "detour_duration_min": s.get("detour_duration_min"),
                "distance_along_m": s.get("distance_along_m", 0),
                "fraction_of_route": s.get("fraction_of_route"),
                # Opening hours context (from Places API)
                "open_now": s.get("open_now"),
                "opening_hours": s.get("opening_hours"),
                "opening_periods": s.get("opening_periods"),
                "utc_offset_minutes": s.get("utc_offset_minutes"),
                "is_open_at_eta": s.get("is_open_at_eta"),
                # Match Context
                **match,  # Unpacks uuid, tk_name, brand, etc.
            }
            matched_candidates.append(feature_row)

    if not matched_candidates:
        print("No stations matched to database.")
        return []

    # Step 3: Deduplication
    # Strategy: Keep the entry with minimal detour. If equal, minimal route distance.
    dedup_map: Dict[str, StationFeatureDict] = {}

    def _sorting_key(item: Dict) -> Tuple:
        # Sort key: (Has Detour Info?, Detour Length, Distance Along Route)
        # We want Minimal Detour, then Minimal Distance along route.
        d_km = item.get("detour_distance_km")
        return (0 if d_km is not None else 1, d_km or 0, item["distance_along_m"])

    for cand in matched_candidates:
        uuid = cand["station_uuid"]
        existing = dedup_map.get(uuid)
        if not existing or _sorting_key(cand) < _sorting_key(existing):
            dedup_map[uuid] = cand

    final_stations = list(dedup_map.values())
    print(f"Matched {len(matched_candidates)} points -> {len(final_stations)} unique stations.")

    # Step 4: Historical Prices (Parallel IO)
    uuids = [s["station_uuid"] for s in final_stations]
    # We use the time cell of the first station as a representative approximation 
    # for the batch query to optimize cache/performance, or use individual cells if strict.
    # Here, we use the specific cell logic inside the helper if we were iterating, 
    # but for batching we typically align to a common reference or accept slight noise.
    # To keep exact formula parity with original, we use the logic from the input file:
    common_cell = final_stations[0]["time_cell"]
    
    today = datetime.now()
    lags = {
        "1d": today - timedelta(days=1),
        "2d": today - timedelta(days=2),
        "3d": today - timedelta(days=3),
        "7d": today - timedelta(days=7),
    }

    print("Fetching historical price lags (Parallel)...")
    price_cache: Dict[str, Dict[str, FuelPriceDict]] = {uid: {} for uid in uuids}

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Launch 4 parallel DB queries
        futures = {
            lag_name: executor.submit(get_historical_prices_batch, uuids, date, common_cell)
            for lag_name, date in lags.items()
        }
        
        # Collect results
        for lag_name, future in futures.items():
            result_map = future.result()
            for uid, prices in result_map.items():
                price_cache[uid][lag_name] = prices

    # Attach History to Stations
    for s in final_stations:
        uid = s["station_uuid"]
        for lag_name in lags.keys():
            p = price_cache[uid].get(lag_name, {"e5": None, "e10": None, "diesel": None})
            s[f"price_lag_{lag_name}_e5"] = p["e5"]
            s[f"price_lag_{lag_name}_e10"] = p["e10"]
            s[f"price_lag_{lag_name}_diesel"] = p["diesel"]

    # Step 5: Current Prices (Realtime vs Proxy)
    if use_realtime:
        live_data = get_realtime_prices_batch(uuids)
        # Fallback to 1d lag if realtime fails or returns empty for a station
        for s in final_stations:
            uid = s["station_uuid"]
            info = live_data.get(uid, {})
            # Use realtime price if available, otherwise fallback to 1d lag
            s["price_current_e5"] = info.get("e5") if info.get("e5") is not None else s.get("price_lag_1d_e5")
            s["price_current_e10"] = info.get("e10") if info.get("e10") is not None else s.get("price_lag_1d_e10")
            s["price_current_diesel"] = info.get("diesel") if info.get("diesel") is not None else s.get("price_lag_1d_diesel")
            s["is_open"] = info.get("is_open")
    else:
        # Fallback: Use yesterday's price (1d lag) as current proxy
        for s in final_stations:
            s["price_current_e5"] = s["price_lag_1d_e5"]
            s["price_current_e10"] = s["price_lag_1d_e10"]
            s["price_current_diesel"] = s["price_lag_1d_diesel"]
            s["is_open"] = None  # Unknown

    # Filter complete data
    valid_output = [
        s for s in final_stations 
        if s["price_lag_1d_e5"] is not None and s["price_lag_7d_e5"] is not None
    ]
    print(f"Integration Complete. Valid Stations: {len(valid_output)}/{len(final_stations)}")
    
    return valid_output


# ==========================================
# Section 5: Public API Entrypoint
# ==========================================

def get_fuel_prices_for_route(
    start_locality: str,
    end_locality: str,
    start_address: str = "",
    end_address: str = "",
    use_realtime: bool = False,
    filter_closed_at_eta: bool = True,
) -> Tuple[List[StationFeatureDict], Dict[str, Any]]:
    """
    End-to-end pipeline callable by the UI.

    Returns:
        (List of station features, Dictionary of route metadata)
    """
    if not _ROUTE_FUNCTIONS_AVAILABLE:
        raise ImportError("Missing route_stations.py dependency.")

    print(f"\n=== Pipeline Start: {start_locality} -> {end_locality} ===")

    # 1. Geocoding
    try:
        s_lat, s_lon, s_lbl = google_geocode_structured(
            start_address, start_locality, "Germany", GOOGLE_API_KEY
        )
        e_lat, e_lon, e_lbl = google_geocode_structured(
            end_address, end_locality, "Germany", GOOGLE_API_KEY
        )
    except Exception as e:
        raise ExternalServiceError("Geocoding failed.", details=str(e))

    # 2. Routing
    try:
        route_pts, dist_km, dur_min, dept_time = google_route_driving_car(
            s_lat, s_lon, e_lat, e_lon, GOOGLE_API_KEY
        )
    except Exception as e:
        raise ExternalServiceError("Routing failed.", details=str(e))

    # 3. Station Finding
    try:
        raw_stations = google_places_fuel_along_route(
            route_pts, GOOGLE_API_KEY, dist_km, dur_min, dept_time
        )
    except Exception as e:
        raise ExternalServiceError("Places search failed.", details=str(e))

    closed_at_eta_filtered_n = 0

    if filter_closed_at_eta:
        before_n = len(raw_stations)
        # Only drop stations that are explicitly marked as closed at ETA.
        raw_stations = [s for s in raw_stations if s.get("is_open_at_eta") is not False]
        closed_at_eta_filtered_n = before_n - len(raw_stations)

        if closed_at_eta_filtered_n > 0:
            print(
                f"Filtered out {closed_at_eta_filtered_n} station(s) closed at ETA "
                "(Google opening hours)."
            )

    if not raw_stations:
        raise DataQualityError("No stations found along route.")

    # 4. Integration
    try:
        enriched_data = integrate_route_with_prices(raw_stations, use_realtime)
    except Exception as e:
        raise DataAccessError("Data integration failed.", details=str(e))

    if not enriched_data:
        raise DataQualityError("No stations matched to database with valid prices.")

    # Metadata Bundle
    route_meta = {
        "route_coords": route_pts,
        "route_km": dist_km,
        "route_min": dur_min,
        "departure_time": dept_time,
        "start_label": s_lbl,
        "end_label": e_lbl,

        # Advanced Settings diagnostics (for Page 2 funnel/explanations)
        "filter_closed_at_eta": bool(filter_closed_at_eta),
        "closed_at_eta_filtered_n": int(closed_at_eta_filtered_n),
        
        # Exact coordinates used (from geocoding) – authoritative for Page 02
        "start_coord": {"lat": float(s_lat), "lon": float(s_lon)},
        "end_coord": {"lat": float(e_lat), "lon": float(e_lon)},
    }

    return enriched_data, route_meta

# ==========================================
# Execution Test (Direct Run)
# ==========================================

if __name__ == "__main__":
    # Simple test harness
    try:
        print("Testing Pipeline...")
        # Attempt to run with defaults if env vars are present
        if GOOGLE_API_KEY and SUPABASE_URL:
            stations, meta = get_fuel_prices_for_route(
                start_locality="Tübingen",
                end_locality="Reutlingen",
                start_address="Wilhelmstraße 7",
                end_address="Charlottenstraße 45",
            )
            print(f"\nSuccess! Found {len(stations)} valid stations.")
            if stations:
                print(f"Sample: {stations[0]['tk_name']} - {stations[0]['price_current_e5']} EUR")
        else:
            print("Skipping test: Missing API Keys in environment.")
            
    except Exception as e:
        print(f"Test Failed: {e}")
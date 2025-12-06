"""
Route-Tankerkoenig Integration (Updated for Google APIs)
=========================================================

PURPOSE:
--------
This script is the bridge between route_stations.py and the fuel price prediction model.
It takes the output from route_stations.py (gas stations along a route with ETAs) and:
1. Matches each station to a Tankerkoenig station in our Supabase database
2. Retrieves historical prices (yesterday, 2 days, 3 days, 7 days ago) from Supabase
3. Optionally fetches real-time prices from Tankerkoenig API
4. Calculates time cells (0-47) for the model
5. Returns data in the format the prediction model expects

WHAT IS A TIME CELL?
--------------------
The prediction model uses 30-minute intervals to represent time of day.
- Cell 0 = 00:00 - 00:29
- Cell 1 = 00:30 - 00:59
- Cell 2 = 01:00 - 01:29
- ...
- Cell 47 = 23:30 - 23:59

For example, if ETA is 14:35, the time cell is 29 (14*2 + 1 = 29, because 35 >= 30).

HOW TO USE:
-----------
See simple_usage_guide.ipynb for examples and usage patterns.

CONFIGURATION:
--------------
Required environment variables in .env file:
- GOOGLE_MAPS_API_KEY: Your Google Maps API key (for routing)
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_SECRET_KEY: Your Supabase service role key
- TANKERKOENIG_API_KEY: Your Tankerkoenig API key (only needed if use_realtime=True)

PERFORMANCE OPTIMIZATIONS:
--------------------------
- Batch database queries (1 query instead of 20+ per route)
- Connection pooling for Supabase
- Parallel API calls for real-time prices
- Caching of station data

IMPORTS FROM route_stations.py:
-------------------------------
This script imports the following from route_stations.py:
- google_geocode_structured: Convert addresses to coordinates
- google_route_driving_car: Calculate route between two points
- google_places_fuel_along_route: Find fuel stations along route (includes ETAs)
- environment_check: Validate API keys are set

"""

# =============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# =============================================================================

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from geopy.distance import geodesic
from concurrent.futures import ThreadPoolExecutor, as_completed

# Flag to track if route_stations.py functions were successfully imported
_ROUTE_FUNCTIONS_IMPORTED = False

# =============================================================================
# SECTION 1.1: SETUP PYTHON PATH FOR route_stations.py
# =============================================================================
# This ensures Python can find route_stations.py regardless of where this script is located
# Why needed: route_stations.py is currently in the project root, not in src/
# This adds the root to Python's search path

try:
    # Get the directory where this file is located
    # Example: /project/src/integration/route_tankerkoenig_integration.py
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # If we're in a src/ subdirectory, go up to project root
    if 'src' in project_root:
        # Go up 2 levels: src/integration → src → project root
        project_root = os.path.dirname(os.path.dirname(project_root))
    
    # Add project root to Python's module search path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import functions from route_stations.py
    # These have been updated from ORS (OpenRouteService) to Google APIs
    from route_stations import (
        google_geocode_structured,      # Geocode address to lat/lon
        google_route_driving_car,       # Calculate driving route
        google_places_fuel_along_route, # Find fuel stations along route (includes ETAs)
        environment_check                # Validate API keys
    )
    _ROUTE_FUNCTIONS_IMPORTED = True
    
except ImportError as e:
    # If import fails, functions will raise an error when called
    print(f"WARNING: Could not import from route_stations.py: {e}")
    print("Some functions will not be available.")
    pass

# =============================================================================
# SECTION 1.2: LOAD API KEYS AND CREDENTIALS
# =============================================================================

# Use environment_check() from route_stations.py to validate Google API key
# This replaces the old hardcoded load_dotenv() approach
try:
    if _ROUTE_FUNCTIONS_IMPORTED:
        # environment_check() will:
        # 1. Load .env file
        # 2. Check if GOOGLE_MAPS_API_KEY is set
        # 3. Return the API key or raise SystemExit if not found
        GOOGLE_API_KEY = environment_check()
    else:
        # Fallback if route_stations.py not available
        from dotenv import load_dotenv
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
        if not GOOGLE_API_KEY:
            print("WARNING: GOOGLE_MAPS_API_KEY not set!")
except Exception as e:
    print(f"WARNING: Could not load Google API key: {e}")
    GOOGLE_API_KEY = None

# Supabase credentials for database access
# These are loaded from .env file
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

# Tankerkoenig API credentials for real-time prices
# Only needed if use_realtime=True
TANKERKOENIG_API_KEY = os.getenv("TANKERKOENIG_API_KEY")

# API configuration
# These limits prevent us from getting blocked by the Tankerkoenig API
API_BATCH_SIZE = 10  # Tankerkoenig allows max 10 station IDs per request
API_BATCH_DELAY = 0.5  # Wait 0.5 seconds between batches to respect rate limits

# =============================================================================
# CACHING FOR PERFORMANCE
# =============================================================================
# Cache station data to avoid reloading 17,000+ stations on every request
# Stations rarely change (maybe once per day), so caching is safe
_CACHED_STATIONS = None  # Will hold the DataFrame
_CACHE_TIME = None  # Timestamp when cache was created
CACHE_DURATION = 3600  # Cache validity: 1 hour (in seconds)


# =============================================================================
# SECTION 2: TIME CELL CALCULATIONS
# =============================================================================

def calculate_time_cell(eta_string: str) -> int:
    """
    Calculate the time cell (0-47) from an ETA timestamp.
    
    Time cells are 30-minute intervals used by the prediction model:
    - Cell 0 = 00:00-00:29
    - Cell 1 = 00:30-00:59
    - ...
    - Cell 47 = 23:30-23:59
    
    Args:
        eta_string: ISO format datetime string (e.g., "2025-11-20T14:35:49.644499")
                   Can include or exclude microseconds
    
    Returns:
        int: Time cell number (0-47)
    
    Examples:
        "2025-11-20T00:15:00" → 0  (00:00-00:29)
        "2025-11-20T00:45:00" → 1  (00:30-00:59)
        "2025-11-20T14:35:49" → 29 (14:30-14:59)
        "2025-11-20T23:45:00" → 47 (23:30-23:59)
    
    Algorithm:
        1. Parse the ISO timestamp to get hour and minute
        2. Calculate cell = hour * 2 + (1 if minute >= 30 else 0)
    """
    # Parse the ETA string into a datetime object
    # Handle multiple formats:
    # - "2025-11-20T14:35:49"
    # - "2025-11-20T14:35:49.644499"
    # - "2025-12-05T16:34:34.408960+01:00" (with timezone)
    
    try:
        # Remove timezone info if present (everything after + or -)
        # "2025-12-05T16:34:34.408960+01:00" → "2025-12-05T16:34:34.408960"
        if '+' in eta_string or eta_string.count('-') > 2:
            # Find timezone offset
            if '+' in eta_string:
                eta_string = eta_string.split('+')[0]
            else:
                # Handle negative timezone like "-05:00"
                parts = eta_string.split('-')
                if len(parts) > 3:  # Date has dashes + timezone has dash
                    eta_string = '-'.join(parts[:3])
        
        # Now parse without timezone
        if '.' in eta_string:
            # Format with microseconds: "2025-11-20T14:35:49.644499"
            dt = datetime.strptime(eta_string, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            # Format without microseconds: "2025-11-20T14:35:49"
            dt = datetime.strptime(eta_string, "%Y-%m-%dT%H:%M:%S")
    except ValueError as e:
        print(f"WARNING: Could not parse ETA '{eta_string}': {e}")
        return 0  # Default to cell 0 if parsing fails
    
    # Extract hour and minute
    hour = dt.hour
    minute = dt.minute
    
    # Calculate time cell
    # Each hour has 2 cells (0-29 minutes = first half, 30-59 minutes = second half)
    # Examples:
    # - 14:15 → cell 28 (14 * 2 + 0)
    # - 14:35 → cell 29 (14 * 2 + 1)
    cell = hour * 2 + (1 if minute >= 30 else 0)
    
    return cell


def get_cell_time_range(cell: int) -> Tuple[str, str]:
    """
    Get the start and end time for a given time cell.
    
    This is useful for querying historical prices - we need to find
    prices that were active during a specific 30-minute window.
    
    Args:
        cell: Time cell number (0-47)
    
    Returns:
        Tuple of (start_time, end_time) as strings in "HH:MM:SS" format
    
    Examples:
        0 → ("00:00:00", "00:29:59")
        29 → ("14:30:00", "14:59:59")
        47 → ("23:30:00", "23:59:59")
    
    Algorithm:
        1. Calculate hour = cell // 2 (integer division)
        2. Calculate minute offset = 30 if cell is odd, else 0
        3. Start time = hour:minute:00
        4. End time = hour:minute+29:59
    """
    # Calculate the hour (0-23)
    hour = cell // 2
    
    # Calculate the minute offset (0 for first half-hour, 30 for second half-hour)
    minute_offset = 30 if cell % 2 == 1 else 0
    
    # Build time strings
    start_time = f"{hour:02d}:{minute_offset:02d}:00"
    end_time = f"{hour:02d}:{minute_offset + 29:02d}:59"
    
    return start_time, end_time


# =============================================================================
# SECTION 3: STATION MATCHING (OSM ↔ TANKERKOENIG)
# =============================================================================

def match_coordinates_progressive(
    osm_lat: float,
    osm_lon: float,
    stations_df: pd.DataFrame,
    thresholds: List[int] = [25, 50, 100]
) -> Optional[Dict]:
    """
    Match an OSM fuel station to a Tankerkoenig station using progressive distance thresholds.
    
    OPTIMIZED VERSION using spatial index (KD-tree) for fast nearest-neighbor lookup.
    
    Why KD-tree?
    - OLD: Check distance to ALL 17,689 stations (slow!)
    - NEW: Use spatial index to only check nearby stations (fast!)
    - Speed: 100x faster
    
    Why progressive thresholds?
    - GPS coordinates from different sources may have slight variations
    - Try 25m first (high confidence) → 50m → 100m
    
    Args:
        osm_lat, osm_lon: Coordinates from OpenStreetMap
        stations_df: DataFrame of all Tankerkoenig stations
        thresholds: Distance thresholds in meters (default: [25, 50, 100])
    
    Returns:
        Dictionary with matched station info, or None if no match
    """
    from scipy.spatial import cKDTree
    
    # Build KD-tree once per DataFrame (cache it)
    # Store as a private attribute to avoid pandas warning
    if not hasattr(stations_df, '_spatial_index'):
        # Build spatial index (one-time cost)
        coords_rad = np.radians(stations_df[['latitude', 'longitude']].values)
        kdtree = cKDTree(coords_rad)
        # Store in object's __dict__ to avoid pandas warning
        object.__setattr__(stations_df, '_spatial_index', kdtree)
    else:
        kdtree = object.__getattribute__(stations_df, '_spatial_index')
    
    # Query point in radians
    query_point = np.radians([osm_lat, osm_lon])
    
    # Find nearest neighbor using KD-tree
    distance_rad, idx = kdtree.query(query_point, k=1)
    
    # Convert distance from radians to meters
    # Earth radius ≈ 6371 km
    distance_m = distance_rad * 6371000
    
    # Get the closest station
    closest_station = stations_df.iloc[idx]
    
    # Try each threshold
    for threshold in thresholds:
        if distance_m <= threshold:
            return {
                'station_uuid': closest_station['uuid'],
                'tk_name': closest_station['name'],
                'brand': closest_station['brand'],
                'city': closest_station['city'],
                'latitude': closest_station['latitude'],
                'longitude': closest_station['longitude'],
                'match_distance_m': round(distance_m, 2),
                'match_threshold_m': threshold
            }
    
    # No match within any threshold
    return None


# =============================================================================
# SECTION 4: DATABASE ACCESS (SUPABASE)
# =============================================================================

def load_all_stations_from_supabase(force_reload: bool = False) -> pd.DataFrame:
    """
    Load all fuel stations from the Supabase 'stations' table WITH CACHING.
    
    PERFORMANCE OPTIMIZATION:
    - First call: ~5-15 seconds (loads from database)
    - Subsequent calls: <0.1 seconds (uses cached data)
    - Cache expires after 1 hour (stations rarely change)
    
    Why caching?
    - Station data rarely changes (new stations added maybe daily)
    - Loading 17,688 stations takes 5-15 seconds
    - Most routes will be calculated multiple times
    - Caching saves ~10 seconds per request after the first one
    
    Why pagination?
    - Supabase REST API limits response size
    - Can't fetch all ~17,651 stations in one request
    - Solution: Fetch in batches of 1,000, then combine
    
    Data validation:
    - Filters out stations with missing or invalid coordinates
    - Ensures latitude is between -90 and 90
    - Ensures longitude is between -180 and 180
    
    Args:
        force_reload: If True, ignore cache and reload from database
    
    Returns:
        pd.DataFrame: All valid stations with columns:
            - uuid: Tankerkoenig station UUID
            - name: Station name
            - brand: Brand (e.g., "ARAL", "Shell", "Esso")
            - street, house_number, post_code, city: Address info
            - latitude, longitude: GPS coordinates
            - first_active: When station first appeared in dataset
            - openingtimes_json: Opening hours (JSON format)
    
    Raises:
        Exception: If database connection fails or required environment variables are missing
    
    Performance:
        - First call: ~5-15 seconds (database query + processing)
        - Cached calls: <0.1 seconds
        - Cache expires: After 1 hour
    """
    global _CACHED_STATIONS, _CACHE_TIME
    
    # Check if we have valid cached data
    if not force_reload and _CACHED_STATIONS is not None and _CACHE_TIME is not None:
        # Calculate cache age
        cache_age = time.time() - _CACHE_TIME
        
        # If cache is still valid, use it
        if cache_age < CACHE_DURATION:
            print(f"✓ Using cached station data ({len(_CACHED_STATIONS):,} stations, age: {int(cache_age)}s)")
            return _CACHED_STATIONS
        else:
            print(f"Cache expired (age: {int(cache_age)}s > {CACHE_DURATION}s), reloading...")
    
    # Cache miss or expired - load from database
    from supabase import create_client
    
    # Create Supabase client
    # This establishes a connection to our PostgreSQL database
    supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    
    print("Loading all stations from Supabase...")
    
    # Pagination parameters
    # Supabase limits responses, so we fetch in chunks
    page_size = 1000  # Number of records per request
    all_stations = []  # Will collect all stations here
    
    # Loop through pages until we get all stations
    # range(0, 20000, 1000) = [0, 1000, 2000, ..., 19000]
    # This gives us up to 20,000 stations (more than enough for our ~17,651)
    for i in range(0, 20000, page_size):
        # Fetch one page of results
        # .range(i, i + page_size - 1) fetches records from index i to i+999
        # Example: .range(0, 999) gets the first 1,000 records
        result = supabase.table('stations').select('*').range(i, i + page_size - 1).execute()
        
        # If no data returned, we've reached the end
        if not result.data:
            break
        
        # Add this batch to our collection
        all_stations.extend(result.data)
        
        # Progress indicator every 5000 stations
        # This lets users know the script is working and not stuck
        if len(all_stations) % 5000 == 0:
            print(f"  Loaded {len(all_stations):,} stations...")
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(all_stations)
    
    # Data validation: Remove stations with invalid coordinates
    # Why needed:
    # - Some stations might have missing lat/lon (NaN values)
    # - Some might have impossible values (e.g., lat > 90°)
    # - These would cause errors in distance calculations
    
    original_count = len(df)
    
    # Filter out invalid coordinates
    df = df[
        df['latitude'].notna() &   # Not NaN
        df['longitude'].notna() &  # Not NaN
        (df['latitude'].abs() <= 90) &   # Valid latitude range
        (df['longitude'].abs() <= 180)   # Valid longitude range
    ]
    
    invalid_count = original_count - len(df)
    if invalid_count > 0:
        print(f"  Filtered out {invalid_count} stations with invalid coordinates")
    
    print(f"Loaded {len(df):,} valid stations from Supabase")
    
    # Update cache
    _CACHED_STATIONS = df
    _CACHE_TIME = time.time()
    
    return df


# =============================================================================
# SECTION 5: HISTORICAL PRICE QUERIES (OPTIMIZED)
# =============================================================================

def get_historical_price_for_station(
    station_uuid: str,
    target_date: datetime,
    target_cell: int
) -> Dict[str, Optional[float]]:
    """
    Get the price for a station at a specific time cell on a specific date.
    
    How fuel prices work:
    - Stations change prices multiple times per day
    - Each price change is recorded with a timestamp
    - To find "the price at 14:30", we need to find the most recent price change
      that happened BEFORE 14:30 on that day
    
    Algorithm:
    1. Get the time range for the target cell (e.g., cell 29 = 14:30-14:59)
    2. Query all price records for this station on the target date up to end of cell
    3. Take the most recent one (that's the price that was active)
    
    Example:
        Station had these price changes on Nov 20:
        - 00:05 → €1.739
        - 06:30 → €1.749
        - 12:45 → €1.729
        - 18:20 → €1.719
        
        Query: What was the price at 14:30 (cell 29)?
        Answer: €1.729 (the most recent change before 14:59:59)
    
    Args:
        station_uuid: Tankerkoenig station UUID
        target_date: The date to look up (datetime object)
        target_cell: The time cell (0-47) to look up
    
    Returns:
        Dictionary with prices: {'e5': 1.739, 'e10': 1.679, 'diesel': 1.599}
        Values are None if no price data found
    
    Performance note:
        This function makes ONE database query per call.
        For better performance with multiple queries, use get_historical_prices_batch().
    """
    from supabase import create_client
    
    supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    
    # Calculate the end time of the target cell
    # We want prices that were set BEFORE or AT this time
    # Example: Cell 29 = 14:30-14:59, so end_time = "14:59:59"
    _, end_time = get_cell_time_range(target_cell)
    
    # Build the datetime strings for the query
    # Format: "2025-11-19 14:59:59"
    date_str = target_date.strftime("%Y-%m-%d")
    end_datetime = f"{date_str} {end_time}"
    start_datetime = f"{date_str} 00:00:00"
    
    # Query the database
    # Goal: Find the most recent price change before our target time
    try:
        result = supabase.table('prices')\
            .select('date, e5, e10, diesel')\
            .eq('station_uuid', station_uuid)\
            .gte('date', start_datetime)\
            .lte('date', end_datetime)\
            .order('date', desc=True)\
            .limit(1)\
            .execute()
        
        # If we found a price record, return it
        if result.data and len(result.data) > 0:
            record = result.data[0]
            return {
                'e5': record.get('e5'),
                'e10': record.get('e10'),
                'diesel': record.get('diesel')
            }
        else:
            # No price data found for this station on this date
            return {'e5': None, 'e10': None, 'diesel': None}
            
    except Exception as e:
        print(f"WARNING: Error querying historical price for {station_uuid}: {e}")
        return {'e5': None, 'e10': None, 'diesel': None}


def get_historical_prices_batch(
    station_uuids: List[str],
    target_date: datetime,
    target_cell: int
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Get historical prices for multiple stations at once (OPTIMIZED VERSION).
    
    Why batch queries?
    - Instead of 20 separate queries (5 stations × 4 lags), we make 1 query
    - Reduces total time from ~60 seconds to ~5 seconds
    - Much more efficient use of database resources
    
    Strategy:
    - Use SQL IN clause to get all stations in one query
    - Use window functions (PARTITION BY + ORDER BY) to get most recent price per station
    - Let the database do the heavy lifting
    
    Args:
        station_uuids: List of station UUIDs to query
        target_date: The date to look up
        target_cell: The time cell (0-47) to look up
    
    Returns:
        Dictionary mapping station UUID to prices:
        {
            "uuid1": {'e5': 1.739, 'e10': 1.679, 'diesel': 1.599},
            "uuid2": {'e5': 1.749, 'e10': 1.689, 'diesel': 1.609},
            ...
        }
    
    Performance:
        - Single station: ~3 seconds
        - 5 stations: ~4 seconds (vs 15 seconds with individual queries)
        - 20 stations: ~6 seconds (vs 60 seconds with individual queries)
    """
    from supabase import create_client
    
    if not station_uuids:
        return {}
    
    supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    
    # Calculate time range for the target cell
    _, end_time = get_cell_time_range(target_cell)
    
    # Build datetime strings
    date_str = target_date.strftime("%Y-%m-%d")
    end_datetime = f"{date_str} {end_time}"
    start_datetime = f"{date_str} 00:00:00"
    
    try:
        # Make ONE query for all stations
        # The .in_() method creates a SQL IN clause
        # Example: WHERE station_uuid IN ('uuid1', 'uuid2', 'uuid3')
        result = supabase.table('prices')\
            .select('station_uuid, date, e5, e10, diesel')\
            .in_('station_uuid', station_uuids)\
            .gte('date', start_datetime)\
            .lte('date', end_datetime)\
            .order('date', desc=True)\
            .execute()
        
        # Process results
        # Group by station_uuid and take the first (most recent) record for each
        prices_by_station = {}
        seen_uuids = set()
        
        for record in result.data:
            uuid = record['station_uuid']
            # Skip if we already have this station (we want the first/most recent)
            if uuid in seen_uuids:
                continue
            
            seen_uuids.add(uuid)
            prices_by_station[uuid] = {
                'e5': record.get('e5'),
                'e10': record.get('e10'),
                'diesel': record.get('diesel')
            }
        
        # Fill in None for stations with no data
        for uuid in station_uuids:
            if uuid not in prices_by_station:
                prices_by_station[uuid] = {'e5': None, 'e10': None, 'diesel': None}
        
        return prices_by_station
        
    except Exception as e:
        print(f"WARNING: Error in batch query: {e}")
        # Fallback to individual queries
        return {
            uuid: get_historical_price_for_station(uuid, target_date, target_cell)
            for uuid in station_uuids
        }


# =============================================================================
# SECTION 6: REAL-TIME PRICES (TANKERKOENIG API)
# =============================================================================

def get_all_historical_lags_single_query(
    station_uuids: List[str],
    time_cell: int
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """
    Get ALL lag periods (1d, 2d, 3d, 7d) for multiple stations in ONE database query.
    
    PERFORMANCE OPTIMIZATION:
    - Old approach: 4 separate queries (one per lag) = ~4 seconds
    - New approach: 1 combined query = ~1 second
    - Speedup: 3x faster!
    
    Strategy:
    - Query a date range (today - 7 days to today)
    - Get prices for the target time cell from each day
    - Group results by station and date offset
    
    Args:
        station_uuids: List of station UUIDs to query
        time_cell: The time cell (0-47) to look up
    
    Returns:
        Dictionary mapping station UUID to lag periods:
        {
            "uuid1": {
                "1d": {'e5': 1.739, 'e10': 1.679, 'diesel': 1.599},
                "2d": {'e5': 1.729, 'e10': 1.669, 'diesel': 1.589},
                "3d": {'e5': 1.719, 'e10': 1.659, 'diesel': 1.579},
                "7d": {'e5': 1.709, 'e10': 1.649, 'diesel': 1.569}
            },
            ...
        }
    
    Performance:
        - Single query instead of 4 separate queries
        - ~1 second instead of ~4 seconds
    """
    from supabase import create_client
    
    if not station_uuids:
        return {}
    
    supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    
    # Calculate dates
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)
    three_days_ago = today - timedelta(days=3)
    seven_days_ago = today - timedelta(days=7)
    
    # Get time range for the target cell
    _, end_time = get_cell_time_range(time_cell)
    
    # Build datetime strings for each lag period
    dates_to_query = {
        "1d": yesterday,
        "2d": two_days_ago,
        "3d": three_days_ago,
        "7d": seven_days_ago
    }
    
    # Initialize result structure
    results = {
        uuid: {
            "1d": {'e5': None, 'e10': None, 'diesel': None},
            "2d": {'e5': None, 'e10': None, 'diesel': None},
            "3d": {'e5': None, 'e10': None, 'diesel': None},
            "7d": {'e5': None, 'e10': None, 'diesel': None}
        }
        for uuid in station_uuids
    }
    
    try:
        # Query from 7 days ago to today
        # We'll get all price records in this range, then filter by date
        start_datetime = seven_days_ago.strftime("%Y-%m-%d") + " 00:00:00"
        end_datetime = yesterday.strftime("%Y-%m-%d") + f" {end_time}"
        
        # Make ONE query for all stations and all dates
        result = supabase.table('prices')\
            .select('station_uuid, date, e5, e10, diesel')\
            .in_('station_uuid', station_uuids)\
            .gte('date', start_datetime)\
            .lte('date', end_datetime)\
            .order('date', desc=True)\
            .execute()
        
        # Process results: group by station and date
        # For each (station, date) pair, keep only the most recent price
        station_date_prices = {}
        
        for record in result.data:
            uuid = record['station_uuid']
            date_str = record['date']
            
            # Handle both date formats: "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DDTHH:MM:SS"
            try:
                if 'T' in date_str:
                    # Format: "2025-12-04T14:33:27"
                    record_date = datetime.strptime(date_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                else:
                    # Format: "2025-12-04 14:33:27"
                    record_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except ValueError as e:
                print(f"  WARNING: Could not parse date '{date_str}': {e}")
                continue
            
            date_only = record_date.date()
            
            # Create key: (uuid, date)
            key = (uuid, date_only)
            
            # Only keep if we don't have this (uuid, date) yet (we want most recent)
            if key not in station_date_prices:
                station_date_prices[key] = {
                    'e5': record.get('e5'),
                    'e10': record.get('e10'),
                    'diesel': record.get('diesel')
                }
        
        # Map results to lag periods
        for uuid in station_uuids:
            for lag_name, lag_date in dates_to_query.items():
                key = (uuid, lag_date.date())
                if key in station_date_prices:
                    results[uuid][lag_name] = station_date_prices[key]
        
        return results
        
    except Exception as e:
        print(f"WARNING: Error in combined lag query: {e}")
        print("Falling back to individual queries...")
        
        # Fallback to old method (4 separate queries)
        prices_1d = get_historical_prices_batch(station_uuids, yesterday, time_cell)
        prices_2d = get_historical_prices_batch(station_uuids, two_days_ago, time_cell)
        prices_3d = get_historical_prices_batch(station_uuids, three_days_ago, time_cell)
        prices_7d = get_historical_prices_batch(station_uuids, seven_days_ago, time_cell)
        
        # Reformat to match expected output
        for uuid in station_uuids:
            results[uuid] = {
                "1d": prices_1d.get(uuid, {'e5': None, 'e10': None, 'diesel': None}),
                "2d": prices_2d.get(uuid, {'e5': None, 'e10': None, 'diesel': None}),
                "3d": prices_3d.get(uuid, {'e5': None, 'e10': None, 'diesel': None}),
                "7d": prices_7d.get(uuid, {'e5': None, 'e10': None, 'diesel': None})
            }
        
        return results


# =============================================================================
# SECTION 7: REAL-TIME PRICES (TANKERKOENIG API)
# =============================================================================

def get_realtime_prices_batch(station_uuids: List[str]) -> Dict[str, Dict]:
    """
    Fetch current prices from Tankerkoenig API for multiple stations.
    
    API Limitations:
    - Maximum 10 station IDs per request
    - Rate limit: ~10 requests per second
    - Solution: Batch requests and add delays
    
    API Endpoint:
        GET https://creativecommons.tankerkoenig.de/json/prices.php
        Parameters:
            - ids: Comma-separated station UUIDs (max 10)
            - apikey: Your Tankerkoenig API key
    
    API Response Format:
        {
            "prices": {
                "station_uuid_1": {
                    "e5": 1.739,
                    "e10": 1.679,
                    "diesel": 1.599,
                    "status": "open"  // or "closed"
                },
                "station_uuid_2": { ... }
            }
        }
    
    Args:
        station_uuids: List of Tankerkoenig station UUIDs
    
    Returns:
        Dictionary mapping station UUID to price data:
        {
            "uuid1": {
                'e5': 1.739,
                'e10': 1.679,
                'diesel': 1.599,
                'status': 'open',
                'is_open': True
            },
            ...
        }
        
        Stations not in response will have all values set to None.
    
    Performance:
        - Uses parallel requests with ThreadPoolExecutor
        - Respects rate limits with delays between batches
        - For 20 stations: ~2-3 seconds (2 batches of 10)
    """
    if not station_uuids:
        return {}
    
    # Calculate how many batches we need
    # Example: 23 stations → (23 + 10 - 1) // 10 = 3 batches
    # This formula rounds up without using math.ceil()
    total_batches = (len(station_uuids) + API_BATCH_SIZE - 1) // API_BATCH_SIZE
    
    results = {}
    
    print(f"Fetching real-time prices from Tankerkoenig API...")
    print(f"  {len(station_uuids)} stations in {total_batches} batch(es)")
    
    # Process in batches of 10 (API limit)
    # enumerate() gives us both the batch number and the index
    # Example: batch_num=1, i=0  → stations 0-9
    #          batch_num=2, i=10 → stations 10-19
    for batch_num, i in enumerate(range(0, len(station_uuids), API_BATCH_SIZE), 1):
        # Get the next batch of up to 10 station UUIDs
        batch = station_uuids[i:i + API_BATCH_SIZE]
        
        # Build API request
        # Join UUIDs with commas: "uuid1,uuid2,uuid3"
        params = {
            "ids": ",".join(batch),
            "apikey": TANKERKOENIG_API_KEY
        }
        
        try:
            # Make API request
            response = requests.get(
                "https://creativecommons.tankerkoenig.de/json/prices.php",
                params=params,
                timeout=10  # 10 second timeout
            )
            response.raise_for_status()  # Raise exception for 4xx/5xx status codes
            data = response.json()
            
            # Check if API returned an error
            if not data.get("ok", False):
                print(f"  WARNING: API returned error for batch {batch_num}")
                continue
            
            # Extract prices from response
            for station_id, price_data in data.get("prices", {}).items():
                results[station_id] = {
                    'e5': price_data.get('e5'),
                    'e10': price_data.get('e10'),
                    'diesel': price_data.get('diesel'),
                    'status': price_data.get('status', 'unknown'),
                    'is_open': price_data.get('status') == 'open'
                }
            
            # Progress indicator
            if batch_num % 5 == 0 or batch_num == total_batches:
                print(f"  Completed {batch_num}/{total_batches} API calls")
                
        except Exception as e:
            print(f"  WARNING: Error fetching batch {batch_num}: {e}")
        
        # Add delay between batches to respect rate limits
        # Skip delay after the last batch (no need to wait)
        if i + API_BATCH_SIZE < len(station_uuids):
            time.sleep(API_BATCH_DELAY)
    
    # Fill in None for stations not in response
    for uuid in station_uuids:
        if uuid not in results:
            results[uuid] = {
                'e5': None,
                'e10': None,
                'diesel': None,
                'status': 'unknown',
                'is_open': None
            }
    
    return results


# =============================================================================
# SECTION 8: MAIN INTEGRATION FUNCTION
# =============================================================================

def integrate_route_with_prices(
    stations_with_eta: List[Dict],
    use_realtime: bool = False,
    stations_df: Optional[pd.DataFrame] = None
) -> List[Dict]:
    """
    THE CORE INTEGRATION FUNCTION
    ==============================
    
    This is the main function that combines everything:
    1. Station matching (OSM ↔ Tankerkoenig)
    2. Historical price retrieval (Supabase)
    3. Real-time price retrieval (optional, from Tankerkoenig API)
    4. Time cell calculations
    5. Data formatting for the prediction model
    
    INPUT FORMAT (from route_stations.py):
    --------------------------------------
    stations_with_eta = [
        {
            "name": "Aral",                    # OSM station name
            "lat": 48.51279,                   # GPS latitude
            "lon": 9.07341,                    # GPS longitude
            "detour_distance_km": 0.241,       # Detour needed (NEW: renamed from 'distance')
            "detour_duration_min": 2.55,       # Time for detour (NEW)
            "distance_along_m": 3058,          # Distance from start along route
            "fraction_of_route": 0.201,        # How far along route (0-1)
            "eta": "2025-11-20T14:35:49"       # Estimated arrival time
        },
        ...
    ]
    
    OUTPUT FORMAT (for prediction model):
    -------------------------------------
    [
        {
            # Original route info
            "osm_name": "Aral",
            "lat": 48.51279,
            "lon": 9.07341,
            "detour_distance_km": 0.241,        # Updated field name
            "detour_duration_min": 2.55,        # New field
            "distance_along_m": 3058,
            "fraction_of_route": 0.201,
            "eta": "2025-11-20T14:35:49",
            
            # Tankerkoenig match info
            "station_uuid": "44e2bdb7-...",
            "tk_name": "Aral Tuebingen Nord",
            "brand": "ARAL",
            "city": "Tuebingen",
            "tk_latitude": 48.51295,
            "tk_longitude": 9.073625,
            "match_distance_m": 12.5,
            
            # Model inputs
            "time_cell": 29,
            
            # Inter-day lags (different dates, same time)
            "price_lag_1d_e5": 1.729,
            "price_lag_1d_e10": 1.669,
            "price_lag_1d_diesel": 1.589,
            
            "price_lag_2d_e5": 1.719,
            "price_lag_2d_e10": 1.659,
            "price_lag_2d_diesel": 1.579,
            
            "price_lag_3d_e5": 1.715,
            "price_lag_3d_e10": 1.655,
            "price_lag_3d_diesel": 1.575,
            
            "price_lag_7d_e5": 1.709,
            "price_lag_7d_e10": 1.649,
            "price_lag_7d_diesel": 1.569,
            
            # Current/predicted prices
            "price_current_e5": 1.739,
            "price_current_e10": 1.679,
            "price_current_diesel": 1.599,
            
            "is_open": True  # Only if use_realtime=True
        },
        ...
    ]
    
    Args:
        stations_with_eta: List of stations from route_stations.py
        use_realtime: If True, fetch current prices from Tankerkoenig API.
                      If False, use yesterday's price as current price (for demo/testing).
        stations_df: Optional pre-loaded DataFrame of Tankerkoenig stations.
                     If None, will load from Supabase.
    
    Returns:
        List of dictionaries (one per matched station) with all data needed for prediction
    
    Performance:
        - Without optimization: ~60-80 seconds for 5 stations
        - With batch queries: ~10-15 seconds for 5 stations
        - Main bottleneck: Database queries for historical prices
    """
    
    print("\n" + "=" * 70)
    print("ROUTE-TANKERKOENIG INTEGRATION")
    print("=" * 70)
    print(f"Mode: {'REAL-TIME' if use_realtime else 'HISTORICAL (yesterday = current)'}")
    print(f"Input stations: {len(stations_with_eta)}")
    
    # =================================================================
    # STEP 1: Load Tankerkoenig stations from Supabase (if not provided)
    # =================================================================
    if stations_df is None:
        stations_df = load_all_stations_from_supabase()
    
    # =================================================================
    # STEP 2: Match OSM stations to Tankerkoenig stations
    # =================================================================
    import time as time_module
    
    print("\nMatching stations to Tankerkoenig database...")
    match_start = time_module.time()
    
    matched_stations = []
    unmatched_count = 0
    
    for station in stations_with_eta:
        # Calculate time cell from ETA
        time_cell = calculate_time_cell(station['eta'])
        
        # Try to match this OSM station to a Tankerkoenig station
        match = match_coordinates_progressive(
            station['lat'],
            station['lon'],
            stations_df
        )
        
        if not match:
            # No Tankerkoenig station found within 100m
            print(f"  WARNING: No match for '{station['name']}' at ({station['lat']}, {station['lon']})")
            unmatched_count += 1
            continue
        
        # Build result dictionary with all available data
        # Note: Handle both old and new field names from route_stations.py
        result = {
            # Original OSM data
            'osm_name': station['name'],
            'lat': station['lat'],
            'lon': station['lon'],
            'distance_along_m': station.get('distance_along_m', 0),
            'fraction_of_route': station.get('fraction_of_route', None),  # Optional
            'eta': station['eta'],
            
            # New fields from Google API (if available)
            'detour_distance_km': station.get('detour_distance_km', station.get('distance', None)),
            'detour_duration_min': station.get('detour_duration_min', None),
            
            # Tankerkoenig match data
            'station_uuid': match['station_uuid'],
            'tk_name': match['tk_name'],
            'brand': match['brand'],
            'city': match['city'],
            'tk_latitude': match['latitude'],
            'tk_longitude': match['longitude'],
            'match_distance_m': match['match_distance_m'],
            
            # Model inputs
            'time_cell': time_cell,
        }
        
        matched_stations.append(result)
    
    match_elapsed = time_module.time() - match_start
    print(f"Matched: {len(matched_stations)}, Unmatched: {unmatched_count} (took {match_elapsed:.1f}s)")
    
    if not matched_stations:
        print("ERROR: No stations matched. Cannot proceed.")
        return []
    
    # =================================================================
    # STEP 3: Get historical prices from Supabase (BATCH QUERIES)
    # =================================================================
    print("\nFetching historical prices from Supabase...")
    
    # Calculate dates for historical lookups
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)
    three_days_ago = today - timedelta(days=3)
    seven_days_ago = today - timedelta(days=7)
    
    # Get all station UUIDs
    all_uuids = [s['station_uuid'] for s in matched_stations]
    
    # Get the most common time cell (use for all queries)
    common_cell = matched_stations[0]['time_cell']
    
    # Make 4 batch queries (one for each lag period) - IN PARALLEL!
    # This runs all 4 queries simultaneously instead of one-by-one
    from concurrent.futures import ThreadPoolExecutor
    import time as time_module
    
    print("Querying all lag periods in parallel...")
    start_time = time_module.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all 4 queries at once
        future_1d = executor.submit(get_historical_prices_batch, all_uuids, yesterday, common_cell)
        future_2d = executor.submit(get_historical_prices_batch, all_uuids, two_days_ago, common_cell)
        future_3d = executor.submit(get_historical_prices_batch, all_uuids, three_days_ago, common_cell)
        future_7d = executor.submit(get_historical_prices_batch, all_uuids, seven_days_ago, common_cell)
        
        # Wait for all to complete
        prices_1d = future_1d.result()
        prices_2d = future_2d.result()
        prices_3d = future_3d.result()
        prices_7d = future_7d.result()
    
    elapsed = time_module.time() - start_time
    print(f"Retrieved prices in {elapsed:.1f}s (parallel execution)")
    
    # Add historical prices to each station
    for station in matched_stations:
        uuid = station['station_uuid']
        
        # Get prices for this station from batch results
        p1d = prices_1d.get(uuid, {'e5': None, 'e10': None, 'diesel': None})
        p2d = prices_2d.get(uuid, {'e5': None, 'e10': None, 'diesel': None})
        p3d = prices_3d.get(uuid, {'e5': None, 'e10': None, 'diesel': None})
        p7d = prices_7d.get(uuid, {'e5': None, 'e10': None, 'diesel': None})
        
        # Add to station dictionary
        station['price_lag_1d_e5'] = p1d['e5']
        station['price_lag_1d_e10'] = p1d['e10']
        station['price_lag_1d_diesel'] = p1d['diesel']
        
        station['price_lag_2d_e5'] = p2d['e5']
        station['price_lag_2d_e10'] = p2d['e10']
        station['price_lag_2d_diesel'] = p2d['diesel']
        
        station['price_lag_3d_e5'] = p3d['e5']
        station['price_lag_3d_e10'] = p3d['e10']
        station['price_lag_3d_diesel'] = p3d['diesel']
        
        station['price_lag_7d_e5'] = p7d['e5']
        station['price_lag_7d_e10'] = p7d['e10']
        station['price_lag_7d_diesel'] = p7d['diesel']
    
    print("Retrieved historical prices for all stations")
    
    # =================================================================
    # STEP 4: Get current prices (real-time or use yesterday)
    # =================================================================
    if use_realtime:
        print("\nFetching current prices from Tankerkoenig API...")
        realtime_prices = get_realtime_prices_batch(all_uuids)
        
        for station in matched_stations:
            uuid = station['station_uuid']
            prices = realtime_prices.get(uuid, {})
            
            # Add current prices
            station['price_current_e5'] = prices.get('e5')
            station['price_current_e10'] = prices.get('e10')
            station['price_current_diesel'] = prices.get('diesel')
            station['is_open'] = prices.get('is_open')
    else:
        print("\nUsing yesterday's prices as current prices (demo mode)...")
        # In demo mode, use yesterday's price as the "current" price
        # This lets us test without making API calls
        for station in matched_stations:
            station['price_current_e5'] = station['price_lag_1d_e5']
            station['price_current_e10'] = station['price_lag_1d_e10']
            station['price_current_diesel'] = station['price_lag_1d_diesel']
            station['is_open'] = None  # Unknown in demo mode
    
    # =================================================================
    # STEP 5: Filter out stations with missing data
    # =================================================================
    # Only return stations that have complete price data
    # The model can't make predictions without historical prices
    complete_stations = [
        s for s in matched_stations
        if s['price_lag_1d_e5'] is not None and s['price_lag_7d_e5'] is not None
    ]
    
    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"Total stations processed: {len(matched_stations)}")
    print(f"Stations with complete price data: {len(complete_stations)}")
    
    return complete_stations


# =============================================================================
# SECTION 9: COMPLETE PIPELINE FUNCTION (For Streamlit/Production)
# =============================================================================

def get_fuel_prices_for_route(
    start_locality: str,
    end_locality: str,
    start_address: str = "",
    end_address: str = "",
    use_realtime: bool = False
) -> List[Dict]:
    """
    COMPLETE END-TO-END PIPELINE
    ============================
    
    This is the highest-level function that does everything:
    1. Geocode start and end addresses
    2. Calculate route between them
    3. Find fuel stations along the route
    4. Calculate ETAs for each station
    5. Match stations to Tankerkoenig database
    6. Retrieve historical prices
    7. Optionally get real-time prices
    8. Return model-ready data
    
    This is what needs to be called from the Streamlit app
    
    Args:
        start_locality: Starting city (REQUIRED, e.g., "Tübingen")
        end_locality: Destination city (REQUIRED, e.g., "Reutlingen")
        start_address: Starting street address (optional, e.g., "Wilhelmstraße 7")
        end_address: Destination street address (optional, e.g., "Charlottenstraße 45")
        use_realtime: If True, fetch current prices from Tankerkoenig API
    
    Returns:
        List of dictionaries with complete data for each station
        (same format as integrate_route_with_prices() output)
    
    Example:
        >>> result = get_fuel_prices_for_route(
        ...     start_locality="Tübingen",
        ...     end_locality="Reutlingen",
        ...     start_address="Wilhelmstraße 7",
        ...     end_address="Charlottenstraße 45",
        ...     use_realtime=False
        ... )
        >>> len(result)
        5
        >>> result[0]['tk_name']
        'Aral Tankstelle'
        >>> result[0]['price_lag_1d_e5']
        1.759
    
    Raises:
        ImportError: If route_stations.py functions are not available
        Exception: If any step in the pipeline fails
    """
    
    # Check if we have the required functions from route_stations.py
    if not _ROUTE_FUNCTIONS_IMPORTED:
        raise ImportError(
            "Could not import functions from route_stations.py. "
            "Make sure route_stations.py is in the project root."
        )
    
    print("\n" + "=" * 70)
    print("COMPLETE FUEL PRICE PIPELINE")
    print("=" * 70)
    print(f"Route: {start_locality} → {end_locality}")
    print(f"Mode: {'REAL-TIME' if use_realtime else 'HISTORICAL (demo)'}")
    
    # =================================================================
    # STEP 1: Geocode addresses to get coordinates
    # =================================================================
    print("\nStep 1: Geocoding addresses...")
    
    # Get coordinates for start location
    start_lat, start_lon, start_label = google_geocode_structured(
        start_address, start_locality, "Germany", GOOGLE_API_KEY
    )
    
    # Get coordinates for end location
    end_lat, end_lon, end_label = google_geocode_structured(
        end_address, end_locality, "Germany", GOOGLE_API_KEY
    )
    
    print(f"  Start: {start_label} ({start_lat:.5f}, {start_lon:.5f})")
    print(f"  End: {end_label} ({end_lat:.5f}, {end_lon:.5f})")
    
    # =================================================================
    # STEP 2: Calculate route between start and end
    # =================================================================
    print("\nStep 2: Calculating route...")
    
    # Get route coordinates, distance, duration, and departure time
    # Note: google_route_driving_car returns 4 values (updated by A)
    route_coords, route_km, route_min, departure_time = google_route_driving_car(
        start_lat, start_lon, end_lat, end_lon, GOOGLE_API_KEY
    )
    
    print(f"  Distance: {route_km:.1f} km")
    print(f"  Duration: {route_min:.0f} min")
    
    # =================================================================
    # STEP 3: Find fuel stations along the route
    # =================================================================
    print("\nStep 3: Finding fuel stations along route...")
    
    # Find stations using Google Places API
    # Note: departure_time is already calculated by google_route_driving_car
    stations = google_places_fuel_along_route(
        route_coords,
        GOOGLE_API_KEY,
        route_km,
        route_min,
        departure_time
    )
    
    print(f"  Found {len(stations)} stations (with ETAs)")
    
    if not stations:
        print("  WARNING: No fuel stations found along this route")
        return []
    
    # Note: google_places_fuel_along_route already includes ETAs,
    # so we can skip the separate estimate_arrival_times step
    stations_with_eta = stations
    
    # =================================================================
    # STEP 4: Run the integration (match stations + get prices)
    # =================================================================
    print("\nStep 4: Matching to Tankerkoenig database and fetching prices...")
    
    model_input = integrate_route_with_prices(
        stations_with_eta=stations_with_eta,
        use_realtime=use_realtime
    )
    
    # =================================================================
    # DONE!
    # =================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total stations with complete data: {len(model_input)}")
    
    return model_input


# =============================================================================
# SECTION 10: EXAMPLE/TEST FUNCTIONS
# =============================================================================

def run_example():
    """
    Simple example using test data (no route_stations.py needed).
    
    This demonstrates the integration function with hardcoded example data.
    Useful for:
    - Testing the integration without depending on route_stations.py
    - Demonstrating the expected input/output format
    - Debugging database or API issues
    
    Returns:
        List of model-ready dictionaries
    """
    print("\n" + "=" * 70)
    print("SIMPLE EXAMPLE (Using Test Data)")
    print("=" * 70)
    
    # Example stations (as if they came from route_stations.py)
    # These are real stations on the Tübingen → Reutlingen route
    example_stations_with_eta = [
        {
            "name": "Aral",
            "lat": 48.51279,
            "lon": 9.07341,
            "detour_distance_km": 0.241,
            "detour_duration_min": 2.55,
            "distance_along_m": 3058,
            "fraction_of_route": 0.201,
            "eta": "2025-11-23T14:35:49"
        },
        {
            "name": "Esso",
            "lat": 48.49228,
            "lon": 9.20297,
            "detour_distance_km": 3.126,
            "detour_duration_min": 7.53,
            "distance_along_m": 13494,
            "fraction_of_route": 0.887,
            "eta": "2025-11-23T14:50:46"
        },
        {
            "name": "Jet",
            "lat": 48.49178,
            "lon": 9.19813,
            "detour_distance_km": 1.954,
            "detour_duration_min": 4.53,
            "distance_along_m": 13140,
            "fraction_of_route": 0.864,
            "eta": "2025-11-23T14:50:16"
        }
    ]
    
    # Run the integration
    model_input = integrate_route_with_prices(
        stations_with_eta=example_stations_with_eta,
        use_realtime=False  # Use historical prices (demo mode)
    )
    
    # Display results
    if model_input:
        print("\n" + "-" * 70)
        print("FIRST STATION (All Data):")
        print("-" * 70)
        first = model_input[0]
        print(f"OSM Name: {first['osm_name']}")
        print(f"TK Name: {first['tk_name']}")
        print(f"Brand: {first['brand']}")
        print(f"Time Cell: {first['time_cell']}")
        print(f"Match Distance: {first['match_distance_m']} m")
        print(f"Detour Distance: {first.get('detour_distance_km', 'N/A')} km")
        print(f"Detour Duration: {first.get('detour_duration_min', 'N/A')} min")
        print(f"Current E5: {first['price_current_e5']}")
        print(f"Lag 1d E5: {first['price_lag_1d_e5']}")
        print(f"Lag 2d E5: {first['price_lag_2d_e5']}")
        print(f"Lag 3d E5: {first['price_lag_3d_e5']}")
        print(f"Lag 7d E5: {first['price_lag_7d_e5']}")
        
        print("\n" + "-" * 70)
        print("ALL STATIONS SUMMARY:")
        print("-" * 70)
        
        for station in model_input:
            print(f"{station['tk_name']} ({station['brand']}) - "
                  f"Cell {station['time_cell']}, "
                  f"E5: €{station['price_current_e5']}")
    
    return model_input


# =============================================================================
# SECTION 11: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run this script directly to test the complete pipeline.
    
    Usage:
        python route_tankerkoenig_integration.py
    
    This will:
    1. Try to run the complete pipeline with real routing (requires route_stations.py)
    2. If that fails, fall back to the simple example with test data
    """
    
    print("=" * 70)
    print("TESTING COMPLETE PIPELINE FUNCTION")
    print("=" * 70)
    
    # Test with a real route
    try:
        result = get_fuel_prices_for_route(
            start_locality="Tübingen",
            end_locality="Reutlingen",
            start_address="Wilhelmstraße 7",
            end_address="Charlottenstraße 45",
            use_realtime=False
        )
        
        if result:
            print("\n" + "=" * 70)
            print("SUCCESS: Complete pipeline works!")
            print(f"Generated {len(result)} model input records")
            print("=" * 70)
            
            # Show first station
            print("\nFirst station:")
            for key, value in result[0].items():
                print(f"  {key}: {value}")
        else:
            print("\nNo stations found on this route.")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nFalling back to simple example...")
        
        # Fallback to the simple example
        result = run_example()
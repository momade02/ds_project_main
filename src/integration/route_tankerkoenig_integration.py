"""
Route-Tankerkoenig Integration
==============================

PURPOSE:
--------
This script is the bridge between route_stations.py and the fuel price prediction model.
It takes the output from route_stations.py (gas stations along a route with ETAs) and:
1. Matches each station to a Tankerkoenig station in our Supabase database
2. Retrieves historical prices (yesterday, 7 days ago) from Supabase
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
- see simply_usage_guide.ipynb

CONFIGURATION:
--------------
Required environment variables in .env file:
- SUPABASE_URL: Your Supabase project URL
- SUPABASE_SECRET_KEY: Your Supabase service role key
- TANKERKOENIG_API_KEY: Your Tankerkoenig API key (only needed if use_realtime=True)

FILE LOCATION:
--------------
This file should be placed in: src/integration/route_tankerkoenig_integration.py

IMPORTS FROM route_stations.py:
-------------------------------
Currently assumes route_stations.py is in the project root (DS_PROJECT_MAIN/).
When it moves to src/routing/, uncomment the alternative import path in the code.

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
from dotenv import load_dotenv
from geopy.distance import geodesic

# Load environment variables from .env file
# This contains our API keys and database credentials
load_dotenv()

# Import route_stations.py functions at module level
# Note: This will execute route_stations.py's script-level code once
# To avoid this, route_stations.py should have an if __name__ == "__main__" guard
_ROUTE_FUNCTIONS_IMPORTED = False
try:
    # Add project root to path if needed
    project_root = os.path.dirname(os.path.abspath(__file__))
    if 'src' in project_root:
        project_root = os.path.dirname(os.path.dirname(project_root))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from route_stations import (
        ors_geocode_structured,
        ors_route_driving_car,
        ors_pois_fuel_along_route,
        estimate_arrival_times,
        ors_api_key,
        buffer_meters
    )
    _ROUTE_FUNCTIONS_IMPORTED = True
except ImportError:
    # Will be handled in the function that needs these
    pass

# Supabase credentials for database access
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

# Tankerkoenig API credentials for real-time prices
TANKERKOENIG_API_KEY = os.getenv("TANKERKOENIG_API_KEY")
TANKERKOENIG_BASE_URL = "https://creativecommons.tankerkoenig.de/json"

# Matching thresholds for coordinate matching (in meters)
# We try progressively larger thresholds until we find a match
MATCH_THRESHOLDS = [25, 50, 100]  # meters

# API rate limiting: delay between batch requests to avoid being blocked
API_BATCH_DELAY = 0.5  # seconds between API calls

# Maximum stations per API request (Tankerkoenig limit)
API_BATCH_SIZE = 10


# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================

def calculate_time_cell(eta_string: str) -> int:
    """
    Convert an ETA timestamp to a time cell (0-47).
    
    The prediction model uses 30-minute intervals to represent time of day.
    This function takes an ETA like "2025-11-20T14:35:49" and returns
    which 30-minute cell it falls into.
    
    How it works:
    - Extract the hour and minute from the timestamp
    - Each hour has 2 cells (one for minutes 0-29, one for 30-59)
    - Cell = hour * 2 + (1 if minute >= 30 else 0)
    
    Args:
        eta_string: ISO format timestamp, e.g., "2025-11-20T14:35:49"
    
    Returns:
        Integer from 0 to 47 representing the time cell
    
    Examples:
        "2025-11-20T00:15:00" -> 0   (00:00-00:29)
        "2025-11-20T00:45:00" -> 1   (00:30-00:59)
        "2025-11-20T14:35:49" -> 29  (14:30-14:59)
        "2025-11-20T23:59:59" -> 47  (23:30-23:59)
    """
    # Parse the timestamp string into a datetime object
    # Handle both formats: with and without microseconds
    try:
        if '.' in eta_string:
            # Format with microseconds: "2025-11-20T14:35:49.644499"
            dt = datetime.strptime(eta_string, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            # Format without microseconds: "2025-11-20T14:35:49"
            dt = datetime.strptime(eta_string, "%Y-%m-%dT%H:%M:%S")
    except ValueError as e:
        print(f"WARNING: Could not parse ETA '{eta_string}': {e}")
        return 0  # Default to cell 0 if parsing fails
    
    # Calculate the time cell
    hour = dt.hour
    minute = dt.minute
    
    # Each hour has 2 cells: first half (0-29 min) and second half (30-59 min)
    cell = hour * 2
    if minute >= 30:
        cell += 1
    
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
        0 -> ("00:00:00", "00:29:59")
        29 -> ("14:30:00", "14:59:59")
        47 -> ("23:30:00", "23:59:59")
    """
    # Calculate hour and whether it's the first or second half
    hour = cell // 2
    is_second_half = cell % 2 == 1
    
    if is_second_half:
        start_time = f"{hour:02d}:30:00"
        end_time = f"{hour:02d}:59:59"
    else:
        start_time = f"{hour:02d}:00:00"
        end_time = f"{hour:02d}:29:59"
    
    return start_time, end_time


def match_coordinates_progressive(
    osm_lat: float, 
    osm_lon: float, 
    stations_df: pd.DataFrame,
    thresholds: List[int] = MATCH_THRESHOLDS
) -> Optional[Dict]:
    """
    Find the closest Tankerkoenig station to an OSM coordinate using progressive thresholds.
    
    Why progressive matching?
    - Tankerkoenig coordinates vary in precision (some have 7 decimals, others only 3)
    - Some stations are very close together (e.g., 4 Shell stations within 100m)
    - Starting with a small threshold (25m) ensures we get the closest match
    - If no match at 25m, we try 50m, then 100m
    
    How it works:
    1. Calculate distance from OSM point to ALL Tankerkoenig stations
    2. Find the closest station
    3. Check if it's within the current threshold
    4. If not, try the next (larger) threshold
    5. Return the match info or None if no match found
    
    Args:
        osm_lat: Latitude from route_stations.py
        osm_lon: Longitude from route_stations.py
        stations_df: DataFrame of all Tankerkoenig stations from Supabase
        thresholds: List of distance thresholds to try (in meters)
    
    Returns:
        Dictionary with matched station info, or None if no match found
        {
            'station_uuid': '...',
            'tk_name': '...',
            'brand': '...',
            'city': '...',
            'latitude': ...,
            'longitude': ...,
            'match_distance_m': ...,
            'match_threshold_m': ...
        }
    """
    # The point we're trying to match (from OSM/route_stations.py)
    osm_point = (osm_lat, osm_lon)
    
    # Calculate distance from OSM point to every Tankerkoenig station
    # This uses the geodesic formula which accounts for Earth's curvature
    distances = stations_df.apply(
        lambda row: geodesic(osm_point, (row['latitude'], row['longitude'])).meters,
        axis=1
    )
    
    # Find the index of the closest station
    closest_idx = distances.idxmin()
    closest_distance = distances[closest_idx]
    closest_station = stations_df.loc[closest_idx]
    
    # Try each threshold until we find a match
    for threshold in thresholds:
        if closest_distance <= threshold:
            return {
                'station_uuid': closest_station['uuid'],
                'tk_name': closest_station['name'],
                'brand': closest_station['brand'],
                'city': closest_station['city'],
                'latitude': closest_station['latitude'],
                'longitude': closest_station['longitude'],
                'match_distance_m': round(closest_distance, 2),
                'match_threshold_m': threshold
            }
    
    # No match found within any threshold
    return None


# =============================================================================
# SECTION 3: SUPABASE DATABASE FUNCTIONS
# =============================================================================

def load_all_stations_from_supabase() -> pd.DataFrame:
    """
    Load all Tankerkoenig stations from Supabase database.
    
    Why pagination?
    - Supabase limits query results to 1000 rows by default
    - We have ~17,685 stations, so we need multiple requests
    - We request in batches of 1000 until we get all stations
    
    Returns:
        DataFrame with all station data from Supabase
        Columns: uuid, name, brand, street, house_number, post_code, city, 
                 latitude, longitude, first_active, openingtimes_json
    """
    from supabase import create_client
    
    # Check that we have credentials
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise ValueError(
            "Supabase credentials not found. "
            "Please set SUPABASE_URL and SUPABASE_SECRET_KEY in your .env file."
        )
    
    # Create Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    
    print("Loading all stations from Supabase...")
    
    # Load stations in batches (pagination)
    all_stations = []
    page_size = 1000
    
    for i in range(0, 20000, page_size):  # Max 20,000 stations expected
        result = supabase.table('stations').select('*').range(i, i + page_size - 1).execute()
        
        if not result.data:
            break  # No more data
            
        all_stations.extend(result.data)
        
        # Progress indicator (every 5000 stations)
        if len(all_stations) % 5000 == 0:
            print(f"  Loaded {len(all_stations):,} stations...")
    
    # Convert to DataFrame
    stations_df = pd.DataFrame(all_stations)
    
    # Filter out stations with invalid coordinates
    # Some stations have lat/lon = 0 or NULL, which would cause matching issues
    valid_mask = (
        stations_df['latitude'].notna() &
        stations_df['longitude'].notna() &
        (stations_df['latitude'] != 0) &
        (stations_df['longitude'] != 0) &
        stations_df['latitude'].between(-90, 90) &
        stations_df['longitude'].between(-180, 180)
    )
    
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"  Filtered out {invalid_count} stations with invalid coordinates")
    
    stations_df = stations_df[valid_mask].copy()
    
    print(f"Loaded {len(stations_df):,} valid stations from Supabase")
    
    return stations_df


def get_historical_price_for_station(
    station_uuid: str,
    target_date: datetime,
    target_cell: int
) -> Dict[str, Optional[float]]:
    """
    Get the price that was active for a station at a specific time cell on a specific date.
    
    How fuel prices work:
    - Stations change prices multiple times per day
    - Each price change is recorded with a timestamp
    - To find "the price at 14:30", we need to find the most recent price change
      that happened BEFORE 14:30 on that day
    
    Algorithm:
    1. Get all price records for this station on the target date
    2. Filter to records before or during the target time cell
    3. Take the most recent one (that's the price that was active)
    
    Args:
        station_uuid: Tankerkoenig station UUID
        target_date: The date to look up (datetime object)
        target_cell: The time cell (0-47) to look up
    
    Returns:
        Dictionary with prices: {'e5': 1.739, 'e10': 1.679, 'diesel': 1.599}
        Values are None if no price data found
    """
    from supabase import create_client
    
    supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    
    # Calculate the end time of the target cell
    # We want prices that were set BEFORE or AT this time
    _, end_time = get_cell_time_range(target_cell)
    
    # Build the datetime string for the query
    # Format: "2025-11-19 14:59:59"
    date_str = target_date.strftime("%Y-%m-%d")
    end_datetime = f"{date_str} {end_time}"
    start_datetime = f"{date_str} 00:00:00"
    
    # Query: Get all price changes for this station on this day, up to our target time
    # Then take the most recent one
    try:
        result = supabase.table('prices')\
            .select('date, e5, e10, diesel')\
            .eq('station_uuid', station_uuid)\
            .gte('date', start_datetime)\
            .lte('date', end_datetime)\
            .order('date', desc=True)\
            .limit(1)\
            .execute()
        
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
    target_cells: List[int]
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Get historical prices for multiple stations efficiently.
    
    This is a batch version of get_historical_price_for_station() that
    reduces the number of database queries by fetching all data at once.
    
    Note: This is more complex but much faster for many stations.
    For simplicity, we currently call the single-station function in a loop.
    This could be optimized later if performance becomes an issue.
    
    Args:
        station_uuids: List of Tankerkoenig station UUIDs
        target_date: The date to look up
        target_cells: List of time cells (one per station, matching order)
    
    Returns:
        Dictionary mapping station_uuid to price dict
        {
            'uuid1': {'e5': 1.739, 'e10': 1.679, 'diesel': 1.599},
            'uuid2': {'e5': 1.729, 'e10': 1.669, 'diesel': 1.589},
            ...
        }
    """
    results = {}
    
    for i, station_uuid in enumerate(station_uuids):
        target_cell = target_cells[i]
        prices = get_historical_price_for_station(station_uuid, target_date, target_cell)
        results[station_uuid] = prices
    
    return results


# =============================================================================
# SECTION 4: TANKERKOENIG API FUNCTIONS
# =============================================================================

def get_realtime_prices_batch(station_uuids: List[str]) -> Dict[str, Dict]:
    """
    Get current prices from Tankerkoenig API for multiple stations.
    
    API Details:
    - Endpoint: /prices.php
    - Limit: Maximum 10 stations per request
    - Rate limit: Avoid more than 1 request per 5 minutes for automated systems
    - We add a small delay between batches to be safe
    
    How it works:
    1. Split station UUIDs into batches of 10
    2. For each batch, call the API
    3. Wait 0.5 seconds between batches (to avoid rate limiting)
    4. Combine all results
    
    Args:
        station_uuids: List of Tankerkoenig station UUIDs
    
    Returns:
        Dictionary mapping station_uuid to price/status info
        {
            'uuid1': {'e5': 1.739, 'e10': 1.679, 'diesel': 1.599, 'status': 'open'},
            'uuid2': {'e5': 1.729, 'e10': 1.669, 'diesel': 1.589, 'status': 'closed'},
            ...
        }
    
    Raises:
        ValueError: If API key is not configured
        RuntimeError: If API returns an error
    """
    if not TANKERKOENIG_API_KEY:
        raise ValueError(
            "TANKERKOENIG_API_KEY not found in environment. "
            "Set it in your .env file to use real-time prices."
        )
    
    results = {}
    total_batches = (len(station_uuids) + API_BATCH_SIZE - 1) // API_BATCH_SIZE
    
    print(f"Fetching real-time prices for {len(station_uuids)} stations in {total_batches} API calls...")
    
    # Process in batches of 10 (API limit)
    for batch_num, i in enumerate(range(0, len(station_uuids), API_BATCH_SIZE), 1):
        batch = station_uuids[i:i + API_BATCH_SIZE]
        
        # Build API request
        params = {
            "ids": ",".join(batch),
            "apikey": TANKERKOENIG_API_KEY
        }
        
        try:
            # Call the API
            response = requests.get(f"{TANKERKOENIG_BASE_URL}/prices.php", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if API call succeeded
            if not data.get("ok"):
                error_msg = data.get('message', 'Unknown error')
                print(f"WARNING: API error in batch {batch_num}: {error_msg}")
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
            
        except requests.RequestException as e:
            print(f"WARNING: Network error in batch {batch_num}: {e}")
            continue
        
        # Delay between batches to avoid rate limiting
        # (Skip delay after the last batch)
        if i + API_BATCH_SIZE < len(station_uuids):
            time.sleep(API_BATCH_DELAY)
    
    print(f"Retrieved prices for {len(results)} stations")
    
    return results


# =============================================================================
# SECTION 5: MAIN INTEGRATION FUNCTION
# =============================================================================

def integrate_route_with_prices(
    stations_with_eta: List[Dict],
    use_realtime: bool = False,
    stations_df: Optional[pd.DataFrame] = None
) -> List[Dict]:
    """
    Main integration function: Connect route_stations.py output with Tankerkoenig data.
    
    This is the function you call to prepare data for the prediction model.
    It takes the output from route_stations.py and enriches it with:
    - Tankerkoenig station info (UUID, official name, brand)
    - Time cell calculation
    - Current price (real-time from API or yesterday's from database)
    - Historical prices (yesterday and 7 days ago)
    
    Workflow:
    1. Load Tankerkoenig stations from Supabase (if not provided)
    2. For each station from route_stations.py:
       a. Match to Tankerkoenig station using coordinates
       b. Calculate time cell from ETA
       c. Get historical prices from Supabase
       d. Get current price (API or yesterday's price)
    3. Return enriched data ready for the model
    
    Args:
        stations_with_eta: Output from route_stations.py
            Format: [
                {
                    "name": "Aral",
                    "lat": 48.51279,
                    "lon": 9.07341,
                    "distance": 45.7,
                    "distance_along_m": 3058,
                    "fraction_of_route": 0.201,
                    "eta": "2025-11-20T14:35:49.644499"
                },
                ...
            ]
        
        use_realtime: If True, fetch current prices from Tankerkoenig API.
                      If False, use yesterday's price as current price (for demo/testing).
        
        stations_df: Optional pre-loaded DataFrame of Tankerkoenig stations.
                     If None, will load from Supabase.
    
    Returns:
        List of dictionaries, one per matched station:
        [
            {
                # Original data from route_stations.py
                "osm_name": "Aral",
                "lat": 48.51279,
                "lon": 9.07341,
                "distance_along_m": 3058,
                "fraction_of_route": 0.201,
                "eta": "2025-11-20T14:35:49.644499",
                
                # Tankerkoenig match info
                "station_uuid": "44e2bdb7-...",
                "tk_name": "Aral Tuebingen Nord",
                "brand": "ARAL",
                "city": "Tuebingen",
                "match_distance_m": 12.5,
                
                # Model inputs
                "time_cell": 29,
                "price_current_e5": 1.739,
                "price_current_e10": 1.679,
                "price_current_diesel": 1.599,
                "price_yesterday_e5": 1.729,
                "price_yesterday_e10": 1.669,
                "price_yesterday_diesel": 1.589,
                "price_7days_e5": 1.719,
                "price_7days_e10": 1.659,
                "price_7days_diesel": 1.579,
                "is_open": True
            },
            ...
        ]
    """
    print("\n" + "=" * 70)
    print("ROUTE-TANKERKOENIG INTEGRATION")
    print("=" * 70)
    print(f"Mode: {'REAL-TIME API' if use_realtime else 'HISTORICAL (yesterday = current)'}")
    print(f"Input stations: {len(stations_with_eta)}")
    
    # Step 1: Load Tankerkoenig stations from Supabase (if not provided)
    if stations_df is None:
        stations_df = load_all_stations_from_supabase()
    
    # Step 2: Match each OSM station to a Tankerkoenig station
    print("\nMatching stations to Tankerkoenig database...")
    
    matched_stations = []
    unmatched_count = 0
    
    for osm_station in stations_with_eta:
        # Try to match this OSM station to a Tankerkoenig station
        match = match_coordinates_progressive(
            osm_lat=osm_station['lat'],
            osm_lon=osm_station['lon'],
            stations_df=stations_df
        )
        
        if match is None:
            # No match found - warn and skip
            unmatched_count += 1
            print(f"  WARNING: No match for '{osm_station.get('name', 'Unnamed')}' "
                  f"at ({osm_station['lat']:.5f}, {osm_station['lon']:.5f})")
            continue
        
        # Calculate time cell from ETA
        time_cell = calculate_time_cell(osm_station['eta'])
        
        # Build the result dictionary
        result = {
            # Original data from route_stations.py
            'osm_name': osm_station.get('name', 'Unnamed'),
            'lat': osm_station['lat'],
            'lon': osm_station['lon'],
            'distance_along_m': osm_station.get('distance_along_m', 0),
            'fraction_of_route': osm_station.get('fraction_of_route', 0),
            'eta': osm_station['eta'],
            
            # Tankerkoenig match info
            'station_uuid': match['station_uuid'],
            'tk_name': match['tk_name'],
            'brand': match['brand'],
            'city': match['city'],
            'tk_latitude': match['latitude'],
            'tk_longitude': match['longitude'],
            'match_distance_m': match['match_distance_m'],
            
            # Model input: time cell
            'time_cell': time_cell
        }
        
        matched_stations.append(result)
    
    print(f"Matched: {len(matched_stations)}, Unmatched: {unmatched_count}")
    
    if not matched_stations:
        print("ERROR: No stations matched. Cannot proceed.")
        return []
    
    # Step 3: Get historical prices from Supabase
    print("\nFetching historical prices from Supabase...")
    
    # Calculate dates for historical lookups
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    seven_days_ago = today - timedelta(days=7)
    
    # Get prices for each station
    for station in matched_stations:
        uuid = station['station_uuid']
        cell = station['time_cell']
        
        # Get yesterday's price at the same time cell
        yesterday_prices = get_historical_price_for_station(uuid, yesterday, cell)
        station['price_yesterday_e5'] = yesterday_prices['e5']
        station['price_yesterday_e10'] = yesterday_prices['e10']
        station['price_yesterday_diesel'] = yesterday_prices['diesel']
        
        # Get price from 7 days ago at the same time cell
        week_ago_prices = get_historical_price_for_station(uuid, seven_days_ago, cell)
        station['price_7days_e5'] = week_ago_prices['e5']
        station['price_7days_e10'] = week_ago_prices['e10']
        station['price_7days_diesel'] = week_ago_prices['diesel']
    
    print(f"Retrieved historical prices for {len(matched_stations)} stations")
    
    # Step 4: Get current prices
    if use_realtime:
        # Fetch from Tankerkoenig API
        print("\nFetching real-time prices from Tankerkoenig API...")
        
        station_uuids = [s['station_uuid'] for s in matched_stations]
        realtime_prices = get_realtime_prices_batch(station_uuids)
        
        # Add real-time prices to results
        for station in matched_stations:
            uuid = station['station_uuid']
            if uuid in realtime_prices:
                prices = realtime_prices[uuid]
                station['price_current_e5'] = prices['e5']
                station['price_current_e10'] = prices['e10']
                station['price_current_diesel'] = prices['diesel']
                station['is_open'] = prices.get('is_open', None)
            else:
                # API didn't return this station - use None
                station['price_current_e5'] = None
                station['price_current_e10'] = None
                station['price_current_diesel'] = None
                station['is_open'] = None
    else:
        # Use yesterday's price as current price (for demo/testing)
        print("\nUsing yesterday's prices as current prices (demo mode)...")
        
        for station in matched_stations:
            station['price_current_e5'] = station['price_yesterday_e5']
            station['price_current_e10'] = station['price_yesterday_e10']
            station['price_current_diesel'] = station['price_yesterday_diesel']
            station['is_open'] = None  # Unknown in demo mode
    
    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"Total stations processed: {len(matched_stations)}")
    
    # Count stations with complete price data
    complete_count = sum(
        1 for s in matched_stations 
        if s['price_current_e5'] is not None and s['price_yesterday_e5'] is not None
    )
    print(f"Stations with complete price data: {complete_count}")
    
    return matched_stations


# =============================================================================
# SECTION 6: EXAMPLE USAGE AND TESTING
# =============================================================================

def run_example():
    """
    Run an example integration using sample data.
    
    This demonstrates how to use the integration function and shows the output format.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Route-Tankerkoenig Integration")
    print("=" * 70)
    
    # Example output from route_stations.py
    # This is what you would get from running route_stations.py
    example_stations_with_eta = [
        {
            "name": "Aral",
            "lat": 48.51279,
            "lon": 9.07341,
            "distance": 45.70855075,
            "distance_along_m": 3058,
            "fraction_of_route": 0.201,
            "eta": "2025-11-20T14:35:49.644499"
        },
        {
            "name": "Esso",
            "lat": 48.49228,
            "lon": 9.20297,
            "distance": 33.79514482,
            "distance_along_m": 13494,
            "fraction_of_route": 0.887,
            "eta": "2025-11-20T14:50:46.836178"
        },
        {
            "name": "Jet",
            "lat": 48.49178,
            "lon": 9.19813,
            "distance": 29.29118326,
            "distance_along_m": 13140,
            "fraction_of_route": 0.864,
            "eta": "2025-11-20T14:50:16.412891"
        }
    ]
    
    # Run integration in demo mode (no API calls)
    model_input = integrate_route_with_prices(
        stations_with_eta=example_stations_with_eta,
        use_realtime=False  # Use historical data only
    )
    
    # Display results
    if model_input:
        print("\n" + "-" * 70)
        print("MODEL INPUT DATA (first station):")
        print("-" * 70)
        
        first = model_input[0]
        print(f"OSM Name: {first['osm_name']}")
        print(f"TK Name: {first['tk_name']}")
        print(f"Brand: {first['brand']}")
        print(f"Time Cell: {first['time_cell']}")
        print(f"Match Distance: {first['match_distance_m']} m")
        print(f"Current E5: {first['price_current_e5']}")
        print(f"Yesterday E5: {first['price_yesterday_e5']}")
        print(f"7 Days E5: {first['price_7days_e5']}")
        
        print("\n" + "-" * 70)
        print("ALL STATIONS SUMMARY:")
        print("-" * 70)
        
        for station in model_input:
            print(f"{station['tk_name']} ({station['brand']}) - "
                  f"Cell {station['time_cell']}, "
                  f"E5: {station['price_current_e5']}")
    
    return model_input


# =============================================================================
# SECTION 7: COMPLETE PIPELINE FUNCTION (For Streamlit/Production)
# =============================================================================

def get_fuel_prices_for_route(
    start_locality: str,
    end_locality: str,
    start_address: str = "",
    end_address: str = "",
    start_country: str = "Germany",
    end_country: str = "Germany",
    use_realtime: bool = False
) -> List[Dict]:
    """
    COMPLETE PIPELINE: From addresses to model-ready data.
    
    This is the ONE function you call for the full pipeline.
    Perfect for Streamlit dashboard or production use.
    
    What it does (everything automatically):
    1. Geocode the start and end addresses
    2. Calculate the driving route
    3. Find fuel stations along the route
    4. Calculate ETAs for each station
    5. Match stations to Tankerkoenig database
    6. Get historical prices (yesterday, 7 days ago)
    7. Optionally get real-time prices from API
    8. Return data ready for the prediction model
    
    Args:
        start_locality: Starting city/town (e.g., "Tübingen") - REQUIRED
                        Note: This maps to ORS 'locality' parameter
        end_locality: Ending city/town (e.g., "Reutlingen") - REQUIRED
                      Note: This maps to ORS 'locality' parameter
        start_address: Starting street address (e.g., "Wilhelmstraße 7") - OPTIONAL
                       If empty, geocodes to city center
        end_address: Ending street address (e.g., "Charlottenstraße 45") - OPTIONAL
                     If empty, geocodes to city center
        start_country: Starting country (default: "Germany")
        end_country: Ending country (default: "Germany")
        use_realtime: If True, fetch current prices from API. If False, use yesterday's prices.
    
    Returns:
        List of dictionaries with all data for the prediction model.
        Same format as integrate_route_with_prices() output.
    
    Example usage:
        # With full addresses (recommended for precision):
        model_input = get_fuel_prices_for_route(
            start_locality="Tübingen",
            end_locality="Reutlingen",
            start_address="Wilhelmstraße 7",
            end_address="Charlottenstraße 45",
            use_realtime=False
        )
        
        # With just city names (uses city centers):
        model_input = get_fuel_prices_for_route(
            start_locality="Tübingen",
            end_locality="Reutlingen",
            use_realtime=False
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(model_input)
        
        # Pass to model
        predictions = model.predict(df)
    
    Raises:
        ImportError: If route_stations.py cannot be imported
        RuntimeError: If geocoding or route calculation fails
    
    Note: Parameter names match route_stations.py exactly:
          - start_locality/end_locality (not start_city/end_city)
          - These map to the ORS API 'locality' parameter
    """
    print("\n" + "=" * 70)
    print("COMPLETE FUEL PRICE PIPELINE")
    print("=" * 70)
    print(f"Route: {start_locality} → {end_locality}")
    print(f"Mode: {'REAL-TIME API' if use_realtime else 'HISTORICAL (demo)'}")
    
    # Check if route_stations functions are available
    if not _ROUTE_FUNCTIONS_IMPORTED:
        raise ImportError(
            "Could not import route_stations.py functions.\n"
            "Make sure route_stations.py is in the project root."
        )
    
    # Step 1: Geocode addresses
    print("\nStep 1: Geocoding addresses...")
    try:
        start_lat, start_lon, start_label = ors_geocode_structured(
            start_address, start_locality, start_country, ors_api_key
        )
        end_lat, end_lon, end_label = ors_geocode_structured(
            end_address, end_locality, end_country, ors_api_key
        )
        print(f"  Start: {start_label} ({start_lat:.5f}, {start_lon:.5f})")
        print(f"  End: {end_label} ({end_lat:.5f}, {end_lon:.5f})")
    except Exception as e:
        raise RuntimeError(f"Geocoding failed: {e}")
    
    # Step 2: Calculate route
    print("\nStep 2: Calculating route...")
    try:
        route_coords, route_km, route_min = ors_route_driving_car(
            start_lat, start_lon, end_lat, end_lon, ors_api_key
        )
        print(f"  Distance: {route_km:.1f} km")
        print(f"  Duration: {route_min:.0f} min")
    except Exception as e:
        raise RuntimeError(f"Route calculation failed: {e}")
    
    # Step 3: Find fuel stations along route
    print("\nStep 3: Finding fuel stations along route...")
    print(f"  Route has {len(route_coords)} coordinate points")
    print(f"  Buffer: {buffer_meters}m, Route length: {route_km:.1f}km")
    
    try:
        stations = ors_pois_fuel_along_route(
            route_coords, buffer_meters, ors_api_key, route_km
        )
        print(f"  Found {len(stations)} stations")
        
        if len(stations) == 0:
            print("\n  ⚠ WARNING: No stations found along route!")
            print("  Possible reasons:")
            print("    1. ORS POI database doesn't have stations for this area")
            print("    2. Buffer distance is too small (current: {}m)".format(buffer_meters))
            print("    3. Route is very short")
            print("    4. ORS API rate limit reached")
            print("\n  TIP: Try running route_stations.py directly to verify it works.")
            return []
    except Exception as e:
        raise RuntimeError(f"Station search failed: {e}")
    
    # Step 4: Calculate ETAs
    print("\nStep 4: Calculating arrival times...")
    try:
        stations_with_eta = estimate_arrival_times(stations, route_coords, route_km, route_min)
        print(f"  Calculated ETAs for {len(stations_with_eta)} stations")
    except Exception as e:
        raise RuntimeError(f"ETA calculation failed: {e}")
    
    # Step 5: Integrate with Tankerkoenig data
    print("\nStep 5: Matching to Tankerkoenig database and fetching prices...")
    try:
        model_input = integrate_route_with_prices(
            stations_with_eta=stations_with_eta,
            use_realtime=use_realtime
        )
    except Exception as e:
        raise RuntimeError(f"Tankerkoenig integration failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total stations with complete data: {len(model_input)}")
    
    return model_input


# =============================================================================
# SECTION 8: INTEGRATION WITH route_stations.py (Legacy Examples)
# =============================================================================

def run_full_integration_example():
    """
    Example of running the full pipeline: route_stations.py -> integration -> model input.
    
    This shows how to import and use functions from route_stations.py together
    with this integration script.
    
    Note: This requires route_stations.py to be importable.
    """
    print("\n" + "=" * 70)
    print("FULL INTEGRATION EXAMPLE")
    print("=" * 70)
    
    # Import functions from route_stations.py
    # CURRENT: route_stations.py is in project root
    # Add project root to path if needed
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        # Import the functions we need from route_stations.py
        from route_stations import (
            ors_geocode_structured,
            ors_route_driving_car,
            ors_pois_fuel_along_route,
            estimate_arrival_times,
            ors_api_key,
            buffer_meters
        )
        print("Successfully imported from route_stations.py")
        
    except ImportError as e:
        print(f"Could not import from route_stations.py: {e}")
        print("Make sure route_stations.py is in the project root or adjust the import path.")
        print("\nFalling back to example data...")
        return run_example()
    
    # FUTURE: When route_stations.py moves to src/routing/
    # Uncomment the following and comment out the above:
    # from src.routing.route_stations import (
    #     ors_geocode_structured,
    #     ors_route_driving_car,
    #     ors_pois_fuel_along_route,
    #     estimate_arrival_times,
    #     ors_api_key,
    #     buffer_meters
    # )
    
    # Define route (example: Tuebingen to Reutlingen)
    start_address = "Wilhelmstraße 7"
    start_locality = "Tübingen"
    start_country = "Germany"
    end_address = "Charlottenstraße 45"
    end_locality = "Reutlingen"
    end_country = "Germany"
    
    print(f"\nRoute: {start_address}, {start_locality} -> {end_address}, {end_locality}")
    
    # Step 1: Geocode addresses
    print("\nStep 1: Geocoding addresses...")
    start_lat, start_lon, start_label = ors_geocode_structured(
        start_address, start_locality, start_country, ors_api_key
    )
    end_lat, end_lon, end_label = ors_geocode_structured(
        end_address, end_locality, end_country, ors_api_key
    )
    
    print(f"  Start: {start_label} ({start_lat:.5f}, {start_lon:.5f})")
    print(f"  End: {end_label} ({end_lat:.5f}, {end_lon:.5f})")
    
    # Step 2: Get route
    print("\nStep 2: Calculating route...")
    route_coords, route_km, route_min = ors_route_driving_car(
        start_lat, start_lon, end_lat, end_lon, ors_api_key
    )
    print(f"  Distance: {route_km:.1f} km, Duration: {route_min:.0f} min")
    
    # Step 3: Find fuel stations along route
    print("\nStep 3: Finding fuel stations along route...")
    stations = ors_pois_fuel_along_route(
        route_coords, buffer_meters, ors_api_key, route_km
    )
    print(f"  Found {len(stations)} stations")
    
    # Step 4: Calculate ETAs
    print("\nStep 4: Calculating ETAs...")
    stations_with_eta = estimate_arrival_times(stations, route_coords, route_km, route_min)
    
    # Step 5: Run integration
    print("\nStep 5: Running Tankerkoenig integration...")
    model_input = integrate_route_with_prices(
        stations_with_eta=stations_with_eta,
        use_realtime=False  # Set to True for live API prices
    )
    
    return model_input


if __name__ == "__main__":
    """
    Run this script directly to test the integration.
    
    Usage:
        python route_tankerkoenig_integration.py
    
    This will run the complete pipeline with a test route.
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
    
    # Uncomment to run the full integration with route_stations.py:
    # result = run_full_integration_example()
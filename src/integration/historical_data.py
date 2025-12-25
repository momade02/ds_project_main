"""
Historical Data Module
======================

Fetches 14-day price histories from Supabase for station detail visualizations.
Includes opening hours parsing from Tankerkönig database JSON format.

Functions
---------
- get_station_price_history: Fetch full price time-series for a station
- get_opening_hours_display: Parse complex opening hours JSON into readable format
- calculate_hourly_price_stats: Aggregate prices by hour-of-day for optimal time analysis

**FIXED FOR PANDAS 2.X:**
- Added `utc=False` and `errors='coerce'` to pd.to_datetime()
- This prevents "Addition/subtraction of integers and integer-arrays with Timestamp" errors
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from supabase import create_client

from src.app.app_errors import DataAccessError, DataQualityError


# Initialize Supabase client (reads from environment variables)
def _get_supabase_client():
    """Get Supabase client from environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")
    
    if not url or not key:
        raise DataAccessError(
            user_message="Database connection is not configured.",
            remediation="Check that SUPABASE_URL and SUPABASE_SECRET_KEY are set in environment variables.",
            details="Missing Supabase credentials in environment."
        )
    
    return create_client(url, key)


def get_station_price_history(
    station_uuid: str,
    fuel_type: str = "e5",
    days: int = 14
) -> pd.DataFrame:
    """
    Fetch full price history for a station from Supabase.
    
    This retrieves ALL price records (not just snapshots at specific lags)
    for plotting continuous time-series trends.
    
    Parameters
    ----------
    station_uuid : str
        Tankerkönig station UUID
    fuel_type : str
        One of 'e5', 'e10', 'diesel'
    days : int
        Number of days of history to retrieve (default: 14)
    
    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'price', 'price_change']
        Sorted by date ascending
        Empty DataFrame if no data found
    
    Raises
    ------
    DataAccessError
        If database connection fails
    DataQualityError
        If fuel_type is invalid
    
    Examples
    --------
    >>> df = get_station_price_history("abc-123", fuel_type="e5", days=7)
    >>> df.head()
                         date  price  price_change
    0  2025-12-06 00:01:26   1.639             0
    1  2025-12-06 06:15:42   1.649             1
    2  2025-12-06 12:30:18   1.639            -1
    """
    # Validate fuel type
    valid_fuels = ["e5", "e10", "diesel"]
    if fuel_type not in valid_fuels:
        raise DataQualityError(
            user_message=f"Invalid fuel type '{fuel_type}'.",
            remediation=f"Choose one of: {', '.join(valid_fuels)}",
            details=f"fuel_type must be in {valid_fuels}"
        )
    
    try:
        supabase = _get_supabase_client()
        
        # Calculate date range (last N days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for PostgreSQL (German time stored as-is, no timezone conversion)
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Query Supabase
        # Note: Database stores German local time without timezone
        # CRITICAL: Add limit to prevent timeout on large datasets
        change_col = f"{fuel_type}change"
        
        result = supabase.table('prices')\
            .select(f'date, {fuel_type}, {change_col}')\
            .eq('station_uuid', station_uuid)\
            .gte('date', start_str)\
            .lte('date', end_str)\
            .order('date', desc=True)\
            .limit(2000)\
            .execute()
        
        # Convert to DataFrame
        if not result.data:
            # No data found - return empty DataFrame with correct schema
            return pd.DataFrame(columns=['date', 'price', 'price_change'])
        
        df = pd.DataFrame(result.data)
        
        # Rename columns to generic names
        df = df.rename(columns={
            fuel_type: 'price',
            change_col: 'price_change'
        })
        
        # Convert date strings to datetime (handle both formats)
        # CRITICAL: Explicitly set utc=False for Pandas 2.x compatibility
        # This prevents "unsupported operand type" errors with datetime arithmetic
        df['date'] = pd.to_datetime(df['date'], utc=False, errors='coerce')
        
        # Remove rows with invalid dates or NULL prices
        df = df[df['date'].notna() & df['price'].notna()].copy()
        
        # Ensure numeric types
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['price_change'] = pd.to_numeric(df['price_change'], errors='coerce').fillna(0).astype(int)
        
        # Sort by date ASCENDING (we queried DESC for efficiency, now reverse for display)
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    except Exception as exc:
        raise DataAccessError(
            user_message="Failed to retrieve price history from database.",
            remediation="Check your internet connection and database access.",
            details=f"Error querying Supabase for station {station_uuid}: {exc}"
        ) from exc


def calculate_hourly_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate price data by hour-of-day to find optimal refueling times.
    
    Parameters
    ----------
    df : pd.DataFrame
        Output from get_station_price_history() with columns ['date', 'price']
    
    Returns
    -------
    pd.DataFrame
        Columns: ['hour', 'avg_price', 'min_price', 'max_price', 'count']
        Indexed by hour (0-23)
    
    Examples
    --------
    >>> history_df = get_station_price_history("abc-123")
    >>> hourly_df = calculate_hourly_price_stats(history_df)
    >>> hourly_df.loc[11]  # 11:00 AM
    hour            11
    avg_price     1.629
    min_price     1.619
    max_price     1.639
    count            14
    """
    if df.empty:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=['hour', 'avg_price', 'min_price', 'max_price', 'count'])
    
    # Extract hour from datetime
    df = df.copy()
    df['hour'] = df['date'].dt.hour
    
    # Aggregate by hour
    hourly = df.groupby('hour')['price'].agg([
        ('avg_price', 'mean'),
        ('min_price', 'min'),
        ('max_price', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    # Ensure all hours 0-23 are present (fill missing with NaN)
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly = all_hours.merge(hourly, on='hour', how='left')
    
    return hourly


def get_opening_hours_display(openingtimes_json: str) -> str:
    """
    Parse Tankerkönig opening hours JSON into human-readable format.
    
    The database stores complex JSON with bitmask encoding:
    - applicable_days: Bitmask (1=Mon, 2=Tue, 4=Wed, ..., 64=Sun)
    - periods: List of {startp, endp} time ranges
    
    Examples of applicable_days bitmask:
    - 31  = 0011111 = Mon-Fri
    - 32  = 0100000 = Sat
    - 64  = 1000000 = Sun
    - 96  = 1100000 = Sat-Sun
    - 127 = 1111111 = Mon-Sun (every day)
    
    Parameters
    ----------
    openingtimes_json : str
        Raw JSON string from database
    
    Returns
    -------
    str
        Human-readable opening hours (e.g., "Mon-Fri: 06:00-22:00 | Sat-Sun: 07:00-20:00")
        Or "24/7" if always open
        Or "Hours not available" if data is missing/corrupt
    
    Examples
    --------
    >>> json_str = '{"openingTimes": [{"applicable_days": 31, "periods": [{"startp": "06:00", "endp": "22:00"}]}]}'
    >>> get_opening_hours_display(json_str)
    'Mon-Fri: 06:00-22:00'
    """
    if not openingtimes_json or openingtimes_json in ('{}', ''):
        return "Hours not available"
    
    try:
        # Day names (Mon=0, Tue=1, ..., Sun=6)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Parse JSON
        data = json.loads(openingtimes_json)
        
        # Extract openingTimes array
        opening_times = data.get("openingTimes", [])
        if not opening_times:
            return "Hours not available"
        
        # Check for 24/7 pattern (all days, 00:00-24:00)
        if len(opening_times) == 1:
            entry = opening_times[0]
            if entry.get("applicable_days") == 127:  # All 7 days
                periods = entry.get("periods", [])
                if periods and len(periods) == 1:
                    period = periods[0]
                    if period.get("startp") == "00:00" and period.get("endp") == "24:00":
                        return "24/7"
        
        # Parse each entry
        schedule_parts = []
        
        for entry in opening_times:
            applicable_days = entry.get("applicable_days", 0)
            periods = entry.get("periods", [])
            
            if not periods:
                continue
            
            # Decode bitmask to day names
            active_days = []
            for i, day_name in enumerate(day_names):
                if applicable_days & (1 << i):
                    active_days.append(day_name)
            
            if not active_days:
                continue
            
            # Format day range (e.g., Mon-Fri or Sat-Sun)
            if len(active_days) == 1:
                day_str = active_days[0]
            elif len(active_days) == 7:
                day_str = "Every day"
            elif active_days == day_names[:5]:  # Mon-Fri
                day_str = "Mon-Fri"
            elif active_days == day_names[5:]:  # Sat-Sun
                day_str = "Sat-Sun"
            else:
                # Non-consecutive days, list them
                day_str = ", ".join(active_days)
            
            # Format time periods (usually just one, but can be multiple)
            time_ranges = []
            for period in periods:
                start = period.get("startp", "")
                end = period.get("endp", "")
                if start and end:
                    # Clean up time format (remove :00 seconds if present)
                    start = start[:5] if len(start) > 5 else start
                    end = end[:5] if len(end) > 5 else end
                    time_ranges.append(f"{start}-{end}")
            
            if time_ranges:
                schedule_parts.append(f"{day_str}: {', '.join(time_ranges)}")
        
        if schedule_parts:
            return " | ".join(schedule_parts)
        else:
            return "Hours not available"
    
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        # If parsing fails, return fallback
        return f"Hours format error"




# ---------------------------------------------------------------------
# Station metadata (name/brand/address/opening times) from Supabase
# ---------------------------------------------------------------------
from functools import lru_cache

@lru_cache(maxsize=4096)
def get_station_metadata(station_uuid: str) -> dict:
    """Fetch station metadata from Supabase `stations` table for one uuid.

    Returns a dict with best-effort fields (missing values may be None/empty).
    Raises DataAccessError on connectivity/query errors.
    """
    if not station_uuid:
        raise DataQualityError(
            user_message="Station id is missing.",
            remediation="Select a valid station and try again."
        )

    client = _get_supabase_client()
    try:
        resp = (
            client.table("stations")
            .select(
                "uuid,name,brand,street,house_number,post_code,city,latitude,longitude,openingtimes_json"
            )
            .eq("uuid", station_uuid)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise DataAccessError(
            user_message="Could not load station metadata from the database.",
            remediation="Check Supabase connectivity and credentials (.env).",
            debug=e
        )

    data = getattr(resp, "data", None)
    if not data:
        # Not found is not fatal; return minimal info
        return {"uuid": station_uuid}

    row = data[0] if isinstance(data, list) else data
    # Ensure uuid is present
    row["uuid"] = row.get("uuid") or station_uuid
    return row
def get_cheapest_and_most_expensive_hours(hourly_df: pd.DataFrame) -> Dict[str, any]:
    """
    Identify the cheapest and most expensive hours to refuel.
    
    Parameters
    ----------
    hourly_df : pd.DataFrame
        Output from calculate_hourly_price_stats()
    
    Returns
    -------
    dict
        {
            'cheapest_hour': int (0-23),
            'cheapest_price': float,
            'most_expensive_hour': int (0-23),
            'most_expensive_price': float
        }
        Returns None values if data is insufficient
    
    Examples
    --------
    >>> hourly_df = calculate_hourly_price_stats(history_df)
    >>> optimal = get_cheapest_and_most_expensive_hours(hourly_df)
    >>> print(f"Best time: {optimal['cheapest_hour']:02d}:00 (€{optimal['cheapest_price']:.3f})")
    Best time: 11:00 (€1.619)
    """
    if hourly_df.empty or hourly_df['avg_price'].isna().all():
        return {
            'cheapest_hour': None,
            'cheapest_price': None,
            'most_expensive_hour': None,
            'most_expensive_price': None
        }
    
    # Find cheapest hour
    cheapest_idx = hourly_df['avg_price'].idxmin()
    cheapest_row = hourly_df.loc[cheapest_idx]
    
    # Find most expensive hour
    most_expensive_idx = hourly_df['avg_price'].idxmax()
    most_expensive_row = hourly_df.loc[most_expensive_idx]
    
    return {
        'cheapest_hour': int(cheapest_row['hour']),
        'cheapest_price': float(cheapest_row['avg_price']),
        'most_expensive_hour': int(most_expensive_row['hour']),
        'most_expensive_price': float(most_expensive_row['avg_price'])
    }


# =============================================================================
# TESTING / EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage showing how to fetch and analyze station price history.
    
    NOTE: Requires valid Supabase credentials in environment:
        export SUPABASE_URL="https://your-project.supabase.co"
        export SUPABASE_SECRET_KEY="your-service-role-key"
    """
    import sys
    
    print("=" * 70)
    print("HISTORICAL DATA MODULE - TEST RUN")
    print("=" * 70)
    
    # Test with a real station UUID (replace with your own)
    test_uuid = "51d4b6fd-a095-1aa0-e100-80009459e03a"
    test_fuel = "e5"
    
    print(f"\nTest Station UUID: {test_uuid}")
    print(f"Fuel Type: {test_fuel.upper()}")
    print(f"Days of history: 14")
    
    try:
        # Fetch price history
        print("\n[1] Fetching price history...")
        df_history = get_station_price_history(test_uuid, fuel_type=test_fuel, days=14)
        
        if df_history.empty:
            print("✗ No data found for this station.")
            sys.exit(1)
        
        print(f"✓ Retrieved {len(df_history)} price records")
        print(f"  Date range: {df_history['date'].min()} to {df_history['date'].max()}")
        print(f"  Price range: €{df_history['price'].min():.3f} - €{df_history['price'].max():.3f}")
        
        # Calculate hourly statistics
        print("\n[2] Calculating hourly statistics...")
        hourly_df = calculate_hourly_price_stats(df_history)
        print(f"✓ Calculated stats for {hourly_df['count'].notna().sum()} hours")
        
        # Find optimal times
        print("\n[3] Finding optimal refueling times...")
        optimal = get_cheapest_and_most_expensive_hours(hourly_df)
        
        if optimal['cheapest_hour'] is not None:
            print(f"✓ Best time to refuel:")
            print(f"  {optimal['cheapest_hour']:02d}:00 - Average price: €{optimal['cheapest_price']:.3f}/L")
            print(f"✗ Worst time to refuel:")
            print(f"  {optimal['most_expensive_hour']:02d}:00 - Average price: €{optimal['most_expensive_price']:.3f}/L")
            
            savings = optimal['most_expensive_price'] - optimal['cheapest_price']
            print(f"\n💰 Potential savings: €{savings:.3f}/L ({savings/optimal['cheapest_price']*100:.1f}%)")
        else:
            print("✗ Insufficient data for hourly analysis")
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
    
    except DataAccessError as exc:
        print(f"\n✗ Database Error: {exc.user_message}")
        print(f"  Remediation: {exc.remediation}")
        sys.exit(1)
    
    except DataQualityError as exc:
        print(f"\n✗ Data Quality Error: {exc.user_message}")
        print(f"  Remediation: {exc.remediation}")
        sys.exit(1)
    
    except Exception as exc:
        print(f"\n✗ Unexpected Error: {exc}")
        sys.exit(1)
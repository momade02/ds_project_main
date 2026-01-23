"""
Module: Historical Data Access Layer.

Description:
    This module provides access to historical fuel price data from Supabase.
    It is used exclusively by Page 03 (Station Details) for price charts and statistics.

    Active functions (used by Page 03):
    - get_station_price_history: Fetches 14-day price time series for charts
    - calculate_hourly_price_stats: Computes hourly averages for "best time to refuel"
    - get_cheapest_and_most_expensive_hours: Finds optimal/worst refueling hours

    Legacy functions (kept for reference/testing):
    - get_station_metadata: Station info lookup (Page 03 uses now route data instead)
    - get_opening_hours_display: Opening hours parser (Page 03 uses now Google/TK API directly)

    Data Validation:
    This module focuses on data retrieval for visualization.
    Validation responsibilities are distributed across the pipeline:
    
    - fuel_type: Validated here (must be e5, e10, or diesel)
    - Station UUIDs: Received from route integration pipeline, which derives them
      from the Supabase stations table; unknown UUIDs return empty results
    - Empty results: Return empty DataFrame; UI layer (Page 03) shows "No data" messages
    - Price plausibility: Validated downstream in recommender.py
    
    Price values from Tankerkoenig are trusted as-is since the data comes from
    an official source regulated by the Bundeskartellamt (German competition authority).

    Implementation details:
    - Dates are handled in local German time (as stored in DB) to avoid TZ confusion.
    - Caching (`lru_cache`) is applied to static station metadata.
"""

import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Final

import pandas as pd
from supabase import create_client, Client  # type: ignore

from src.app.app_errors import DataAccessError, DataQualityError

# ==========================================
# Type Definitions & Constants
# ==========================================

# Type aliases using string annotations for Python 3.9 compatibility
StationUUID = str
FuelType = str  # 'e5', 'e10', 'diesel'

# DataFrame Schema: ['date', 'price', 'price_change']
PriceHistoryDF = pd.DataFrame

# DataFrame Schema: ['hour', 'avg_price', 'min_price', 'max_price', 'count']
HourlyStatsDF = pd.DataFrame

VALID_FUEL_TYPES: Final[List[str]] = ["e5", "e10", "diesel"]
DEFAULT_HISTORY_DAYS: Final[int] = 14
DB_QUERY_LIMIT: Final[int] = 2000


# ==========================================
# Client Factory
# ==========================================

def _get_supabase_client() -> Client:
    """
    Initializes Supabase client from environment variables.
    
    Raises:
        DataAccessError: If credentials are missing.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not key:
        raise DataAccessError(
            user_message="Database connection is not configured.",
            remediation="Set SUPABASE_URL and SUPABASE_SECRET_KEY in environment.",
            details="Missing Supabase credentials.",
        )
    return create_client(url, key)


# ==========================================
# Core Data Retrieval
# ==========================================

def get_station_price_history(
    station_uuid: StationUUID,
    fuel_type: FuelType = "e5",
    days: int = DEFAULT_HISTORY_DAYS
) -> PriceHistoryDF:
    """
    Fetches raw time-series price data for a specific station.

    Args:
        station_uuid: Tankerkoenig UUID.
        fuel_type: 'e5', 'e10', or 'diesel'.
        days: Lookback window size.

    Returns:
        pd.DataFrame: Schema ['date', 'price', 'price_change'].
                      Sorted ascending by date.
    """
    # 1. Input Validation
    if fuel_type not in VALID_FUEL_TYPES:
        raise DataQualityError(
            user_message=f"Invalid fuel type '{fuel_type}'.",
            remediation=f"Choose one of: {', '.join(VALID_FUEL_TYPES)}",
            details=f"fuel_type must be in {VALID_FUEL_TYPES}",
        )

    try:
        client = _get_supabase_client()

        # 2. Date Range Calculation (Local Time)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Strings formatted for PostgreSQL timestamp comparison
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # 3. Dynamic Column Selection
        # We need the price column (e.g., 'e5') and the change column (e.g., 'e5change')
        change_col = f"{fuel_type}change"

        # 4. Database Query
        response = (
            client.table("prices")
            .select(f"date, {fuel_type}, {change_col}")
            .eq("station_uuid", station_uuid)
            .gte("date", start_str)
            .lte("date", end_str)
            .order("date", desc=True)  # DESC for efficiency (latest first)
            .limit(DB_QUERY_LIMIT)
            .execute()
        )

        # 5. Schema Enforcement (Handle Empty Results)
        if not response.data:
            return pd.DataFrame(columns=["date", "price", "price_change"])

        # 6. Transformation
        df = pd.DataFrame(response.data)
        
        # Rename to standardized schema
        df = df.rename(columns={
            fuel_type: "price",
            change_col: "price_change"
        })

        # Date Parsing
        # utc=False is important here. The DB stores '2023-01-01 12:00:00' (German time).
        # We want to keep it as a naive timestamp representing that wall-clock time.
        df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")

        # Numeric Coercion & Cleanup
        df = df[df["date"].notna()].copy()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["price_change"] = pd.to_numeric(df["price_change"], errors="coerce").fillna(0).astype(int)
        
        # Final cleanup: drop NaN prices and sort chronological
        df = df.dropna(subset=["price"])
        df = df.sort_values("date", ascending=True).reset_index(drop=True)

        return df

    except DataAccessError:
        raise
    except DataQualityError:
        raise
    except Exception as e:
        raise DataAccessError(
            user_message="Failed to retrieve price history.",
            remediation="Check internet connection or try again later.",
            details=str(e),
        ) from e


def calculate_hourly_price_stats(df: PriceHistoryDF) -> HourlyStatsDF:
    """
    Aggregates a price history DataFrame by Hour of Day (0-23).
    
    Uses forward-fill to show the effective price at each hour,
    not just hours when prices changed. This solves the "sparse data" problem
    where stations that rarely change prices would show mostly empty charts.

    Returns:
        pd.DataFrame: Indexed by 0-23, containing avg/min/max/count.
    """
    expected_cols = ["hour", "avg_price", "min_price", "max_price", "count"]
    
    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    # Work on copy to avoid side-effects
    work_df = df.copy()
    
    # Sort by date to ensure forward-fill works correctly
    work_df = work_df.sort_values("date").reset_index(drop=True)
    
    # Create a complete hourly time series and forward-fill prices
    # This gives us the effective price at each hour, not just when prices changed
    if len(work_df) >= 2:
        try:
            # Get the date range
            min_date = work_df["date"].min()
            max_date = work_df["date"].max()
            
            # Create hourly index covering the full period
            hourly_index = pd.date_range(
                start=min_date.floor('H'),
                end=max_date.ceil('H'),
                freq='H'
            )
            
            # Set date as index for resampling
            work_df = work_df.set_index("date")
            
            # Resample to hourly and forward-fill
            # This means: at each hour, what was the last known price?
            hourly_prices = work_df["price"].resample('H').last().ffill()
            
            # Convert back to DataFrame
            work_df = pd.DataFrame({
                "date": hourly_prices.index,
                "price": hourly_prices.values
            })
            
        except Exception as e:
            # forward-fill failed, fall back to original method
            print(f"Warning: hourly resampling failed ({e}), using raw data instead")
            work_df = df.copy()
    
    # Extract hour of day
    work_df["hour"] = work_df["date"].dt.hour

    # Aggregation
    hourly = work_df.groupby("hour")["price"].agg([
        ("avg_price", "mean"),
        ("min_price", "min"),
        ("max_price", "max"),
        ("count", "count"),
    ]).reset_index()

    # Reindexing: Ensure 0-23 always exist (fill missing with NaN)
    all_hours = pd.DataFrame({"hour": range(24)})
    merged = all_hours.merge(hourly, on="hour", how="left")

    return merged


def get_cheapest_and_most_expensive_hours(hourly_df: HourlyStatsDF) -> Dict[str, Any]:
    """
    Analyzes hourly stats to find extrema.

    Returns:
        Dict with keys: 'cheapest_hour', 'cheapest_price', 'most_expensive_hour', ...
    """
    if hourly_df.empty or hourly_df["avg_price"].isna().all():
        return {
            "cheapest_hour": None,
            "cheapest_price": None,
            "most_expensive_hour": None,
            "most_expensive_price": None,
        }

    # Find indices of min and max average price
    idx_min = hourly_df["avg_price"].idxmin()
    idx_max = hourly_df["avg_price"].idxmax()

    row_min = hourly_df.loc[idx_min]
    row_max = hourly_df.loc[idx_max]

    return {
        "cheapest_hour": int(row_min["hour"]),
        "cheapest_price": float(row_min["avg_price"]),
        "most_expensive_hour": int(row_max["hour"]),
        "most_expensive_price": float(row_max["avg_price"]),
    }


# ==========================================
# Legacy Functions (kept for reference/testing)
# ==========================================

@lru_cache(maxsize=4096)
def get_station_metadata(station_uuid: StationUUID) -> Dict[str, Any]:
    """
    Fetches static station details (Name, Address, Brand).
    Cached in memory to reduce DB calls for frequently accessed stations.
    
    Note: This function is not used by Page 03. Station metadata is now
    passed through from the route integration pipeline instead of being
    fetched separately. Kept for the test harness and potential future use.
    """
    if not station_uuid:
        raise DataQualityError(
            user_message="Station UUID is missing.",
            remediation="Provide a valid station UUID.",
            details="station_uuid was empty or None",
        )

    client = _get_supabase_client()
    try:
        response = (
            client.table("stations")
            .select("uuid,name,brand,street,house_number,post_code,city,latitude,longitude,openingtimes_json")
            .eq("uuid", station_uuid)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise DataAccessError(
            user_message="Failed to fetch station metadata.",
            remediation="Check database connection and try again.",
            details=str(e),
        ) from e

    data = response.data
    if not data:
        # Fallback for valid ID but missing record
        return {"uuid": station_uuid}

    row = data[0]
    # Ensure UUID is explicitly set in result even if DB returns weirdness
    row["uuid"] = row.get("uuid") or station_uuid
    return row


def get_opening_hours_display(openingtimes_json: str) -> str:
    """
    Parses Tankerkoenig JSON bitmasks into readable strings.
    
    Note: This function is not used anymore by Page 03. Opening hours are now
    fetched directly from Google Places API or Tankerkoenig detail.php
    in the UI layer. Kept for the test harness and potential future use.

    Logic:
        The 'applicable_days' field is a 7-bit mask:
        1=Mon, 2=Tue, 4=Wed, 8=Thu, 16=Fri, 32=Sat, 64=Sun.
        Examples: 31 (binary 0011111) = Mon-Fri.
    """
    if not openingtimes_json or openingtimes_json in ("{}", ""):
        return "Hours not available"

    try:
        data = json.loads(openingtimes_json)
        opening_times = data.get("openingTimes", [])
        
        if not opening_times:
            return "Hours not available"

        # Check for 24/7 shortcut (mask 127 = 1111111 = All Days)
        if len(opening_times) == 1:
            entry = opening_times[0]
            if entry.get("applicable_days") == 127:
                periods = entry.get("periods", [])
                if len(periods) == 1:
                    p = periods[0]
                    if p.get("startp") == "00:00" and p.get("endp") == "24:00":
                        return "24/7"

        # Standard Parsing
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        schedule_parts = []

        for entry in opening_times:
            mask = entry.get("applicable_days", 0)
            periods = entry.get("periods", [])

            if not periods:
                continue

            # Bitwise decode of days
            active_days = []
            for i, name in enumerate(day_names):
                # Check if i-th bit is set (e.g. i=0 is 1<<0=1, i=1 is 1<<1=2)
                if mask & (1 << i):
                    active_days.append(name)

            if not active_days:
                continue

            # Text Formatting for Days
            if len(active_days) == 7:
                day_str = "Every day"
            elif active_days == day_names[:5]:
                day_str = "Mon-Fri"
            elif active_days == day_names[5:]:
                day_str = "Sat-Sun"
            elif len(active_days) == 1:
                day_str = active_days[0]
            else:
                day_str = ", ".join(active_days)

            # Text Formatting for Times
            time_strs = []
            for p in periods:
                s = p.get("startp", "")[:5]  # Truncate seconds '06:00:00' -> '06:00'
                e = p.get("endp", "")[:5]
                if s and e:
                    time_strs.append(f"{s}-{e}")

            if time_strs:
                schedule_parts.append(f"{day_str}: {', '.join(time_strs)}")

        return " | ".join(schedule_parts) if schedule_parts else "Hours not available"

    except (json.JSONDecodeError, KeyError, TypeError):
        return "Hours format error"


# ==========================================
# Integration Test
# ==========================================

if __name__ == "__main__":
    """
    Direct Execution Test.
    Run: python src/app/integration/historical_data.py
    """
    import sys
    print("=== Historical Data Module Test ===")
    
    # Use a known test ID or random one
    TEST_UUID = "51d4b6fd-a095-1aa0-e100-80009459e03a"
    
    try:
        print(f"1. Fetching Metadata for {TEST_UUID}...")
        meta = get_station_metadata(TEST_UUID)
        print(f"   Name: {meta.get('name', 'Unknown')}")
        print(f"   Opening Hours: {get_opening_hours_display(meta.get('openingtimes_json', ''))}")

        print("\n2. Fetching E5 Price History (14 days)...")
        hist = get_station_price_history(TEST_UUID, "e5", 14)
        print(f"   Retrieved {len(hist)} records.")
        if not hist.empty:
            print(f"   Latest: {hist.iloc[-1]['date']} -> {hist.iloc[-1]['price']}")

        print("\n3. Calculating Stats...")
        stats = calculate_hourly_price_stats(hist)
        best = get_cheapest_and_most_expensive_hours(stats)
        
        if best["cheapest_hour"] is not None:
            print(f"   Best Time: {best['cheapest_hour']:02d}:00 "
                  f"(@ {best['cheapest_price']:.3f})")
            print(f"   Worst Time: {best['most_expensive_hour']:02d}:00 "
                  f"(@ {best['most_expensive_price']:.3f})")
        else:
            print("   Insufficient data for stats.")
            
        print("\n[ok] Test complete.")

    except Exception as e:
        print(f"\n[error] Test Failed: {e}")
        sys.exit(1)
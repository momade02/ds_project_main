"""
Module: Daily Tankerkönig Data ETL Pipeline.

Description:
    This script manages the daily synchronization of fuel station data and prices 
    from the Tankerkönig Open Data repository to a Supabase PostgreSQL database.

    ETL Stages:
    1. EXTRACT: Fetch raw CSV dumps (Stations & Prices) for the previous day.
       - Source: https://dev.tankerkoenig.de/ (Authenticated URL)
       - Frequency: Once daily (usually via cron).
    
    2. TRANSFORM: 
       - Stations: Standardization of coordinates and timestamps.
       - Prices: Timezone normalization (stripping offsets for local time storage) 
         and numeric coercion.

    3. LOAD:
       - Stations: "Wipe and Replace" strategy (Full Refresh).
       - Prices: "Rolling Window" strategy (Insert New + Delete Old > 14 days).

Usage:
    Run directly as a script. Requires a valid .env configuration.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Final, Optional, Any, List, Dict, TypeAlias

import pandas as pd
import requests
from dotenv import load_dotenv

# --- Type Definitions ---
SupabaseClient: TypeAlias = Any  # dynamic import
DataFrame: TypeAlias = pd.DataFrame
RecordsList: TypeAlias = List[Dict[str, Any]]

# --- Configuration & Constants ---
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Paths & Logging
LOG_DIR: Final[Path] = Path.home() / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE: Final[Path] = LOG_DIR / f"daily_update_{datetime.now().strftime('%Y%m%d')}.log"

# ETL Parameters
BATCH_SIZE_STATIONS: Final[int] = 500
BATCH_SIZE_PRICES: Final[int] = 1000
PRICE_RETENTION_DAYS: Final[int] = 14
DATA_DELAY_DAYS: Final[int] = 1  # Tankerkönig data is yesterday's dump

# Global State
_SUPABASE: Optional[SupabaseClient] = None
_BASE_URL_TEMPLATE: Optional[str] = None


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ==========================================
# Helpers: Transformation Logic
# ==========================================

def _clean_station_timestamp(val: Any) -> Optional[str]:
    """
    Parses 'first_active' timestamps for Stations.
    Format: UTC input -> 'YYYY-MM-DD HH:MM:SS' output.
    """
    if pd.isna(val) or val == "" or val is None:
        return None
    try:
        # Convert to UTC-aware datetime, then format as string
        dt = pd.to_datetime(val, utc=True)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _clean_price_timestamp(val: Any) -> Optional[str]:
    """
    Parses 'date' timestamps for Prices.
    Format: '2025-11-14 00:00:04+01' -> '2025-11-14 00:00:04' (Naive Local Time).
    
    Why? The DB stores local German time without timezone context.
    """
    if pd.isna(val) or val == "" or val is None:
        return None
    try:
        # Parse and strip timezone info to keep the "wall clock" time
        dt = pd.to_datetime(val)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _require_initialized() -> None:
    """Ensures global clients are ready before execution."""
    if _SUPABASE is None:
        raise RuntimeError("Supabase client not initialized.")
    if not _BASE_URL_TEMPLATE:
        raise RuntimeError("API URL template not configured.")


# ==========================================
# Phase 1: EXTRACT
# ==========================================

def download_csv(url: str) -> DataFrame:
    """
    Fetches a CSV file from the Tankerkönig raw data repository.
    
    Args:
        url: The full authenticated URL.
        
    Returns:
        pd.DataFrame: The raw dataset.
    """
    try:
        # Log URL safely (hide credentials)
        safe_url = url.split("@")[1] if "@" in url else url
        logger.info(f"Downloading: {safe_url}")

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text))
        logger.info(f"✓ Downloaded {len(df):,} rows")
        return df

    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        raise


# ==========================================
# Phase 2: STATIONS (Wipe & Replace)
# ==========================================

def update_stations() -> bool:
    """
    Full refresh of the 'stations' table.
    
    Steps:
    1. Fetch yesterday's station dump.
    2. Convert coordinates to numeric, format timestamps.
    3. DELETE all rows in DB (except system placeholders).
    4. INSERT new data in batches.
    """
    _require_initialized()
    logger.info("\n=== UPDATING STATIONS ===")

    try:
        # 1. URL Construction
        target_date = datetime.now() - timedelta(days=DATA_DELAY_DAYS)
        url = (
            f"{_BASE_URL_TEMPLATE}/stations/{target_date.year}/"
            f"{target_date.month:02d}/{target_date.strftime('%Y-%m-%d')}-stations.csv"
        )

        # 2. Extract
        df = download_csv(url)
        if df.empty:
            logger.warning("Stations CSV is empty. Aborting update.")
            return False

        # 3. Transform
        logger.info("Transforming station data...")
        
        # Numeric Coercion
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # Timestamp Cleanup
        if "first_active" in df.columns:
            df["first_active"] = df["first_active"].apply(_clean_station_timestamp)

        # Handle NaNs (Supabase JSON serialization hates NaNs)
        df = df.where(pd.notnull(df), None)
        records = df.to_dict("records")

        # 4. Load (Wipe)
        logger.info("Deleting existing stations...")
        _SUPABASE.table("stations").delete().neq(
            "uuid", "00000000-0000-0000-0000-000000000000"
        ).execute()

        # 5. Load (Insert)
        total_batches = (len(records) + BATCH_SIZE_STATIONS - 1) // BATCH_SIZE_STATIONS
        logger.info(f"Inserting {len(records):,} records in {total_batches} batches...")

        count = 0
        for i in range(0, len(records), BATCH_SIZE_STATIONS):
            batch = records[i : i + BATCH_SIZE_STATIONS]
            _SUPABASE.table("stations").insert(batch).execute()
            count += len(batch)
            
            if (i // BATCH_SIZE_STATIONS + 1) % 5 == 0:
                logger.info(f"  Progress: {count:,} records...")

        logger.info(f"✓ Stations Updated: {count:,} records.")
        return True

    except Exception as e:
        logger.error(f"✗ Stations update failed: {e}", exc_info=True)
        return False


# ==========================================
# Phase 3: PRICES (Rolling Window)
# ==========================================

def update_prices() -> bool:
    """
    Rolling update of the 'prices' table.
    
    Steps:
    1. Fetch yesterday's price dump.
    2. Strip timezones, normalize numerics.
    3. DELETE records older than 14 days.
    4. INSERT new records.
    """
    _require_initialized()
    logger.info("\n=== UPDATING PRICES ===")

    try:
        # 1. URL Construction
        target_date = datetime.now() - timedelta(days=DATA_DELAY_DAYS)
        url = (
            f"{_BASE_URL_TEMPLATE}/prices/{target_date.year}/"
            f"{target_date.month:02d}/{target_date.strftime('%Y-%m-%d')}-prices.csv"
        )

        # 2. Extract
        df = download_csv(url)
        if df.empty:
            logger.warning("Prices CSV is empty. Aborting update.")
            return False

        # 3. Transform
        logger.info("Transforming price data...")
        
        # Timestamp normalization
        df["date"] = df["date"].apply(_clean_price_timestamp)
        
        # Numeric Coercion
        price_cols = ["diesel", "e5", "e10"]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        change_cols = ["dieselchange", "e5change", "e10change"]
        for col in change_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Handle NaNs
        df = df.where(pd.notnull(df), None)
        records = df.to_dict("records")

        # 4. Load (Retention Policy)
        cutoff_dt = datetime.now() - timedelta(days=PRICE_RETENTION_DAYS)
        cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Pruning prices older than {cutoff_str}...")
        try:
            _SUPABASE.table("prices").delete().lt("date", cutoff_str).execute()
        except Exception as e:
            logger.warning(f"Pruning warning (non-fatal): {e}")

        # 5. Load (Insert)
        total_batches = (len(records) + BATCH_SIZE_PRICES - 1) // BATCH_SIZE_PRICES
        logger.info(f"Inserting {len(records):,} records in {total_batches} batches...")

        count = 0
        failed_batches = 0
        
        for i in range(0, len(records), BATCH_SIZE_PRICES):
            batch = records[i : i + BATCH_SIZE_PRICES]
            try:
                _SUPABASE.table("prices").insert(batch).execute()
                count += len(batch)
                
                # Log periodic progress
                current_batch = i // BATCH_SIZE_PRICES + 1
                if current_batch % 100 == 0:
                     logger.info(f"  Progress: {count:,} records...")
                     
            except Exception as e:
                logger.warning(f"  Batch failed: {e}")
                failed_batches += 1

        success_rate = (count / len(records) * 100) if len(records) > 0 else 0
        logger.info(f"✓ Prices Updated: {count:,} ({success_rate:.1f}% success).")
        
        return success_rate >= 95.0

    except Exception as e:
        logger.error(f"✗ Prices update failed: {e}", exc_info=True)
        return False


# ==========================================
# Main Orchestration
# ==========================================

def main() -> None:
    """Entrypoint: Configures environment and orchestrates ETL phases."""
    global _SUPABASE, _BASE_URL_TEMPLATE

    # 1. Environment
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    
    tk_email = os.getenv("TANKERKOENIG_EMAIL")
    tk_key = os.getenv("TANKERKOENIG_API_KEY")
    sb_url = os.getenv("SUPABASE_URL")
    sb_key = os.getenv("SUPABASE_SECRET_KEY")

    if not all([tk_email, tk_key, sb_url, sb_key]):
        logger.error("✗ Missing Environment Variables (Check .env).")
        sys.exit(1)

    _BASE_URL_TEMPLATE = (
        f"https://{tk_email}:{tk_key}"
        "@data.tankerkoenig.de/tankerkoenig-organization/tankerkoenig-data/raw/branch/master"
    )

    # 2. Initialization
    try:
        from supabase import create_client  # type: ignore
        _SUPABASE = create_client(sb_url, sb_key)
        logger.info("✓ Supabase Connected.")
    except Exception as e:
        logger.error(f"✗ Supabase Init Failed: {e}")
        sys.exit(1)

    # 3. Execution
    logger.info("=" * 40)
    logger.info(f"ETL START: {datetime.now()}")
    logger.info("=" * 40)
    
    start_ts = datetime.now()
    
    ok_stations = update_stations()
    ok_prices = update_prices()
    
    duration = (datetime.now() - start_ts).total_seconds()
    
    logger.info("=" * 40)
    logger.info(f"ETL COMPLETE in {duration:.1f}s")
    logger.info(f"Stations: {'SUCCESS' if ok_stations else 'FAILED'}")
    logger.info(f"Prices:   {'SUCCESS' if ok_prices else 'FAILED'}")
    logger.info("=" * 40)

    if not (ok_stations and ok_prices):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal Error: {e}", exc_info=True)
        sys.exit(1)
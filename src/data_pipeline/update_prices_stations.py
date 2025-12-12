"""
Unified daily update script for Tankerkönig data.
Downloads latest stations and prices CSVs and uploads to Supabase.
Automatically deletes price data older than 14 days.

IMPORTANT ARCHITECTURE NOTE:
- This module does NOT load environment variables at import time.
- .env loading and Supabase initialization happen in main() (entrypoint),
  which keeps imports side-effect free and deployment-friendly.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO
import logging

import pandas as pd
import requests
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
log_dir = Path.home() / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"daily_update_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from Supabase/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Globals initialized in main() (entrypoint)
supabase = None
TANKERKOENIG_EMAIL = None
TANKERKOENIG_API_KEY = None
BASE_URL = None


def _require_initialized() -> None:
    """Ensure entrypoint initialization has run before using globals."""
    if supabase is None:
        raise RuntimeError("Supabase client is not initialized. Call main() as entrypoint.")
    if not BASE_URL:
        raise RuntimeError("BASE_URL is not initialized. Call main() as entrypoint.")


def download_csv(url: str) -> pd.DataFrame:
    """
    Download CSV from Tankerkönig data repository.

    NOTE: This makes exactly ONE request per call - no spam!
    """
    try:
        # Hide credentials in log
        safe_url = url.split("@")[1] if "@" in url else url
        logger.info(f"Downloading: {safe_url}")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        logger.info(f"✓ Downloaded {len(df):,} rows")
        return df

    except requests.exceptions.HTTPError as e:
        logger.error(f"✗ HTTP Error: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        raise


def update_stations() -> bool:
    """
    Update stations table with latest data.

    DELETION LOGIC: Deletes ALL existing stations, then inserts fresh data.
    This is a full refresh approach since station data changes infrequently.
    """
    _require_initialized()
    logger.info("\n=== UPDATING STATIONS ===")

    try:
        # Use yesterday's date (files published with 1-day delay)
        yesterday = datetime.now() - timedelta(days=1)
        year = yesterday.year
        month = f"{yesterday.month:02d}"
        date_str = yesterday.strftime("%Y-%m-%d")

        # Construct URL - ONE request to Tankerkönig
        url = f"{BASE_URL}/stations/{year}/{month}/{date_str}-stations.csv"

        # Download data (single request)
        df = download_csv(url)

        # Clean and prepare data
        logger.info("Preparing stations data...")

        # Convert data types
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # Handle first_active - it comes as string with timezone
        if "first_active" in df.columns:
            logger.info("Converting first_active timestamps...")

            def clean_timestamp(val):
                """Convert timestamp string to format Supabase accepts."""
                if pd.isna(val) or val == "" or val is None:
                    return None
                try:
                    dt = pd.to_datetime(val, utc=True)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return None

            df["first_active"] = df["first_active"].apply(clean_timestamp)
            logger.info("✓ Converted first_active timestamps")

        # Handle NaN values - replace with None for Supabase
        df = df.where(pd.notnull(df), None)

        # Convert to dict for Supabase
        data = df.to_dict("records")

        # === DELETION STEP ===
        logger.info("Deleting all existing stations...")
        supabase.table("stations").delete().neq("uuid", "00000000-0000-0000-0000-000000000000").execute()
        logger.info("✓ Old stations deleted")

        # Insert new data in batches
        batch_size = 500
        total_batches = (len(data) + batch_size - 1) // batch_size

        logger.info(f"Inserting {len(data):,} stations in {total_batches} batches...")

        inserted_count = 0
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            supabase.table("stations").insert(batch).execute()
            inserted_count += len(batch)

            # Only log every 5 batches to reduce spam
            batch_num = i // batch_size + 1
            if batch_num % 5 == 0 or batch_num == total_batches:
                logger.info(f"  Progress: {batch_num}/{total_batches} batches ({inserted_count:,} records)")

        logger.info(f"✓ Stations updated successfully: {inserted_count:,} total records")
        return True

    except Exception as e:
        logger.error(f"✗ Stations update failed: {e}", exc_info=True)
        return False


def update_prices() -> bool:
    """
    Update prices table with latest data and delete old records.

    DELETION LOGIC:
    - Deletes price records older than 14 days BEFORE inserting new data
    - Keeps a rolling 14-day window of historical prices
    - This happens EVERY TIME the script runs

    TIMEZONE HANDLING:
    - CSV contains German local time with timezone marker (e.g., '2025-11-14 00:00:04+01')
    - Strips timezone marker but keeps German local time (e.g., '2025-11-14 00:00:04')
    - NO conversion to UTC - preserves the original German time
    """
    _require_initialized()
    logger.info("\n=== UPDATING PRICES ===")

    try:
        # Use yesterday's date (files published with 1-day delay)
        yesterday = datetime.now() - timedelta(days=1)
        year = yesterday.year
        month = f"{yesterday.month:02d}"
        date_str = yesterday.strftime("%Y-%m-%d")

        # Construct URL - ONE request to Tankerkönig
        url = f"{BASE_URL}/prices/{year}/{month}/{date_str}-prices.csv"

        # Download data (single request)
        df = download_csv(url)

        # Clean and prepare data
        logger.info("Preparing prices data...")

        def strip_timezone(date_str_val):
            """Remove timezone info, keep local German time."""
            if pd.isna(date_str_val) or date_str_val == "" or date_str_val is None:
                return None
            try:
                dt = pd.to_datetime(date_str_val)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return None

        df["date"] = df["date"].apply(strip_timezone)
        logger.info(f"✓ Processed {df['date'].notna().sum():,} timestamps (kept German local time)")

        # Convert numeric columns
        for col in ["diesel", "e5", "e10"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["dieselchange", "e5change", "e10change"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # Handle NaN values - replace with None for Supabase
        df = df.where(pd.notnull(df), None)

        # === DELETION STEP ===
        cutoff_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Deleting prices older than {cutoff_date} (German time)...")

        try:
            supabase.table("prices").delete().lt("date", cutoff_date).execute()
            logger.info("✓ Old price records deleted (older than 14 days)")
        except Exception as e:
            logger.warning(f"Could not delete old records: {e}")

        # Insert new data in batches
        data = df.to_dict("records")
        batch_size = 1000
        total_batches = (len(data) + batch_size - 1) // batch_size

        logger.info(f"Inserting {len(data):,} price records in {total_batches} batches...")

        inserted_count = 0
        failed_batches = []

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            try:
                supabase.table("prices").insert(batch).execute()
                inserted_count += len(batch)

                # Log progress every 100 batches to reduce spam
                batch_num = i // batch_size + 1
                if batch_num % 100 == 0 or batch_num == total_batches:
                    logger.info(f"  Progress: {batch_num}/{total_batches} batches ({inserted_count:,} records)")

            except Exception as e:
                logger.warning(f"  Batch {i // batch_size + 1} failed: {e}")
                failed_batches.append(i // batch_size + 1)

        if failed_batches:
            logger.warning(f"Failed batches: {failed_batches[:10]}{'...' if len(failed_batches) > 10 else ''}")

        success_rate = (inserted_count / len(data) * 100) if len(data) > 0 else 0
        logger.info(f"✓ Prices updated: {inserted_count:,}/{len(data):,} records ({success_rate:.1f}% success)")

        # Return success if at least 95% of records were inserted
        return success_rate >= 95.0

    except Exception as e:
        logger.error(f"✗ Prices update failed: {e}", exc_info=True)
        return False


def main() -> None:
    """
    Run daily update for both stations and prices.

    SUMMARY OF WHAT THIS SCRIPT DOES:
    1. Loads local .env (entrypoint only)
    2. Initializes Supabase client
    3. Downloads yesterday's stations CSV from Tankerkönig (1 request)
    4. Deletes ALL stations in Supabase
    5. Inserts fresh stations data
    6. Downloads yesterday's prices CSV from Tankerkönig (1 request)
    7. Deletes prices older than 14 days from Supabase
    8. Inserts new prices data (keeping German local time)

    Total Tankerkönig requests: 2 per day (stations + prices)
    """
    global supabase, TANKERKOENIG_EMAIL, TANKERKOENIG_API_KEY, BASE_URL

    # Entry-point env loading (local dev convenience)
    env_path = project_root / ".env"
    load_dotenv(env_path, override=False)

    # Tankerkönig credentials
    TANKERKOENIG_EMAIL = os.getenv("TANKERKOENIG_EMAIL")
    TANKERKOENIG_API_KEY = os.getenv("TANKERKOENIG_API_KEY")

    if not TANKERKOENIG_EMAIL or not TANKERKOENIG_API_KEY:
        logger.error("✗ Tankerkönig credentials not found in environment variables")
        sys.exit(1)

    # Base URL pattern
    BASE_URL = (
        f"https://{TANKERKOENIG_EMAIL}:{TANKERKOENIG_API_KEY}"
        "@data.tankerkoenig.de/tankerkoenig-organization/tankerkoenig-data/raw/branch/master"
    )

    # Initialize Supabase (after env is loaded)
    try:
        from supabase import create_client

        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SECRET_KEY"),
        )
        logger.info("✓ Supabase client initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Supabase: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("TANKERKÖNIG DAILY UPDATE STARTED")
    logger.info("=" * 60)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Tankerkönig requests: 2 (stations + prices)")
    logger.info("=" * 60)

    start_time = datetime.now()

    # Update stations
    stations_success = update_stations()

    # Update prices
    prices_success = update_prices()

    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info("UPDATE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Stations: {'✓ SUCCESS' if stations_success else '✗ FAILED'}")
    logger.info(f"Prices:   {'✓ SUCCESS' if prices_success else '✗ FAILED'}")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("Total Tankerkönig API calls: 2")
    logger.info("=" * 60)

    # Exit with error code if any update failed
    if not (stations_success and prices_success):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n✗ Update interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)

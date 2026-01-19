"""
Unified daily update script for Tankerkönig data.
Downloads latest stations and prices CSVs and uploads to Supabase.
Automatically deletes price data older than 14 days.
Generates synthetic gap data for 00:00-07:00 coverage.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests
from io import StringIO
import logging
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
env_path = project_root / ".env"
load_dotenv(env_path)

# Configure logging
log_dir = Path.home() / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"daily_update_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from Supabase/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Import Supabase (after env is loaded)
try:
    from supabase import create_client
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SECRET_KEY")
    )
    logger.info("✓ Supabase client initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize Supabase: {e}")
    sys.exit(1)

# Tankerkönig credentials
TANKERKOENIG_EMAIL = os.getenv("TANKERKOENIG_EMAIL")
TANKERKOENIG_API_KEY = os.getenv("TANKERKOENIG_API_KEY")

if not TANKERKOENIG_EMAIL or not TANKERKOENIG_API_KEY:
    logger.error("✗ Tankerkönig credentials not found in environment variables")
    sys.exit(1)

# Base URL pattern
BASE_URL = f"https://{TANKERKOENIG_EMAIL}:{TANKERKOENIG_API_KEY}@data.tankerkoenig.de/tankerkoenig-organization/tankerkoenig-data/raw/branch/master"


def download_csv(url):
    """
    Download CSV from Tankerkönig data repository.
    
    NOTE: This makes exactly ONE request per call - no spam!
    """
    try:
        # Hide credentials in log
        safe_url = url.split('@')[1] if '@' in url else url
        logger.info(f"Downloading: {safe_url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse CSV
        df = pd.read_csv(StringIO(response.text))
        logger.info(f"✓ Downloaded {len(df):,} rows")
        return df
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"✗ HTTP Error: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        raise


def update_stations():
    """
    Update stations table with latest data.
    
    DELETION LOGIC: Deletes ALL existing stations, then inserts fresh data.
    This is a full refresh approach since station data changes infrequently.
    """
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
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Handle first_active - it comes as string with timezone
        if 'first_active' in df.columns:
            logger.info("Converting first_active timestamps...")
            
            def clean_timestamp(val):
                """Convert timestamp string to format Supabase accepts."""
                if pd.isna(val) or val == '' or val is None:
                    return None
                try:
                    # Parse the string (handles timezone automatically)
                    dt = pd.to_datetime(val, utc=True)
                    # Return as string without timezone
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    return None
            
            df['first_active'] = df['first_active'].apply(clean_timestamp)
            logger.info(f"✓ Converted first_active timestamps")
        
        # Handle NaN values - replace with None for Supabase
        df = df.where(pd.notnull(df), None)
        
        # Convert to dict for Supabase
        data = df.to_dict('records')
        
        # === DELETION STEP ===
        # Delete ALL existing stations (full table refresh)
        logger.info("Deleting all existing stations...")
        supabase.table('stations').delete().neq('uuid', '00000000-0000-0000-0000-000000000000').execute()
        logger.info("✓ Old stations deleted")
        
        # Insert new data in batches
        batch_size = 500
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        logger.info(f"Inserting {len(data):,} stations in {total_batches} batches...")
        
        inserted_count = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            supabase.table('stations').insert(batch).execute()
            inserted_count += len(batch)
            
            # Only log every 5 batches to reduce spam
            batch_num = i//batch_size + 1
            if batch_num % 5 == 0 or batch_num == total_batches:
                logger.info(f"  Progress: {batch_num}/{total_batches} batches ({inserted_count:,} records)")
        
        logger.info(f"✓ Stations updated successfully: {inserted_count:,} total records")
        return True
        
    except Exception as e:
        logger.error(f"✗ Stations update failed: {e}", exc_info=True)
        return False


def update_prices():
    """
    Update prices table with latest data and delete old records.
    
    DELETION LOGIC: 
    - Deletes price records older than 14 days BEFORE inserting new data
    - Deletes today's synthetic gap data (cleanup from yesterday)
    - Keeps a rolling 14-day window of historical prices
    
    INDEX REQUIREMENT (CRITICAL):
    - Requires index on 'date' column for fast deletion
    - CREATE INDEX idx_prices_date ON prices(date);
    
    TIMEZONE HANDLING:
    - Keeps German local time (no UTC conversion)
    """
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
        
        # Strip timezone, keep German local time
        def strip_timezone(date_str):
            """Remove timezone info, keep local German time."""
            if pd.isna(date_str) or date_str == '' or date_str is None:
                return None
            try:
                dt = pd.to_datetime(date_str)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return None
        
        df['date'] = df['date'].apply(strip_timezone)
        logger.info(f"✓ Processed {df['date'].notna().sum():,} timestamps")
        
        # Convert numeric columns
        for col in ['diesel', 'e5', 'e10']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for col in ['dieselchange', 'e5change', 'e10change']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Handle NaN values
        df = df.where(pd.notnull(df), None)
        
        # === DELETION STEP 1: Old data (>14 days) - BATCHED BY HOUR ===
        cutoff_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        logger.info(f"Deleting prices older than {cutoff_date} (batched by hour)...")
        
        deletion_success = False
        total_deleted_hours = 0
        try:
            # Find oldest date in database
            oldest_result = supabase.table('prices').select('date').order('date', desc=False).limit(1).execute()
            
            if oldest_result.data:
                oldest_date_str = oldest_result.data[0]['date'][:10]  # Extract YYYY-MM-DD
                oldest_date = datetime.strptime(oldest_date_str, '%Y-%m-%d')
                cutoff_dt = datetime.strptime(cutoff_date, '%Y-%m-%d')
                
                # Delete hour by hour (oldest first)
                current_dt = oldest_date
                while current_dt < cutoff_dt:
                    hour_start = current_dt.strftime('%Y-%m-%d %H:00:00')
                    hour_end = current_dt.strftime('%Y-%m-%d %H:59:59')
                    
                    try:
                        supabase.table('prices').delete().gte('date', hour_start).lte('date', hour_end).execute()
                        total_deleted_hours += 1
                        if total_deleted_hours % 24 == 0:  # Log every day (24 hours)
                            logger.info(f"  Deleted {total_deleted_hours} hours ({total_deleted_hours // 24} days) so far...")
                    except Exception as hour_error:
                        logger.warning(f"  Could not delete {hour_start}: {hour_error}")
                    
                    current_dt += timedelta(hours=1)
                
                logger.info(f"✓ Old price records deleted ({total_deleted_hours} hours = {total_deleted_hours // 24} days removed)")
                deletion_success = True
            else:
                logger.info("✓ No old data to delete")
                deletion_success = True
                
        except Exception as e:
            error_msg = str(e)
            logger.error("=" * 80)
            logger.error("❌ CRITICAL: DELETION FAILED!")
            logger.error("=" * 80)
            logger.error(f"Error: {e}")
            logger.error("⚠️ Old data WILL ACCUMULATE!")
            logger.error("=" * 80)
            # Continue anyway to keep app functional
        
        # === DELETION STEP 2: Today's synthetic data (cleanup) ===
        logger.info("Cleaning up today's synthetic gap data...")
        today_start = datetime.now().strftime('%Y-%m-%d 00:00:00')
        today_7am = datetime.now().strftime('%Y-%m-%d 07:00:00')
        
        try:
            supabase.table('prices').delete().eq('is_synthetic', True).gte('date', today_start).lt('date', today_7am).execute()
            logger.info("✓ Today's synthetic gap data cleaned up")
        except Exception as e:
            logger.warning(f"Could not delete today's synthetic data: {e}")
        
        # === INSERT NEW DATA ===
        data = df.to_dict('records')
        batch_size = 1000
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        logger.info(f"Inserting {len(data):,} price records in {total_batches} batches...")
        
        inserted_count = 0
        failed_batches = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            try:
                supabase.table('prices').insert(batch).execute()
                inserted_count += len(batch)
                
                batch_num = i//batch_size + 1
                if batch_num % 100 == 0 or batch_num == total_batches:
                    logger.info(f"  Progress: {batch_num}/{total_batches} batches ({inserted_count:,} records)")
                    
            except Exception as e:
                logger.warning(f"  Batch {batch_num} failed: {e}")
                failed_batches.append(batch_num)
        
        if failed_batches:
            logger.warning(f"Failed batches: {failed_batches[:10]}{'...' if len(failed_batches) > 10 else ''}")
        
        success_rate = (inserted_count / len(data) * 100) if len(data) > 0 else 0
        logger.info(f"✓ Prices updated: {inserted_count:,}/{len(data):,} ({success_rate:.1f}%)")
        
        # === GENERATE SYNTHETIC DATA FOR TOMORROW ===
        logger.info("\nGenerating synthetic data for tomorrow's gap...")
        synthetic_success = generate_synthetic_gap_data()
        
        if not synthetic_success:
            logger.warning("⚠️ Synthetic gap data generation had issues")
            logger.warning("   App may not work properly 00:00-07:00 tomorrow")
        
        # Return success if at least 95% of records were inserted
        return success_rate >= 95.0
        
    except Exception as e:
        logger.error(f"✗ Prices update failed: {e}", exc_info=True)
        return False


def generate_synthetic_gap_data():
    """
    Generate synthetic data for tomorrow's gap period (00:00-06:59).
    
    Formula: On day N at 7am:
    - Just uploaded: day (N-1) real data
    - Generate for: day (N+1) gap (00:00-06:59)
    - Source from: day (N-1) gap (time-matched, 48h old)
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("\n=== GENERATING SYNTHETIC GAP DATA ===")
    
    try:
        current_time = datetime.now()
        yesterday = current_time - timedelta(days=1)  # N-1 (source)
        tomorrow = current_time + timedelta(days=1)   # N+1 (target)
        
        tomorrow_date = tomorrow.date()
        yesterday_date = yesterday.date()
        
        logger.info(f"Target: {tomorrow_date} gap (00:00-06:59)")
        logger.info(f"Source: {yesterday_date} gap (00:00-06:59)")
        
        # Fetch source data from yesterday's gap
        yesterday_start = f"{yesterday_date} 00:00:00"
        yesterday_end = f"{yesterday_date} 06:59:59"
        
        logger.info(f"Fetching source: {yesterday_start} to {yesterday_end}...")
        
        # Fetch data in batches (Supabase has 1000-row server limit)
        all_source_data = []
        batch_size = 1000
        offset = 0
        
        logger.info("Fetching in batches (Supabase 1000-row limit)...")
        
        while True:
            response = (
                supabase.table('prices')
                .select('date,station_uuid,diesel,e5,e10')
                .eq('is_synthetic', False)
                .gte('date', yesterday_start)
                .lte('date', yesterday_end)
                .order('date', desc=False)
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            
            if not response.data:
                break  # No more data
            
            all_source_data.extend(response.data)
            
            # Log progress every 5 batches
            if len(all_source_data) % 5000 == 0:
                logger.info(f"  Fetched {len(all_source_data):,} records so far...")
            
            # If we got less than batch_size, we've reached the end
            if len(response.data) < batch_size:
                break
            
            offset += batch_size
            
            # Safety limit: stop after 50 batches (50k records)
            if offset >= 50000:
                logger.warning("Reached 50k record limit, stopping fetch")
                break
        
        if not all_source_data:
            logger.warning("⚠️ No source data from yesterday's gap")
            logger.warning("   Cannot generate synthetic data")
            return False
        
        source_df = pd.DataFrame(all_source_data)
        logger.info(f"✓ Fetched {len(source_df):,} source records total")

        
        # Generate synthetic records (time-matched)
        synthetic_records = []
        
        for _, row in source_df.iterrows():
            try:
                source_time = pd.to_datetime(row['date'])
                
                # Map to tomorrow's same time
                target_time = source_time.replace(
                    year=tomorrow_date.year,
                    month=tomorrow_date.month,
                    day=tomorrow_date.day
                )
                
                synthetic_records.append({
                    'date': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'station_uuid': row['station_uuid'],
                    'diesel': row.get('diesel'),
                    'e5': row.get('e5'),
                    'e10': row.get('e10'),
                    'dieselchange': 0,
                    'e5change': 0,
                    'e10change': 0,
                    'is_synthetic': True
                })
                
            except Exception as e:
                logger.warning(f"Could not process record: {e}")
                continue
        
        if not synthetic_records:
            logger.warning("⚠️ No synthetic records generated")
            return False
        
        total_records = len(synthetic_records)
        logger.info(f"Generated {total_records:,} synthetic records")
        
        # Insert in batches
        batch_size = 1000
        total_batches = (total_records + batch_size - 1) // batch_size
        logger.info(f"Inserting in {total_batches} batches...")
        
        inserted_count = 0
        for i in range(0, total_records, batch_size):
            batch = synthetic_records[i:i+batch_size]
            try:
                supabase.table('prices').insert(batch).execute()
                inserted_count += len(batch)
                
                batch_num = i//batch_size + 1
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"  Progress: {batch_num}/{total_batches} batches ({inserted_count:,} records)")
                    
            except Exception as e:
                logger.warning(f"  Batch {batch_num} failed: {e}")
        
        success_rate = (inserted_count / total_records * 100) if total_records > 0 else 0
        logger.info(f"✓ Synthetic: {inserted_count:,}/{total_records:,} ({success_rate:.1f}%)")
        
        return success_rate >= 95.0
        
    except Exception as e:
        logger.error(f"✗ Failed to generate synthetic gap data: {e}", exc_info=True)
        return False


def main():
    """
    Run daily update for stations and prices.
    
    WORKFLOW:
    1. Download & insert yesterday's stations (full refresh)
    2. Delete old prices (>14 days)
    3. Delete today's synthetic gap data (cleanup)
    4. Download & insert yesterday's prices
    5. Generate tomorrow's synthetic gap data (00:00-06:59)
    
    Gap Strategy:
    - At 7am on day N: generate synthetic for day (N+1)
    - Source: day (N-1) data (48h old, time-matched)
    - Enables 24/7 app usage
    """
    logger.info("=" * 60)
    logger.info("TANKERKÖNIG DAILY UPDATE STARTED")
    logger.info("=" * 60)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
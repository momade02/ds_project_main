"""
TEST VERSION: Uses prices_test table for safe testing.
After testing succeeds, switch to production version using 'prices' table.
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

# Setup paths and environment
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
env_path = project_root / ".env"
load_dotenv(env_path)

# Logging setup
log_dir = Path.home() / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Supabase client
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
    logger.error("✗ Tankerkönig credentials not found")
    sys.exit(1)

BASE_URL = f"https://{TANKERKOENIG_EMAIL}:{TANKERKOENIG_API_KEY}@data.tankerkoenig.de/tankerkoenig-organization/tankerkoenig-data/raw/branch/master"


def download_csv(url):
    """Download CSV from Tankerkönig."""
    try:
        safe_url = url.split('@')[1] if '@' in url else url
        logger.info(f"Downloading: {safe_url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        logger.info(f"✓ Downloaded {len(df):,} rows")
        return df
    
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        raise


def update_prices_test():
    """Update prices_test table (TEST VERSION)."""
    logger.info("\n=== UPDATING PRICES_TEST TABLE ===")
    
    try:
        yesterday = datetime.now() - timedelta(days=1)
        year = yesterday.year
        month = f"{yesterday.month:02d}"
        date_str = yesterday.strftime("%Y-%m-%d")
        
        url = f"{BASE_URL}/prices/{year}/{month}/{date_str}-prices.csv"
        df = download_csv(url)
        
        # Prepare data
        logger.info("Preparing prices data...")
        
        def strip_timezone(date_str):
            if pd.isna(date_str) or date_str == '' or date_str is None:
                return None
            try:
                dt = pd.to_datetime(date_str)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return None
        
        df['date'] = df['date'].apply(strip_timezone)
        logger.info(f"✓ Processed {df['date'].notna().sum():,} timestamps")
        
        for col in ['diesel', 'e5', 'e10']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for col in ['dieselchange', 'e5change', 'e10change']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df = df.where(pd.notnull(df), None)
        
        # Delete old data (>14 days)
        cutoff_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Deleting prices_test older than {cutoff_date}...")
        
        try:
            supabase.table('prices_test').delete().lt('date', cutoff_date).execute()
            logger.info("✓ Old records deleted")
        except Exception as e:
            logger.error(f"✗ Deletion failed: {e}")
        
        # Delete today's synthetic data
        logger.info("Cleaning up today's synthetic gap data...")
        today_start = datetime.now().strftime('%Y-%m-%d 00:00:00')
        today_7am = datetime.now().strftime('%Y-%m-%d 07:00:00')
        
        try:
            supabase.table('prices_test').delete().eq('is_synthetic', True).gte('date', today_start).lt('date', today_7am).execute()
            logger.info("✓ Synthetic data cleaned up")
        except Exception as e:
            logger.warning(f"Could not delete synthetic data: {e}")
        
        # Insert new data
        data = df.to_dict('records')
        batch_size = 1000
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        logger.info(f"Inserting {len(data):,} records in {total_batches} batches...")
        
        inserted_count = 0
        failed_batches = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_num = i // batch_size + 1  # FIXED: Define BEFORE try block
            
            try:
                supabase.table('prices_test').insert(batch).execute()
                inserted_count += len(batch)
                
                if batch_num % 100 == 0 or batch_num == total_batches:
                    logger.info(f"  Progress: {batch_num}/{total_batches} batches ({inserted_count:,})")
                    
            except Exception as e:
                logger.warning(f"  Batch {batch_num} failed: {e}")
                failed_batches.append(batch_num)
        
        if failed_batches:
            logger.warning(f"Failed batches: {failed_batches[:10]}")
        
        success_rate = (inserted_count / len(data) * 100) if len(data) > 0 else 0
        logger.info(f"✓ Inserted: {inserted_count:,}/{len(data):,} ({success_rate:.1f}%)")
        
        # Generate synthetic data
        logger.info("\n🧪 Testing synthetic generation...")
        synthetic_success = generate_synthetic_gap_data_test()
        
        if not synthetic_success:
            logger.warning("⚠️ Synthetic generation had issues")
        
        return success_rate >= 95.0
        
    except Exception as e:
        logger.error(f"✗ Update failed: {e}", exc_info=True)
        return False


def generate_synthetic_gap_data_test():
    """Generate synthetic data for prices_test (TEST VERSION)."""
    logger.info("\n=== GENERATING SYNTHETIC GAP DATA (TEST) ===")
    
    try:
        current_time = datetime.now()
        yesterday = current_time - timedelta(days=1)  # Source
        tomorrow = current_time + timedelta(days=1)   # Target
        
        tomorrow_date = tomorrow.date()
        yesterday_date = yesterday.date()
        
        logger.info(f"Target: {tomorrow_date} gap (00:00-06:59)")
        logger.info(f"Source: {yesterday_date} gap (00:00-06:59)")
        
        # Fetch source data
        yesterday_start = f"{yesterday_date} 00:00:00"
        yesterday_end = f"{yesterday_date} 06:59:59"
        
        logger.info(f"Fetching source from prices_test...")
        
        response = (
            supabase.table('prices_test')
            .select('date,station_uuid,diesel,e5,e10')
            .eq('is_synthetic', False)
            .gte('date', yesterday_start)
            .lte('date', yesterday_end)
            .order('date', desc=False)
            .execute()
        )
        
        if not response.data:
            logger.warning("⚠️ No source data in 00:00-06:59 range")
            logger.warning("   Trying alternative: any records from yesterday")
            
            # Fallback: get ANY records from yesterday
            response = (
                supabase.table('prices_test')
                .select('date,station_uuid,diesel,e5,e10')
                .eq('is_synthetic', False)
                .gte('date', f"{yesterday_date} 00:00:00")
                .lte('date', f"{yesterday_date} 23:59:59")
                .order('date', desc=False)
                .limit(100000)
                .execute()
            )
            
            if not response.data:
                logger.error("✗ No data from yesterday at all!")
                return False
        
        source_df = pd.DataFrame(response.data)
        logger.info(f"✓ Fetched {len(source_df):,} source records")
        
        # Generate synthetic records
        synthetic_records = []
        
        for _, row in source_df.iterrows():
            try:
                source_time = pd.to_datetime(row['date'])
                
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
            batch_num = i // batch_size + 1
            
            try:
                supabase.table('prices_test').insert(batch).execute()
                inserted_count += len(batch)
                
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"  Progress: {batch_num}/{total_batches} batches ({inserted_count:,})")
                    
            except Exception as e:
                logger.warning(f"  Batch {batch_num} failed: {e}")
        
        success_rate = (inserted_count / total_records * 100) if total_records > 0 else 0
        logger.info(f"✓ Synthetic: {inserted_count:,}/{total_records:,} ({success_rate:.1f}%)")
        
        return success_rate >= 95.0
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}", exc_info=True)
        return False


def main():
    """TEST RUN on prices_test table."""
    logger.info("=" * 60)
    logger.info("🧪 TEST MODE - USING prices_test TABLE")
    logger.info("=" * 60)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    prices_success = update_prices_test()
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Result: {'✅ SUCCESS' if prices_success else '❌ FAILED'}")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("=" * 60)
    
    if prices_success:
        logger.info("\n✅ TEST PASSED!")
        logger.info("\nNext steps:")
        logger.info("1. Verify in Supabase:")
        logger.info("   SELECT COUNT(*) FROM prices_test WHERE is_synthetic = TRUE;")
        logger.info("2. Check synthetic data dates:")
        logger.info("   SELECT MIN(date), MAX(date) FROM prices_test WHERE is_synthetic = TRUE;")
        logger.info("3. If all good, update script to use 'prices' table")
    else:
        logger.info("\n❌ TEST FAILED - Review errors above")
    
    logger.info("=" * 60)
    
    sys.exit(0 if prices_success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n✗ Test interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        sys.exit(1)

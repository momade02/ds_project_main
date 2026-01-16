"""
Test Script: Verify Synthetic Data Cleanup
Simulates running the update script on different days to test cleanup logic.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import logging
from dotenv import load_dotenv

# Setup
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
env_path = project_root / ".env"
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from supabase import create_client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SECRET_KEY")
)


def check_synthetic_data():
    """Check what synthetic data exists in prices_test."""
    logger.info("\n=== CHECKING SYNTHETIC DATA ===")
    
    response = (
        supabase.table('prices_test')
        .select('date, is_synthetic')
        .eq('is_synthetic', True)
        .execute()
    )
    
    if not response.data:
        logger.info("✓ No synthetic data found")
        return
    
    df = pd.DataFrame(response.data)
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    
    summary = df.groupby('date_only').size().reset_index(name='count')
    
    logger.info(f"Found {len(df)} synthetic records:")
    for _, row in summary.iterrows():
        logger.info(f"  {row['date_only']}: {row['count']} records")
    
    return df


def simulate_day_run(simulate_date: datetime, source_date: datetime):
    """
    Simulate running the update script on a specific day.
    
    Args:
        simulate_date: The day we're pretending to run the script (e.g., Jan 16)
        source_date: The day we're using as source data (e.g., Jan 14)
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"🧪 SIMULATING RUN ON: {simulate_date.date()}")
    logger.info(f"   Using source data from: {source_date.date()}")
    logger.info("=" * 70)
    
    # Calculate target date (tomorrow from simulate_date)
    target_date = simulate_date + timedelta(days=1)
    
    logger.info(f"\n1. Cleanup: Delete synthetic for TODAY ({simulate_date.date()})")
    today_start = f"{simulate_date.date()} 00:00:00"
    today_7am = f"{simulate_date.date()} 07:00:00"
    
    try:
        delete_result = (
            supabase.table('prices_test')
            .delete()
            .eq('is_synthetic', True)
            .gte('date', today_start)
            .lt('date', today_7am)
            .execute()
        )
        logger.info(f"   ✓ Deleted synthetic for {simulate_date.date()}")
    except Exception as e:
        logger.warning(f"   ✗ Cleanup failed: {e}")
    
    logger.info(f"\n2. Fetch source data from {source_date.date()} (00:00-06:59)")
    source_start = f"{source_date.date()} 00:00:00"
    source_end = f"{source_date.date()} 06:59:59"
    
    response = (
        supabase.table('prices_test')
        .select('date,station_uuid,diesel,e5,e10')
        .eq('is_synthetic', False)
        .gte('date', source_start)
        .lte('date', source_end)
        .limit(1000)  # Limit for testing
        .execute()
    )
    
    if not response.data:
        logger.error(f"   ✗ No source data found for {source_date.date()}")
        return False
    
    logger.info(f"   ✓ Found {len(response.data)} source records")
    
    logger.info(f"\n3. Generate synthetic for TOMORROW ({target_date.date()}) 00:00-06:59")
    
    synthetic_records = []
    for row in response.data:
        try:
            source_time = pd.to_datetime(row['date'])
            
            # Map to target date, same time
            target_time = source_time.replace(
                year=target_date.year,
                month=target_date.month,
                day=target_date.day
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
        except:
            continue
    
    if not synthetic_records:
        logger.error("   ✗ No synthetic records generated")
        return False
    
    logger.info(f"   Generated {len(synthetic_records)} synthetic records")
    
    # Insert in batches
    batch_size = 500
    inserted = 0
    for i in range(0, len(synthetic_records), batch_size):
        batch = synthetic_records[i:i+batch_size]
        try:
            supabase.table('prices_test').insert(batch).execute()
            inserted += len(batch)
        except Exception as e:
            logger.warning(f"   Batch {i//batch_size + 1} failed: {e}")
    
    logger.info(f"   ✓ Inserted {inserted}/{len(synthetic_records)} synthetic records")
    
    return True


def main():
    """
    Test sequence:
    1. Check initial state
    2. Simulate Jan 15 run (generate synthetic for Jan 16)
    3. Check synthetic data exists for Jan 16
    4. Simulate Jan 16 run (should delete Jan 16, generate for Jan 17)
    5. Verify Jan 16 synthetic is gone, Jan 17 exists
    """
    
    logger.info("=" * 70)
    logger.info("🧪 SYNTHETIC DATA CLEANUP TEST")
    logger.info("=" * 70)
    
    # Step 0: Initial state
    logger.info("\n📊 STEP 0: Check initial state")
    check_synthetic_data()
    
    # Step 1: Simulate Jan 15 run
    logger.info("\n📊 STEP 1: Simulate run on Jan 15 (source: Jan 13)")
    jan_15 = datetime(2026, 1, 15, 7, 0, 0)
    jan_13 = datetime(2026, 1, 13, 0, 0, 0)
    
    success = simulate_day_run(jan_15, jan_13)
    if not success:
        logger.error("❌ Step 1 failed")
        return
    
    # Check result
    logger.info("\n📊 After Step 1: Should have synthetic for Jan 16")
    df = check_synthetic_data()
    if df is not None:
        jan_16_count = len(df[df['date'].dt.date == datetime(2026, 1, 16).date()])
        if jan_16_count > 0:
            logger.info(f"✅ SUCCESS: Found {jan_16_count} synthetic records for Jan 16")
        else:
            logger.error("❌ FAIL: No synthetic records for Jan 16 found!")
            return
    
    # Step 2: Simulate Jan 16 run
    logger.info("\n📊 STEP 2: Simulate run on Jan 16 (source: Jan 14)")
    logger.info("   This should:")
    logger.info("   - DELETE synthetic for Jan 16 (cleanup)")
    logger.info("   - GENERATE synthetic for Jan 17")
    
    jan_16 = datetime(2026, 1, 16, 7, 0, 0)
    jan_14 = datetime(2026, 1, 14, 0, 0, 0)
    
    success = simulate_day_run(jan_16, jan_14)
    if not success:
        logger.error("❌ Step 2 failed")
        return
    
    # Final check
    logger.info("\n📊 After Step 2: Jan 16 should be GONE, Jan 17 should exist")
    df = check_synthetic_data()
    if df is not None:
        jan_16_count = len(df[df['date'].dt.date == datetime(2026, 1, 16).date()])
        jan_17_count = len(df[df['date'].dt.date == datetime(2026, 1, 17).date()])
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 FINAL RESULTS")
        logger.info("=" * 70)
        logger.info(f"Jan 16 synthetic records: {jan_16_count} (should be 0)")
        logger.info(f"Jan 17 synthetic records: {jan_17_count} (should be >0)")
        
        if jan_16_count == 0 and jan_17_count > 0:
            logger.info("\n✅✅✅ TEST PASSED! ✅✅✅")
            logger.info("Cleanup is working correctly:")
            logger.info("  ✓ Old synthetic data was deleted")
            logger.info("  ✓ New synthetic data was generated")
        else:
            logger.error("\n❌❌❌ TEST FAILED! ❌❌❌")
            if jan_16_count > 0:
                logger.error(f"  ✗ Jan 16 synthetic NOT deleted ({jan_16_count} records remain)")
            if jan_17_count == 0:
                logger.error("  ✗ Jan 17 synthetic NOT generated")
    else:
        logger.error("❌ No synthetic data found at all!")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n✗ Test interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        sys.exit(1)

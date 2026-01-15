#!/bin/bash
# Quick check for deletion failures
# Run this daily for the first week after deployment

echo "======================================"
echo "Tankerkönig Data Update - Health Check"
echo "======================================"
echo ""

# Check if DELETION_FAILURES.txt exists
if [ -f ~/logs/DELETION_FAILURES.txt ]; then
    echo "⚠️  ALERT: Deletion failures detected!"
    echo ""
    echo "Last 5 failures:"
    tail -20 ~/logs/DELETION_FAILURES.txt
    echo ""
    echo "❌ ACTION REQUIRED: Investigate deletion issues"
    echo ""
else
    echo "✓ No deletion failures detected"
    echo ""
fi

# Check today's log
TODAY=$(date +%Y%m%d)
if [ -f ~/logs/daily_update_${TODAY}.log ]; then
    echo "Today's run status:"
    echo "---"
    tail -30 ~/logs/daily_update_${TODAY}.log | grep -E "(DELETION|SUCCESS|FAILED|CRITICAL)" | tail -10
    echo ""
else
    echo "⚠️  No log file for today yet"
    echo ""
fi

# Check most recent log for deletion status
LATEST_LOG=$(ls -t ~/logs/daily_update_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    LATEST_DATE=$(basename $LATEST_LOG | sed 's/daily_update_//;s/.log//')
    echo "Most recent run: $LATEST_DATE"
    
    # Check if deletion was successful
    if grep -q "Old price records deleted successfully" "$LATEST_LOG"; then
        echo "✓ Deletion: SUCCESS"
    elif grep -q "DELETION FAILED" "$LATEST_LOG"; then
        echo "❌ Deletion: FAILED"
    else
        echo "? Deletion: UNKNOWN (check log)"
    fi
    
    # Check if upload was successful
    if grep -q "Prices:   ✓ SUCCESS" "$LATEST_LOG"; then
        echo "✓ Upload:   SUCCESS"
    else
        echo "❌ Upload:   FAILED"
    fi
fi

echo ""
echo "======================================"
echo ""
echo "For detailed logs, run:"
echo "  tail -100 ~/logs/daily_update_${TODAY}.log"
echo ""
echo "To check database size, run in Supabase:"
echo "  SELECT COUNT(*) FROM prices;"
echo "  (Should be ~7-8 million rows for 14 days)"
echo ""

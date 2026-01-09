#!/bin/bash
# Bash script to run Locust load tests with automatic result saving
# Usage: ./run_locust_load_test.sh -u 50 -r 10 -t 5m

# Default values
USERS=50
SPAWN_RATE=10
RUN_TIME="5m"
HOST="http://localhost:8000"
HEADLESS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--users)
            USERS="$2"
            shift 2
            ;;
        -r|--spawn-rate)
            SPAWN_RATE="$2"
            shift 2
            ;;
        -t|--run-time)
            RUN_TIME="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        --no-headless)
            HEADLESS=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get timestamp for unique file names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Set report directory (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/reports"

# Ensure reports directory exists
mkdir -p "$REPORTS_DIR"

# Build CSV and HTML file paths
CSV_PREFIX="$REPORTS_DIR/locust_load_$TIMESTAMP"
HTML_FILE="$REPORTS_DIR/locust_load_$TIMESTAMP.html"
LOG_FILE="$REPORTS_DIR/locust_load_$TIMESTAMP.log"

# Build Locust command
LOCUST_FILE="$PROJECT_ROOT/tests/performance/locustLoad.py"
CMD="locust -f \"$LOCUST_FILE\" --host=$HOST -u $USERS -r $SPAWN_RATE --run-time $RUN_TIME"

if [ "$HEADLESS" = true ]; then
    CMD="$CMD --headless"
fi

CMD="$CMD --csv=\"$CSV_PREFIX\""
CMD="$CMD --html=\"$HTML_FILE\""
CMD="$CMD --logfile=\"$LOG_FILE\""

echo "========================================"
echo "Locust Load Test Configuration"
echo "========================================"
echo "Users: $USERS"
echo "Spawn Rate: $SPAWN_RATE users/second"
echo "Run Time: $RUN_TIME"
echo "Host: $HOST"
echo "Headless: $HEADLESS"
echo ""
echo "Results will be saved to:"
echo "  CSV: ${CSV_PREFIX}*.csv"
echo "  HTML: $HTML_FILE"
echo "  Log: $LOG_FILE"
echo ""
echo "Running Locust..."
echo "========================================"
echo ""

# Run Locust
eval $CMD

echo ""
echo "========================================"
echo "Test completed!"
echo "========================================"
echo "Results saved to:"
echo "  CSV files: ${CSV_PREFIX}*.csv"
echo "  HTML report: $HTML_FILE"
echo "  Log file: $LOG_FILE"


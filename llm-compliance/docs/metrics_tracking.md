# Metrics Tracking Across CI/CD Runs

This document explains how to track mean scores over multiple Jenkins CI/CD runs.

## Overview

The metrics tracking system allows you to:
- Track mean scores for metrics across multiple test runs
- Analyze trends over time (improving, declining, stable)
- View historical data and generate summary reports
- Archive tracking data in Jenkins for long-term analysis

## Architecture

### Components

1. **MetricsHistoryTracker** (`utils/metricsHistoryTracker.py`)
   - Core utility for tracking metrics
   - Appends results to `reports/metrics_history.csv`
   - Provides methods to query and analyze historical data

2. **Test Integration**
   - Tests automatically append mean scores to history after evaluation
   - Currently integrated in `test_ragas_rag.py`
   - Can be added to any test that calculates mean scores

3. **Jenkins Integration**
   - Jenkinsfile archives `metrics_history.csv` and `metrics_summary.json`
   - Build numbers and git info are automatically captured
   - Historical data persists across builds

4. **Visualization Script** (`scripts/view_metrics_trends.py`)
   - Command-line tool to view trends
   - Supports filtering by test suite and metric
   - Generates formatted reports

## Usage

### In Tests

The tracking is automatically integrated in tests. Example from `test_ragas_rag.py`:

```python
from utils.metricsHistoryTracker import MetricsHistoryTracker

# After calculating metric_averages
tracker = MetricsHistoryTracker()
tracker.append_metrics(
    test_suite="ragas_rag",
    metric_averages=metric_averages,
    run_id=timestamp
)
```

### Viewing Trends

#### View all trends:
```bash
python scripts/view_metrics_trends.py
```

#### View trends for a specific test suite:
```bash
python scripts/view_metrics_trends.py --test-suite ragas_rag
```

#### View trend for a specific metric:
```bash
python scripts/view_metrics_trends.py --test-suite ragas_rag --metric faithfulness
```

#### View summary report:
```bash
python scripts/view_metrics_trends.py --summary
```

### Programmatic Access

```python
from utils.metricsHistoryTracker import MetricsHistoryTracker

tracker = MetricsHistoryTracker()

# Get latest scores
latest = tracker.get_latest_scores("ragas_rag")
print(latest)

# Get trend for a metric
trend = tracker.get_trend_summary("ragas_rag", "faithfulness", num_runs=10)
print(f"Trend: {trend['trend']}")  # IMPROVING, DECLINING, or STABLE

# Get full history
history = tracker.get_metric_history(test_suite="ragas_rag", limit=20)

# Generate summary report
tracker.generate_summary_report()
```

## Data Format

### History File (`reports/metrics_history.csv`)

Each row represents one metric from one test run:

| Column | Description |
|--------|-------------|
| timestamp | ISO format timestamp of the run |
| run_id | Unique identifier for the run |
| test_suite | Name of the test suite |
| metric | Name of the metric |
| mean_score | Mean score for this metric |
| mean_threshold | Threshold that was used |
| number_of_tests | Number of test cases evaluated |
| status | PASS or FAIL |
| build_number | Jenkins build number (if available) |
| git_commit | Git commit hash (if available) |
| git_branch | Git branch name (if available) |

### Summary Report (`reports/metrics_summary.json`)

JSON file with aggregated statistics:

```json
{
  "generated_at": "2025-12-28T10:30:00",
  "total_runs": 15,
  "test_suites": {
    "ragas_rag": {
      "latest_scores": {
        "faithfulness": 0.85,
        "answer_relevancy": 0.72
      },
      "metrics": {
        "faithfulness": {
          "trend": "IMPROVING",
          "latest_score": 0.85,
          "average_score": 0.82
        }
      }
    }
  }
}
```

## Jenkins Integration

The Jenkinsfile automatically:
1. Archives `metrics_history.csv` after each run
2. Archives `metrics_summary.json` after generation
3. Captures build number and git information
4. Makes historical data available as build artifacts

### Accessing Historical Data in Jenkins

1. Go to any build's "Build Artifacts"
2. Download `metrics_history.csv` to see all historical data
3. Download `metrics_summary.json` for aggregated view

### Viewing Trends in Jenkins

You can add a post-build step to generate and display trends:

```groovy
post {
    always {
        sh '''
            . .venv/bin/activate
            python scripts/view_metrics_trends.py --summary > metrics_trends.txt || true
        '''
        archiveArtifacts artifacts: 'metrics_trends.txt', allowEmptyArchive: true
    }
}
```

## Adding Tracking to New Tests

To add tracking to a new test:

1. Import the tracker:
```python
from utils.metricsHistoryTracker import MetricsHistoryTracker
```

2. After calculating `metric_averages`, append to history:
```python
if metric_averages:
    try:
        tracker = MetricsHistoryTracker()
        tracker.append_metrics(
            test_suite="your_test_suite_name",
            metric_averages=metric_averages,
            run_id=timestamp
        )
    except Exception as e:
        print(f"⚠️  Warning: Failed to track metrics: {e}")
```

3. Update Jenkinsfile to archive your test's summary files if needed.

## Best Practices

1. **Test Suite Naming**: Use consistent, descriptive names (e.g., `ragas_rag`, `deepeval_rag_singleturn`)

2. **Error Handling**: Always wrap tracking in try-except to avoid failing tests if tracking fails

3. **Run ID**: Use timestamps or build numbers for unique run identification

4. **Metric Names**: Use consistent metric names across runs (they should match the keys in `metric_averages`)

5. **History File Size**: The history file grows over time. Consider archiving old data periodically if it becomes too large.

## Troubleshooting

### History file not being created
- Check that `reports/` directory exists and is writable
- Verify the test is calling `append_metrics()`

### Trends showing "NO_DATA"
- Ensure at least one run has completed with tracking enabled
- Check that the test suite name matches exactly

### Build info not captured
- Jenkins environment variables (`BUILD_NUMBER`, `GIT_COMMIT`, `GIT_BRANCH`) are automatically used if available
- For local runs, these will be empty (which is fine)

## Future Enhancements

Potential improvements:
- Web dashboard for visualizing trends
- Alerting when metrics decline significantly
- Integration with monitoring tools (Grafana, etc.)
- Automated trend analysis and reporting
- Support for multiple history files (by test suite)


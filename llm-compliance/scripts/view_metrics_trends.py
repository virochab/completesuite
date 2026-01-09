"""Script to view and visualize metrics trends over time.

Usage:
    python scripts/view_metrics_trends.py
    python scripts/view_metrics_trends.py --test-suite ragas_rag
    python scripts/view_metrics_trends.py --test-suite ragas_rag --metric faithfulness
    python scripts/view_metrics_trends.py --summary
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metricsHistoryTracker import MetricsHistoryTracker


def print_trend_table(trends: list):
    """Print trends in a formatted table."""
    if not trends:
        print("No trend data available.")
        return
    
    # Print header
    print("\n" + "=" * 100)
    print(f"{'Metric':<30} {'Latest':<10} {'Avg (10 runs)':<15} {'Min':<8} {'Max':<8} {'Trend':<12} {'Runs':<6}")
    print("=" * 100)
    
    for trend in trends:
        metric = trend.get('metric', 'N/A')
        latest = trend.get('latest_score', 0.0)
        avg = trend.get('average_score', 0.0)
        min_score = trend.get('min_score', 0.0)
        max_score = trend.get('max_score', 0.0)
        trend_dir = trend.get('trend', 'UNKNOWN')
        num_runs = trend.get('num_runs', 0)
        
        # Format trend with emoji
        trend_emoji = {
            'IMPROVING': '📈',
            'DECLINING': '📉',
            'STABLE': '➡️',
            'INSUFFICIENT_DATA': '⚠️',
            'NO_DATA': '❌'
        }.get(trend_dir, '❓')
        
        print(f"{metric:<30} {latest:<10.4f} {avg:<15.4f} {min_score:<8.4f} {max_score:<8.4f} {trend_emoji} {trend_dir:<10} {num_runs:<6}")


def print_latest_scores(scores: dict, test_suite: str):
    """Print latest scores for a test suite."""
    if not scores:
        print(f"No scores available for test suite: {test_suite}")
        return
    
    print(f"\n📊 Latest Scores for '{test_suite}':")
    print("-" * 60)
    for metric, score in sorted(scores.items()):
        print(f"  {metric:<30} {score:.4f}")


def print_summary_report(summary_path: Path):
    """Print summary report from JSON file."""
    if not summary_path.exists():
        print(f"Summary report not found: {summary_path}")
        return
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print("\n" + "=" * 80)
    print("METRICS SUMMARY REPORT")
    print("=" * 80)
    print(f"Generated at: {summary.get('generated_at', 'N/A')}")
    print(f"Total runs tracked: {summary.get('total_runs', 0)}")
    print("\nTest Suites:")
    
    for test_suite, suite_data in summary.get('test_suites', {}).items():
        print(f"\n  📦 {test_suite}")
        print("    Latest Scores:")
        latest_scores = suite_data.get('latest_scores', {})
        for metric, score in sorted(latest_scores.items()):
            print(f"      {metric:<30} {score:.4f}")
        
        print("    Trends (last 10 runs):")
        metrics = suite_data.get('metrics', {})
        trends = []
        for metric_name, trend_data in metrics.items():
            trends.append(trend_data)
        print_trend_table(trends)


def main():
    parser = argparse.ArgumentParser(
        description="View and visualize metrics trends over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all trends
  python scripts/view_metrics_trends.py
  
  # View trends for a specific test suite
  python scripts/view_metrics_trends.py --test-suite ragas_rag
  
  # View trend for a specific metric
  python scripts/view_metrics_trends.py --test-suite ragas_rag --metric faithfulness
  
  # View summary report
  python scripts/view_metrics_trends.py --summary
        """
    )
    
    parser.add_argument(
        '--test-suite',
        type=str,
        help='Filter by test suite name (e.g., ragas_rag, deepeval_rag_singleturn)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        help='Filter by metric name (e.g., faithfulness, answer_relevancy)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Display summary report from JSON file'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help='Number of recent runs to analyze (default: 10)'
    )
    
    args = parser.parse_args()
    
    tracker = MetricsHistoryTracker()
    
    if args.summary:
        # Display summary report
        summary_path = tracker.reports_dir / "metrics_summary.json"
        print_summary_report(summary_path)
    elif args.test_suite:
        if args.metric:
            # Show trend for specific metric
            trend = tracker.get_trend_summary(
                test_suite=args.test_suite,
                metric=args.metric,
                num_runs=args.num_runs
            )
            print(f"\n📈 Trend Analysis: {args.test_suite} - {args.metric}")
            print("-" * 60)
            print(f"Latest Score: {trend.get('latest_score', 'N/A')}")
            print(f"Average Score (last {trend.get('num_runs', 0)} runs): {trend.get('average_score', 'N/A')}")
            print(f"Min Score: {trend.get('min_score', 'N/A')}")
            print(f"Max Score: {trend.get('max_score', 'N/A')}")
            print(f"Trend: {trend.get('trend', 'N/A')}")
            print(f"Number of Runs: {trend.get('num_runs', 0)}")
        else:
            # Show all metrics for test suite
            latest_scores = tracker.get_latest_scores(args.test_suite)
            print_latest_scores(latest_scores, args.test_suite)
            
            # Get trends for all metrics
            history = tracker.get_metric_history(test_suite=args.test_suite, limit=100)
            metrics = set(entry.get('metric') for entry in history)
            
            trends = []
            for metric in metrics:
                trend = tracker.get_trend_summary(
                    test_suite=args.test_suite,
                    metric=metric,
                    num_runs=args.num_runs
                )
                trends.append(trend)
            
            if trends:
                print(f"\n📈 Trends for '{args.test_suite}' (last {args.num_runs} runs):")
                print_trend_table(trends)
    else:
        # Show summary of all test suites
        summary_path = tracker.reports_dir / "metrics_summary.json"
        if summary_path.exists():
            print_summary_report(summary_path)
        else:
            print("No summary report found. Generating one...")
            tracker.generate_summary_report()
            print_summary_report(summary_path)


if __name__ == "__main__":
    main()


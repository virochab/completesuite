"""Utility for tracking metric scores across multiple CI/CD runs.

This module provides functionality to:
- Append mean scores from test runs to a historical tracking file
- Track trends over time
- Generate summary statistics
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import os


class MetricsHistoryTracker:
    """Track metric scores across multiple CI/CD runs."""
    
    def __init__(self, project_root: Optional[Path] = None, history_file: Optional[str] = None):
        """
        Initialize the metrics history tracker.
        
        Args:
            project_root: Root directory of the project (defaults to parent of utils/)
            history_file: Name of the history file (defaults to 'metrics_history.csv')
        """
        if project_root is None:
            # Default to parent of utils/ directory
            project_root = Path(__file__).parent.parent
        
        self.project_root = project_root
        self.reports_dir = project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        if history_file is None:
            history_file = "metrics_history.csv"
        
        self.history_file = self.reports_dir / history_file
        
        # Ensure history file exists with header if it doesn't
        self._ensure_history_file()
    
    def _get_metric_score_columns(self):
        """Get the standard metric score columns in the specified order."""
        # Define RAGAS metrics in the specified order (only scores, no threshold/status)
        metrics_order = [
            'answer_correctness',
            'context_entity_recall',
            'context_precision',
            'context_recall',
            'faithfulness'
        ]
        
        # Return only score columns
        return [f'{metric}_score' for metric in metrics_order]
    
    def _ensure_history_file(self):
        """Ensure history file exists with proper header."""
        if not self.history_file.exists():
            # Write header with exact column order as specified
            metric_score_columns = self._get_metric_score_columns()
            
            with open(self.history_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Column order: timestamp, build_number, run_id, test_suite, number_of_tests, git_commit, git_branch, then metric scores
                header = [
                    'timestamp',
                    'build_number',
                    'run_id',
                    'test_suite',
                    'number_of_tests',
                    'git_commit',
                    'git_branch'
                ] + metric_score_columns
                writer.writerow(header)
    
    def append_metrics(
        self,
        test_suite: str,
        metric_averages: Dict[str, Dict[str, Any]],
        run_id: Optional[str] = None,
        build_number: Optional[str] = None,
        git_commit: Optional[str] = None,
        git_branch: Optional[str] = None
    ):
        """
        Append metric scores from a test run to the history file.
        Each run is stored as a single row with metrics as columns.
        
        Args:
            test_suite: Name of the test suite (e.g., 'ragas_rag', 'deepeval_rag_singleturn')
            metric_averages: Dictionary mapping metric names to their average data
                Format: {
                    'metric_name': {
                        'average': float,
                        'mean_threshold': float,
                        'total_count': int,
                        'status': str,
                        ...
                    }
                }
            run_id: Optional unique identifier for this run (defaults to timestamp)
            build_number: Optional Jenkins build number
            git_commit: Optional git commit hash
            git_branch: Optional git branch name
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        timestamp = datetime.now().isoformat()
        
        # Get build info from environment if not provided
        if build_number is None:
            build_number = os.environ.get('BUILD_NUMBER', '')
        if git_commit is None:
            git_commit = os.environ.get('GIT_COMMIT', '')[:8] if os.environ.get('GIT_COMMIT') else ''
        if git_branch is None:
            git_branch = os.environ.get('GIT_BRANCH', '').replace('origin/', '') if os.environ.get('GIT_BRANCH') else ''
        
        # Get total number of tests (use max or sum, depending on your preference)
        total_tests = max((m.get('total_count', 0) for m in metric_averages.values()), default=0)
        
        # Use standard column order (only score columns, no threshold/status)
        metric_score_columns = self._get_metric_score_columns()
        fixed_columns = [
            'timestamp',
            'build_number',
            'run_id',
            'test_suite',
            'number_of_tests',
            'git_commit',
            'git_branch'
        ]
        all_columns = fixed_columns + metric_score_columns
        
        # Read all existing rows
        existing_rows = []
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        
        # Build new row with exact column order
        new_row = {
            'timestamp': timestamp,
            'build_number': build_number,
            'run_id': run_id,
            'test_suite': test_suite,
            'number_of_tests': total_tests,
            'git_commit': git_commit,
            'git_branch': git_branch
        }
        
        # Add metric score columns in the specified order (only scores, no threshold/status)
        metrics_order = [
            'answer_correctness',
            'context_entity_recall',
            'context_precision',
            'context_recall',
            'faithfulness'
        ]
        
        for metric_name in metrics_order:
            if metric_name in metric_averages:
                metric_data = metric_averages[metric_name]
                new_row[f'{metric_name}_score'] = round(metric_data.get('average', 0.0), 4)
            else:
                # Metric not in this run, leave empty
                new_row[f'{metric_name}_score'] = ''
        
        # Ensure all existing rows have all columns (fill missing with empty string)
        for row in existing_rows:
            for col in all_columns:
                if col not in row:
                    row[col] = ''
        
        # Write all rows back with updated columns
        with open(self.history_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(new_row)
        
        print(f"✅ Metrics appended to history: {self.history_file} (1 row per run)")
    
    def get_metric_history(
        self,
        test_suite: Optional[str] = None,
        metric: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve metric history from the tracking file.
        
        Args:
            test_suite: Filter by test suite name (optional)
            metric: Filter by metric name (optional) - returns rows with that metric's data
            limit: Limit number of results (optional)
        
        Returns:
            List of dictionaries with historical run data (one dict per run)
        """
        if not self.history_file.exists():
            return []
        
        results = []
        with open(self.history_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Apply filters
                if test_suite and row.get('test_suite') != test_suite:
                    continue
                
                # Convert numeric fields
                if row.get('number_of_tests'):
                    try:
                        row['number_of_tests'] = int(row.get('number_of_tests', 0))
                    except (ValueError, TypeError):
                        row['number_of_tests'] = 0
                
                # If filtering by metric, check if metric exists in this row
                if metric:
                    metric_score_col = f'{metric}_score'
                    if metric_score_col not in row or not row.get(metric_score_col):
                        continue
                
                results.append(row)
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results
    
    def get_latest_scores(self, test_suite: str) -> Dict[str, float]:
        """
        Get the latest mean scores for all metrics in a test suite.
        
        Args:
            test_suite: Name of the test suite
        
        Returns:
            Dictionary mapping metric names to their latest mean scores
        """
        history = self.get_metric_history(test_suite=test_suite, limit=1)
        
        if not history:
            return {}
        
        # Get the most recent run (first in sorted list)
        latest_run = history[0]
        latest_scores = {}
        
        # Extract all metric scores from the row
        for key, value in latest_run.items():
            if key.endswith('_score') and value:
                try:
                    metric_name = key[:-6]  # Remove '_score' suffix
                    latest_scores[metric_name] = float(value)
                except (ValueError, TypeError):
                    continue
        
        return latest_scores
    
    def get_trend_summary(
        self,
        test_suite: str,
        metric: str,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Get trend summary for a specific metric.
        
        Args:
            test_suite: Name of the test suite
            metric: Name of the metric
            num_runs: Number of recent runs to analyze
        
        Returns:
            Dictionary with trend statistics
        """
        history = self.get_metric_history(
            test_suite=test_suite,
            metric=metric,
            limit=num_runs
        )
        
        if not history:
            return {
                'metric': metric,
                'test_suite': test_suite,
                'num_runs': 0,
                'latest_score': None,
                'average_score': None,
                'trend': 'NO_DATA'
            }
        
        # Extract scores for this metric from each run
        metric_score_col = f'{metric}_score'
        scores = []
        for entry in history:
            score_val = entry.get(metric_score_col)
            if score_val:
                try:
                    scores.append(float(score_val))
                except (ValueError, TypeError):
                    continue
        
        if not scores:
            return {
                'metric': metric,
                'test_suite': test_suite,
                'num_runs': 0,
                'latest_score': None,
                'average_score': None,
                'trend': 'NO_DATA'
            }
        
        latest_score = scores[0] if scores else None
        average_score = sum(scores) / len(scores) if scores else None
        
        # Determine trend (compare first to last)
        if len(scores) < 2:
            trend = 'INSUFFICIENT_DATA'
        elif scores[0] > scores[-1]:
            trend = 'IMPROVING'
        elif scores[0] < scores[-1]:
            trend = 'DECLINING'
        else:
            trend = 'STABLE'
        
        return {
            'metric': metric,
            'test_suite': test_suite,
            'num_runs': len(scores),
            'latest_score': latest_score,
            'average_score': round(average_score, 4) if average_score else None,
            'min_score': round(min(scores), 4) if scores else None,
            'max_score': round(max(scores), 4) if scores else None,
            'trend': trend
        }
    
    def generate_summary_report(self, output_file: Optional[Path] = None) -> Path:
        """
        Generate a summary report of all tracked metrics.
        
        Args:
            output_file: Optional path for output file (defaults to metrics_summary.json)
        
        Returns:
            Path to the generated report file
        """
        if output_file is None:
            output_file = self.reports_dir / "metrics_summary.json"
        
        # Get all unique test suites and metrics
        history = self.get_metric_history()
        
        if not history:
            summary = {
                'generated_at': datetime.now().isoformat(),
                'total_runs': 0,
                'test_suites': {}
            }
        else:
            test_suites = set(entry.get('test_suite') for entry in history if entry.get('test_suite'))
            
            summary = {
                'generated_at': datetime.now().isoformat(),
                'total_runs': len(set(entry.get('run_id') for entry in history if entry.get('run_id'))),
                'test_suites': {}
            }
            
            for test_suite in test_suites:
                # Get metrics from column names (columns ending with _score)
                suite_history = [h for h in history if h.get('test_suite') == test_suite]
                if not suite_history:
                    continue
                
                # Extract metric names from column names
                suite_metrics = set()
                for entry in suite_history:
                    for key in entry.keys():
                        if key.endswith('_score') and entry.get(key):
                            metric_name = key[:-6]  # Remove '_score' suffix
                            suite_metrics.add(metric_name)
                
                summary['test_suites'][test_suite] = {
                    'metrics': {},
                    'latest_scores': self.get_latest_scores(test_suite)
                }
                
                for metric in suite_metrics:
                    trend = self.get_trend_summary(test_suite, metric)
                    summary['test_suites'][test_suite]['metrics'][metric] = trend
        
        # Write summary to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary report generated: {output_file}")
        return output_file


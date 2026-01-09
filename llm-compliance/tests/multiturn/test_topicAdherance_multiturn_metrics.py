"""Test DeepEval multi-turn conversational metrics on conversational test cases."""

import json
import pytest
from pathlib import Path
import sys
import csv
from datetime import datetime

# DeepEval imports
from deepeval.test_case import ConversationalTestCase

# Add utils to path for metrics
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.deepEvalMultiTurnMetrics import (
    load_conversational_test_cases_from_json,
    verify_conversational_metric
)

# Metric name to threshold path mapping
METRIC_THRESHOLD_MAP = {
    "Topic Adherence": ("deepeval", "multiturn", "topic_adherence_min")
}

# Metric type mapping for verify_conversational_metric
METRIC_TYPE_MAP = {
    "Topic Adherence": "topic_adherence"
}


def get_threshold(thresholds, metric_name):
    """Get threshold value for a metric from nested thresholds structure."""
    if metric_name not in METRIC_THRESHOLD_MAP:
        return 0.7  # Default threshold
    
    path = METRIC_THRESHOLD_MAP[metric_name]
    value = thresholds
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return 0.7  # Default if path not found
    
    return value


def get_metric_model(thresholds, metric_name):
    """Get model for a metric from nested thresholds structure."""
    model_map = {
          "Topic Adherence": ("deepeval", "multiturn", "topic_adherence_model")
    }
    
    if metric_name not in model_map:
        return "gpt-4o"  # Default model
    
    path = model_map[metric_name]
    value = thresholds
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return "gpt-4o"  # Default if path not found
    
    return value


def test_multiturn_metrics(thresholds):
    """Test multi-turn conversational metrics on conversational test cases."""
    # Load conversational test cases
    # Default to RAG conversations, but can be configured
    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / "testData" / "synthetic_data" / "multiturn" / "conversationalTestCases" / "conversations_from_json_20251226_000744.json"
    
    if not json_path.exists():
        pytest.skip(f"Conversational test cases file not found: {json_path}")
    
    # Load test cases
    test_cases = load_conversational_test_cases_from_json(str(json_path))
    
    if not test_cases:
        pytest.skip("No conversational test cases found in file")
    
    # Add chatbot_role to test cases for RoleAdherenceMetric
    # Set appropriate role based on the context (RAG assistant for RAG conversations)
    # Updated role to better accommodate educational and technical content for students
    default_chatbot_role = "You are a helpful AI assistant for students in a school. You provide educational and informative responses about environmental science, geography, and related topics. You maintain a helpful, educational, and clear tone while explaining technical concepts in an accessible way for students. You cite sources when providing information from documents."
    
    
    for test_case in test_cases:
        if not hasattr(test_case, 'chatbot_role') or not test_case.chatbot_role:
            test_case.chatbot_role = default_chatbot_role
        # Set topics if not present (needed for TopicAdherenceMetric)
    # Setup logging directories
    evidence_dir = project_root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = evidence_dir / f"multiturn_metrics_test_log_{timestamp}.jsonl"
    
    # Collect all failures and results
    failures = []
    results = []
    
    # Test all metrics defined in METRIC_THRESHOLD_MAP
    metrics_to_test = list(METRIC_THRESHOLD_MAP.keys())
    
    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        
        # Get test case metadata
        scenario = getattr(test_case, 'scenario', None)
        expected_outcome = getattr(test_case, 'expected_outcome', None)
        user_description = getattr(test_case, 'user_description', None)
        
        # Initialize result entry
        result_entry = {
             "timestamp": datetime.now().isoformat(),
             "case_id": case_id,
             "scenario": scenario,
             "expected_outcome": expected_outcome,
             "user_description": user_description,
             "num_turns": len(test_case.turns),
             "chatbot_role": getattr(test_case, 'chatbot_role', None),
             "topics": getattr(test_case, 'topics', None) or thresholds.get('deepeval', {}).get('multiturn', {}).get('topic_adherence_topics', []),
             "turns": [
                 {"role": turn.role, "content": turn.content[:200] + "..." if len(turn.content) > 200 else turn.content}
                 for turn in test_case.turns
             ],
             "metrics": {},
             "errors": []
         }
        
        # Test each metric for this test case
        for metric_name in metrics_to_test:
            try:
                threshold = get_threshold(thresholds, metric_name)
                model = get_metric_model(thresholds, metric_name)
                metric_type = METRIC_TYPE_MAP[metric_name]
                
                # Verify the metric
                score, reason, passed, error_msg = verify_conversational_metric(
                    test_case=test_case,
                    metric_type=metric_type,
                    threshold=threshold,
                    model=model,
                    include_reason=True,
                    case_id=case_id
                )
                
                if error_msg:
                    # Error occurred during evaluation
                    result_entry["errors"].append(f"{metric_name}: {error_msg}")
                    result_entry["metrics"][metric_name] = {
                        "score": None,
                        "threshold": threshold,
                        "model": model,
                        "passed": False,
                        "error": error_msg
                    }
                    failures.append(f"{metric_name} error for {case_id}: {error_msg}")
                else:
                    # Store metric result
                    # Handle None scores - if score is None, treat as error
                    if score is None:
                        error_msg = f"{metric_name} returned None score for {case_id}"
                        result_entry["errors"].append(error_msg)
                        result_entry["metrics"][metric_name] = {
                            "score": None,
                            "threshold": threshold,
                            "model": model,
                            "passed": False,
                            "error": "Metric returned None score"
                        }
                        failures.append(error_msg)
                    else:
                        metric_result = {
                            "score": round(score, 4),
                            "threshold": threshold,
                            "model": model,
                            "passed": passed,
                            "reason": reason
                        }
                        result_entry["metrics"][metric_name] = metric_result
                        
                        # Check if metric passed
                        if not passed:
                            # Include reason in failure message for debugging, especially for low scores
                            failure_msg = f"{metric_name} score={score:.4f} < threshold {threshold} for {case_id}"
                            if reason:
                                # Truncate reason if too long
                                reason_preview = reason[:150] + "..." if len(reason) > 150 else reason
                                failure_msg += f" (reason: {reason_preview})"
                            failures.append(failure_msg)
                            
                            # Log very low scores for debugging
                            if score == 0.0:
                                print(f"⚠️  Warning: {metric_name} scored 0.0 for {case_id}. Reason: {reason[:200] if reason else 'No reason provided'}")
                
            except Exception as e:
                error_msg = f"Error evaluating {metric_name} for {case_id}: {e}"
                result_entry["errors"].append(error_msg)
                result_entry["metrics"][metric_name] = {
                    "score": None,
                    "threshold": get_threshold(thresholds, metric_name),
                    "model": get_metric_model(thresholds, metric_name),
                    "passed": False,
                    "error": str(e)
                }
                failures.append(error_msg)
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry) + "\n")
    
    # Export results to CSV
    if results:
        csv_file = reports_dir / f"multiturn_metrics_test_results_{timestamp}.csv"
        
        # Flatten results for CSV (one row per test case, metrics as columns)
        csv_rows = []
        for entry in results:
            row = {
                "case_id": entry.get("case_id"),
                "scenario": entry.get("scenario"),
                "expected_outcome": entry.get("expected_outcome"),
                "user_description": entry.get("user_description"),
                "num_turns": entry.get("num_turns"),
                "timestamp": entry.get("timestamp"),
            }
            
            # Add metric scores as columns
            metrics = entry.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                row[f"{metric_name}_score"] = metric_data.get("score")
                row[f"{metric_name}_threshold"] = metric_data.get("threshold")
                row[f"{metric_name}_model"] = metric_data.get("model")
                row[f"{metric_name}_passed"] = metric_data.get("passed")
                row[f"{metric_name}_reason"] = metric_data.get("reason")
                # Also include error if present
                if "error" in metric_data:
                    row[f"{metric_name}_error"] = metric_data.get("error")
            
            # Add errors if any
            errors = entry.get("errors", [])
            row["errors"] = json.dumps(errors) if errors else None
            
            csv_rows.append(row)
        
        # Get all unique fieldnames
        all_fieldnames = set()
        for row in csv_rows:
            all_fieldnames.update(row.keys())
        
        # Define leftmost columns in desired order
        leftmost_columns = ["case_id", "scenario", "expected_outcome", "user_description", "num_turns", "errors", "timestamp"]
        
        # Get remaining columns (metric columns) and sort them
        remaining_columns = sorted([f for f in all_fieldnames if f not in leftmost_columns])
        
        # Combine: leftmost columns first, then remaining columns
        fieldnames = [col for col in leftmost_columns if col in all_fieldnames] + remaining_columns
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                # Ensure all fields are present
                complete_row = {field: row.get(field, None) for field in fieldnames}
                writer.writerow(complete_row)
    
    # Log summary before any assertions
    total_test_cases = len(test_cases)
    total_metrics_tested = total_test_cases * len(metrics_to_test)
    total_failures = len(failures)
    
    print(f"\n📊 Multi-Turn Conversational Metrics Test Summary:")
    print(f"  Total test cases: {total_test_cases}")
    print(f"  Total metrics tested: {total_metrics_tested}")
    print(f"  Total failures: {total_failures}")
    print(f"  Log file: {log_file}")
    if results:
        print(f"  CSV report: {csv_file}")
    
    # Print per-metric summary
    print(f"\n📈 Per-Metric Summary:")
    for metric_name in metrics_to_test:
        metric_passed = sum(
            1 for entry in results
            if entry.get("metrics", {}).get(metric_name, {}).get("passed", False)
        )
        metric_total = sum(
            1 for entry in results
            if metric_name in entry.get("metrics", {})
        )
        if metric_total > 0:
            pass_rate = (metric_passed / metric_total) * 100
            print(f"  {metric_name}: {metric_passed}/{metric_total} passed ({pass_rate:.1f}%)")
    
    # Assert all failures at the end
    if failures:
        failure_message = (
            f"\n\nFound {len(failures)} failure(s):\n" + 
            "\n".join(f"  - {f}" for f in failures[:50])  # Show first 50 failures
        )
        if len(failures) > 50:
            failure_message += f"\n  ... and {len(failures) - 50} more failures"
        pytest.fail(failure_message)


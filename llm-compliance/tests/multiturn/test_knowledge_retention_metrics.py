"""Test DeepEval Knowledge Retention metric on conversational test cases."""

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
    verify_knowledge_retention_metric,
    evaluate_knowledge_retention_test_cases
)


def get_threshold(thresholds, metric_name="Knowledge Retention"):
    """Get threshold value for Knowledge Retention metric from nested thresholds structure."""
    path = ("deepeval", "multiturn", "knowledge_retention_min")
    value = thresholds
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return 0.7  # Default if path not found
    
    return value


def get_metric_model(thresholds, metric_name="Knowledge Retention"):
    """Get model for Knowledge Retention metric from nested thresholds structure."""
    path = ("deepeval", "multiturn", "knowledge_retention_model")
    value = thresholds
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return "gpt-4o"  # Default if path not found
    
    return value


def test_knowledge_retention_metric(thresholds):
    """Test Knowledge Retention metric on conversational test cases."""
    # Load conversational test cases
    # Default to RAG conversations, but can be configured
    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / "testData" / "synthetic_data" / "multiturn" / "conversationalTestCases" / "conversations_from_json_20251225_234506.json"
    
    if not json_path.exists():
        pytest.skip(f"Conversational test cases file not found: {json_path}")
    
    # Load test cases
    test_cases = load_conversational_test_cases_from_json(str(json_path))
    
    if not test_cases:
        pytest.skip("No conversational test cases found in file")
    
    # Setup logging directories
    evidence_dir = project_root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = evidence_dir / f"knowledge_retention_metric_test_log_{timestamp}.jsonl"
    
    # Collect all failures and results
    failures = []
    results = []
    
    # Get threshold and model from config
    threshold = get_threshold(thresholds)
    model = get_metric_model(thresholds)
    metric_name = "Knowledge Retention"
    
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
            "turns": [
                {"role": turn.role, "content": turn.content[:200] + "..." if len(turn.content) > 200 else turn.content}
                for turn in test_case.turns
            ],
            "metric": metric_name,
            "score": None,
            "threshold": threshold,
            "model": model,
            "passed": False,
            "reason": None,
            "error": None
        }
        
        # Test Knowledge Retention metric for this test case
        try:
            # Verify the metric
            score, reason, passed, error_msg = verify_knowledge_retention_metric(
                test_case=test_case,
                threshold=threshold,
                model=model,
                include_reason=True,
                case_id=case_id
            )
            
            if error_msg:
                # Error occurred during evaluation
                result_entry["error"] = error_msg
                result_entry["passed"] = False
                failures.append(f"{metric_name} error for {case_id}: {error_msg}")
            else:
                # Store metric result
                # Handle None scores - if score is None, treat as error
                if score is None:
                    error_msg = f"{metric_name} returned None score for {case_id}"
                    result_entry["error"] = "Metric returned None score"
                    result_entry["passed"] = False
                    failures.append(error_msg)
                else:
                    result_entry["score"] = round(score, 4)
                    result_entry["reason"] = reason
                    result_entry["passed"] = passed
                    
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
            result_entry["error"] = str(e)
            result_entry["passed"] = False
            failures.append(error_msg)
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry) + "\n")
    
    # Export results to CSV
    if results:
        csv_file = reports_dir / f"knowledge_retention_metric_test_results_{timestamp}.csv"
        
        # Flatten results for CSV
        csv_rows = []
        for entry in results:
            row = {
                "case_id": entry.get("case_id"),
                "scenario": entry.get("scenario"),
                "expected_outcome": entry.get("expected_outcome"),
                "user_description": entry.get("user_description"),
                "num_turns": entry.get("num_turns"),
                "metric": entry.get("metric"),
                "score": entry.get("score"),
                "threshold": entry.get("threshold"),
                "model": entry.get("model"),
                "passed": entry.get("passed"),
                "reason": entry.get("reason"),
                "error": entry.get("error"),
                "timestamp": entry.get("timestamp"),
            }
            csv_rows.append(row)
        
        # Define fieldnames
        fieldnames = [
            "case_id", "scenario", "expected_outcome", "user_description", 
            "num_turns", "metric", "score", "threshold", "model", 
            "passed", "reason", "error", "timestamp"
        ]
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                # Ensure all fields are present
                complete_row = {field: row.get(field, None) for field in fieldnames}
                writer.writerow(complete_row)
    
    # Log summary before any assertions
    total_test_cases = len(test_cases)
    total_passed = sum(1 for r in results if r.get("passed", False))
    total_failures = len(failures)
    
    print(f"\n📊 Knowledge Retention Metric Test Summary:")
    print(f"  Total test cases: {total_test_cases}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failures}")
    print(f"  Pass rate: {(total_passed / total_test_cases * 100) if total_test_cases > 0 else 0:.1f}%")
    print(f"  Threshold: {threshold}")
    print(f"  Model: {model}")
    print(f"  Log file: {log_file}")
    if results:
        print(f"  CSV report: {csv_file}")
    
    # Assert all failures at the end
    if failures:
        failure_message = (
            f"\n\nFound {len(failures)} failure(s):\n" + 
            "\n".join(f"  - {f}" for f in failures[:50])  # Show first 50 failures
        )
        if len(failures) > 50:
            failure_message += f"\n  ... and {len(failures) - 50} more failures"
        pytest.fail(failure_message)


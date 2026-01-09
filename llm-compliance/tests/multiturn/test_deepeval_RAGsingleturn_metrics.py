"""Test DeepEval RAG-specific single-turn metrics on single-turn golden test cases.

This test file focuses on RAG-specific metrics that require retrieval context,
such as AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric,
ContextualRecallMetric, and ContextualRelevancyMetric, applied to single-turn Q&A pairs.
"""

import json
import pytest
from pathlib import Path
import sys
import csv
from datetime import datetime

# DeepEval imports
from deepeval.test_case import LLMTestCase

# Add utils to path for metrics
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.deepEvalRAGSingleturnMetrics import (
    load_singleturn_goldens_from_json,
    verify_answer_relevancy_metric,
    verify_faithfulness_metric,
    verify_contextual_precision_metric,
    verify_contextual_recall_metric,
    verify_contextual_relevancy_metric
)

# Metric name to threshold path mapping (RAG-specific metrics only)
METRIC_THRESHOLD_MAP = {
    "Answer Relevancy": ("deepeval", "singleturn_rag", "answer_relevancy_min"),
    "Faithfulness": ("deepeval", "singleturn_rag", "faithfulness_min"),
    "Contextual Precision": ("deepeval", "singleturn_rag", "contextual_precision_min"),
    "Contextual Recall": ("deepeval", "singleturn_rag", "contextual_recall_min"),
    "Contextual Relevancy": ("deepeval", "singleturn_rag", "contextual_relevancy_min"),
}

# Metric name to verification function mapping
METRIC_VERIFY_FUNC_MAP = {
    "Answer Relevancy": verify_answer_relevancy_metric,
    "Faithfulness": verify_faithfulness_metric,
    "Contextual Precision": verify_contextual_precision_metric,
    "Contextual Recall": verify_contextual_recall_metric,
    "Contextual Relevancy": verify_contextual_relevancy_metric,
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
        "Answer Relevancy": ("deepeval", "singleturn_rag", "answer_relevancy_model"),
        "Faithfulness": ("deepeval", "singleturn_rag", "faithfulness_model"),
        "Contextual Precision": ("deepeval", "singleturn_rag", "contextual_precision_model"),
        "Contextual Recall": ("deepeval", "singleturn_rag", "contextual_recall_model"),
        "Contextual Relevancy": ("deepeval", "singleturn_rag", "contextual_relevancy_model"),
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


def get_metric_mean_threshold(thresholds, metric_name):
    """Get mean threshold value for a metric from nested thresholds structure."""
    mean_map = {
        "Answer Relevancy": ("deepeval", "singleturn_rag", "answer_relevancy_mean"),
        "Faithfulness": ("deepeval", "singleturn_rag", "faithfulness_mean"),
        "Contextual Precision": ("deepeval", "singleturn_rag", "contextual_precision_mean"),
        "Contextual Recall": ("deepeval", "singleturn_rag", "contextual_recall_mean"),
        "Contextual Relevancy": ("deepeval", "singleturn_rag", "contextual_relevancy_mean"),
    }
    
    if metric_name not in mean_map:
        return 0.7  # Default mean threshold
    
    path = mean_map[metric_name]
    value = thresholds
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return 0.7  # Default if path not found
    
    return value


def test_rag_singleturn_metrics(thresholds):
    """Test RAG-specific single-turn metrics on single-turn golden test cases.
    
    This test focuses on RAG metrics that require retrieval_context:
    - AnswerRelevancyMetric: Evaluates how relevant the response is to the query
    - FaithfulnessMetric: Evaluates factual accuracy grounded in retrieval context
    - ContextualPrecisionMetric: Evaluates precision of retrieval context usage
    - ContextualRecallMetric: Evaluates recall of relevant information from context
    - ContextualRelevancyMetric: Evaluates relevancy of retrieval context
    
    Single-turn goldens are converted to LLMTestCase format with:
    - input: user query
    - actual_output: actual_output if available, otherwise expected_output
    - expected_output: expected_output (for metrics that need it)
    - retrieval_context: context (for RAG metrics)
    """
    # Load single-turn golden test cases
    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / "testData" / "synthetic_data" / "singleturn" / "singleturnGoldens" / "20251225_232614.json"
    
    if not json_path.exists():
        pytest.skip(f"Single-turn goldens file not found: {json_path}")
    
    # Load test cases (converts single-turn goldens to LLMTestCase format)
    test_cases, metadata_list = load_singleturn_goldens_from_json(str(json_path))
    
    if not test_cases:
        pytest.skip("No test cases found in file")
    
    # Verify that test cases have retrieval_context for RAG metrics
    for i, test_case in enumerate(test_cases, 1):
        if not hasattr(test_case, 'retrieval_context') or not test_case.retrieval_context:
            pytest.skip(f"Test case {i} missing retrieval_context")
    
    # Setup logging directories
    evidence_dir = project_root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = evidence_dir / f"singleturn_RAG_metrics_test_log_{timestamp}.jsonl"
    
    # Collect all failures and results
    failures = []
    results = []
    
    # Test all metrics defined in METRIC_THRESHOLD_MAP
    metrics_to_test = list(METRIC_THRESHOLD_MAP.keys())
    
    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        case_id = f"case_{i}"
        
        # Get test case metadata
        input_query = getattr(test_case, 'input', '')
        actual_output = getattr(test_case, 'actual_output', '')
        expected_output = getattr(test_case, 'expected_output', None)
        retrieval_context = getattr(test_case, 'retrieval_context', [])
        
        # Get source_file from metadata (LLMTestCase doesn't support source_file attribute)
        metadata = metadata_list[i - 1] if i - 1 < len(metadata_list) else {}
        source_file = metadata.get('source_file', None)
        
        # Initialize result entry
        result_entry = {
             "timestamp": datetime.now().isoformat(),
             "case_id": case_id,
             "input": input_query,
             "actual_output": actual_output,
             "expected_output": expected_output,
             "source_file": source_file,
             "has_retrieval_context": bool(retrieval_context),
             "retrieval_context_count": len(retrieval_context) if isinstance(retrieval_context, list) else (1 if retrieval_context else 0),
             "metrics": {},
             "errors": []
         }
        
        # Test each metric for this test case
        for metric_name in metrics_to_test:
            try:
                threshold = get_threshold(thresholds, metric_name)
                model = get_metric_model(thresholds, metric_name)
                verify_func = METRIC_VERIFY_FUNC_MAP[metric_name]
                
                # Verify the metric using the specific verification function
                score, reason, passed, error_msg = verify_func(
                    test_case=test_case,
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
                        "reason": None,  # Always include reason field (even if None)
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
                            "reason": None,  # Always include reason field (even if None)
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
                    "reason": None,  # Always include reason field (even if None)
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
        csv_file = reports_dir / f"singleturn_RAG_metrics_test_results_{timestamp}.csv"
        
        # Flatten results for CSV (one row per test case, metrics as columns)
        csv_rows = []
        for entry in results:
            row = {
                "case_id": entry.get("case_id"),
                "input": entry.get("input"),
                "actual_output": entry.get("actual_output"),
                "expected_output": entry.get("expected_output"),
                "source_file": entry.get("source_file"),
                "has_retrieval_context": entry.get("has_retrieval_context"),
                "retrieval_context_count": entry.get("retrieval_context_count"),
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
        leftmost_columns = ["case_id", "input", "actual_output", "expected_output", "source_file", "has_retrieval_context", "retrieval_context_count", "errors", "timestamp"]
        
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
    
    print(f"\n📊 RAG Single-Turn Metrics Test Summary:")
    print(f"  Total test cases: {total_test_cases}")
    print(f"  Total metrics tested: {total_metrics_tested}")
    print(f"  Total failures: {total_failures}")
    print(f"  Log file: {log_file}")
    if results:
        print(f"  CSV report: {csv_file}")
    
    # Calculate per-metric averages and compare with mean thresholds
    print(f"\n📈 Per-Metric Summary:")
    metric_averages = {}
    metric_mean_failures = []
    
    for metric_name in metrics_to_test:
        metric_passed = sum(
            1 for entry in results
            if entry.get("metrics", {}).get(metric_name, {}).get("passed", False)
        )
        metric_total = sum(
            1 for entry in results
            if metric_name in entry.get("metrics", {})
        )
        
        # Calculate average score for this metric
        metric_scores = []
        for entry in results:
            metric_data = entry.get("metrics", {}).get(metric_name, {})
            score = metric_data.get("score")
            if score is not None:
                metric_scores.append(score)
        
        if metric_total > 0:
            pass_rate = (metric_passed / metric_total) * 100
            avg_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
            
            # Get mean threshold for this metric
            mean_threshold = get_metric_mean_threshold(thresholds, metric_name)
            
            # Check if average meets mean threshold
            meets_threshold = avg_score >= mean_threshold
            status = "PASS" if meets_threshold else "FAIL"
            
            metric_averages[metric_name] = {
                "average": avg_score,
                "mean_threshold": mean_threshold,
                "passed_count": metric_passed,
                "total_count": metric_total,
                "pass_rate": pass_rate,
                "status": status,
                "meets_threshold": meets_threshold
            }
            
            print(f"  {metric_name}:")
            print(f"    Passed: {metric_passed}/{metric_total} ({pass_rate:.1f}%)")
            print(f"    Average Score: {avg_score:.4f}")
            print(f"    Mean Threshold: {mean_threshold:.4f}")
            print(f"    Status: {status}")
            
            # Check if average meets mean threshold
            if not meets_threshold:
                failure_msg = (
                    f"{metric_name} average score {avg_score:.4f} is below mean threshold {mean_threshold:.4f}"
                )
                metric_mean_failures.append(failure_msg)
                print(f"    ⚠️  FAILED: {failure_msg}")
            else:
                print(f"    ✅ PASSED: Average meets mean threshold")
    
    # Export metric-level summary to CSV
    if metric_averages:
        summary_csv_file = reports_dir / f"rag_singleturn_metrics_summary_{timestamp}.csv"
        summary_fieldnames = ["metric", "number_of_tests", "mean", "mean_threshold", "status"]
        
        with open(summary_csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            for metric_name, metric_data in metric_averages.items():
                writer.writerow({
                    "metric": metric_name,
                    "number_of_tests": metric_data["total_count"],
                    "mean": round(metric_data["average"], 4),
                    "mean_threshold": round(metric_data["mean_threshold"], 4),
                    "status": metric_data["status"]
                })
        
        print(f"\n📄 Metric Summary CSV: {summary_csv_file}")
    
    # Assert on mean threshold failures at metric level
    if metric_mean_failures:
        failure_summary = (
            f"\n\nFound {len(metric_mean_failures)} metric(s) with average scores below mean threshold:\n" +
            "\n".join(f"  - {f}" for f in metric_mean_failures)
        )
        pytest.fail(failure_summary)
    
    # Assert individual test case failures at the end (if any)
    if failures:
        failure_message = (
            f"\n\nFound {len(failures)} individual test case failure(s):\n" + 
            "\n".join(f"  - {f}" for f in failures[:50])  # Show first 50 failures
        )
        if len(failures) > 50:
            failure_message += f"\n  ... and {len(failures) - 50} more failures"
        pytest.fail(failure_message)

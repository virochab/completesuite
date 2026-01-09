"""Test DeepEval safety metrics on queries from safety_queries.jsonl."""

import json
import pytest
from pathlib import Path
import sys
import csv
from datetime import datetime

# DeepEval imports
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Add utils to path for metrics
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.deepEvalMetrics import (
    verify_toxicity_metric,
    verify_non_advice_metric,
    verify_misuse_metric,
    verify_role_violation_metric,
    verify_bias_metric
)

# Metric name to threshold path mapping
# References thresholds.yaml structure: deepeval.safety.* (lines 30-34)
METRIC_THRESHOLD_MAP = {
    # Safety metrics (deepeval.safety.*) - thresholds.yaml lines 30-34
    "Toxicity": ("deepeval", "safety", "toxicity_max"),  # thresholds.yaml line 30
    "Non-Advice": ("deepeval", "safety", "non_advice_max"),  # thresholds.yaml line 32
    "Misuse": ("deepeval", "safety", "misuse_max"),  # thresholds.yaml line 33
    "Role Violation": ("deepeval", "safety", "role_violation_max"),  # thresholds.yaml line 34
}

# GEval criteria for supporting metrics (safety metrics use DeepEval built-in methods)
METRIC_CRITERIA = {
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


def is_lower_better(metric_name):
    """Check if lower scores are better for a metric (all safety metrics are max thresholds)."""
    return metric_name in [
        "Toxicity", "Non-Advice", "Misuse", "Role Violation", "Bias"
    ]


def test_safety_metrics(client, thresholds):
    """Test safety metrics on queries from category-specific query files."""
    # Load safety queries from all category files
    safety_data_dir = Path(__file__).parent.parent.parent / "testData" / "safety"
    
    # List of category-specific query files
    query_files = [
        "toxicity_queries.jsonl",
        "non_advice_queries.jsonl",
        "misuse_queries.jsonl",
        "role_violation_queries.jsonl",
    ]
    
    # Load test cases from all category files
    queries = []
    for query_file in query_files:
        queries_path = safety_data_dir / query_file
        if queries_path.exists():
            with open(queries_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            case = json.loads(line)
                            queries.append(case)
                        except json.JSONDecodeError:
                            continue
    
    if not queries:
        pytest.skip("No safety queries found in category files")
    
    # Check if client is implemented
    if not hasattr(client, 'ask'):
        pytest.skip("Client not implemented: missing 'ask' method")
    
    # Setup logging directories
    evidence_dir = Path(__file__).parent.parent.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = evidence_dir / f"safety_test_log_{timestamp}.jsonl"
    
    # Collect all failures and results
    failures = []
    results = []
    
    # Process each query
    for query_case in queries:
        case_id = query_case.get("id", "unknown")
        query = query_case.get("query", "")
        category = query_case.get("category", "unknown")
        # Always test all metrics in METRIC_THRESHOLD_MAP
        metrics_to_test = list(METRIC_THRESHOLD_MAP.keys())
        
        if not query:
            continue
        
        if not metrics_to_test:
            continue
        
        # Initialize result entry
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "case_id": case_id,
            "category": category,
            "query": query,
            "response": None,
            "metrics": {},
            "errors": []
        }
        
        try:
            # Get response from client
            response = client.ask(query)
            response_text = response.text if hasattr(response, 'text') else str(response)
            result_entry["response"] = response_text
            
        except (AttributeError, TypeError) as e:
            # Client implementation issue
            result_entry["errors"].append(f"Client error: {str(e)}")
            failures.append(f"Client error for case {case_id}: {e}")
            results.append(result_entry)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry) + "\n")
            continue
        except Exception as e:
            # Other errors (API errors, etc.)
            result_entry["errors"].append(f"Error: {str(e)}")
            failures.append(f"Failed to get response for case {case_id}: {e}")
            results.append(result_entry)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry) + "\n")
            continue
        
        # Test each metric for this query
        for metric_name in metrics_to_test:
            try:
                threshold = get_threshold(thresholds, metric_name)
                lower_is_better = is_lower_better(metric_name)
                
                # Use DeepEval built-in methods for safety metrics
                if metric_name == "Toxicity":
                    score, reason, passed, error_msg = verify_toxicity_metric(
                        actual_output=response_text,
                        input_query=query,
                        threshold=threshold,
                        model='gpt-4o',
                        include_reason=True,
                        case_id=case_id
                    )
                    if error_msg:
                        raise Exception(error_msg)
                elif metric_name == "Non-Advice":
                    score, reason, passed, error_msg = verify_non_advice_metric(
                        actual_output=response_text,
                        input_query=query,
                        threshold=threshold,
                        advice_types=["financial", "medical", "legal"],
                        model='gpt-4o',
                        include_reason=True,
                        case_id=case_id
                    )
                    if error_msg:
                        raise Exception(error_msg)
                elif metric_name == "Misuse":
                    score, reason, passed, error_msg = verify_misuse_metric(
                        actual_output=response_text,
                        input_query=query,
                        threshold=threshold,
                        domain="financial",
                        model='gpt-4o',
                        include_reason=True,
                        case_id=case_id
                    )
                    if error_msg:
                        raise Exception(error_msg)
                elif metric_name == "Role Violation":
                    score, reason, passed, error_msg = verify_role_violation_metric(
                        actual_output=response_text,
                        input_query=query,
                        threshold=threshold,
                        role="AI assistant for internal support employees at a home insurance company",
                        model='gpt-4o',
                        include_reason=True,
                        case_id=case_id
                    )
                    if error_msg:
                        raise Exception(error_msg)
                elif metric_name == "Bias":
                    score, reason, passed, error_msg = verify_bias_metric(
                        actual_output=response_text,
                        input_query=query,
                        threshold=threshold,
                        model='gpt-4o',
                        include_reason=True,
                        case_id=case_id
                    )
                    if error_msg:
                        raise Exception(error_msg)
                else:
                    # Use GEval for supporting metrics (Privacy, Security, etc.)
                    criteria = METRIC_CRITERIA.get(metric_name, f"{metric_name} - evaluate the quality of the response.")
                    test_case = LLMTestCase(
                        actual_output=response_text,
                        input=query
                    )
                    metric = GEval(
                        name=metric_name,
                        criteria=criteria,
                        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                        threshold=threshold,
                        model='gpt-4o'
                    )
                    metric.measure(test_case)
                    score = metric.score if metric.score is not None else 0.0
                    reason = getattr(metric, 'reason', None)
                    passed = score <= threshold if lower_is_better else score >= threshold
                
                # Store metric result
                metric_result = {
                    "score": round(score, 4) if score is not None else None,
                    "threshold": threshold,
                    "passed": passed,
                    "reason": reason
                }
                result_entry["metrics"][metric_name] = metric_result
                
                # Check if metric passed
                if not passed:
                    failures.append(
                        f"{metric_name} score={score:.4f} {'>' if lower_is_better else '<'} threshold {threshold} "
                        f"for case {case_id}"
                    )
                
            except Exception as e:
                error_msg = f"Error evaluating {metric_name} for case {case_id}: {e}"
                result_entry["errors"].append(error_msg)
                result_entry["metrics"][metric_name] = {
                    "score": None,
                    "threshold": get_threshold(thresholds, metric_name),
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
        csv_file = reports_dir / f"safety_test_results_{timestamp}.csv"
        
        # Flatten results for CSV (one row per query, metrics as columns)
        csv_rows = []
        for entry in results:
            row = {
                "timestamp": entry.get("timestamp"),
                "case_id": entry.get("case_id"),
                "category": entry.get("category"),
                "query": entry.get("query"),
                "response": entry.get("response"),
            }
            
            # Add metric scores as columns
            metrics = entry.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                row[f"{metric_name}_score"] = metric_data.get("score")
                row[f"{metric_name}_threshold"] = metric_data.get("threshold")
                row[f"{metric_name}_passed"] = metric_data.get("passed")
            
            # Add errors if any
            errors = entry.get("errors", [])
            row["errors"] = json.dumps(errors) if errors else None
            
            csv_rows.append(row)
        
        # Get all unique fieldnames
        all_fieldnames = set()
        for row in csv_rows:
            all_fieldnames.update(row.keys())
        
        # Define leftmost columns in desired order
        leftmost_columns = ["case_id", "category", "errors", "query", "response", "timestamp"]
        
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
    total_queries = len(queries)
    total_metrics_tested = sum(len(q.get("metrics", [])) for q in queries)
    total_failures = len(failures)
    
    print(f"\n📊 Safety Metrics Test Summary:")
    print(f"  Total queries: {total_queries}")
    print(f"  Total metrics tested: {total_metrics_tested}")
    print(f"  Total failures: {total_failures}")
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


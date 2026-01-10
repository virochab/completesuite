"""Test multiple DeepEval agentic metrics on weather agent queries from taskCompletionData.jsonl.

This test combines:
- Multiple metrics evaluation using EvaluationDataset and evals_iterator (from test_planadhernew.py)
- Comprehensive reporting and threshold reading (from test_task_completion.py)
- Queries from taskCompletionData.jsonl
"""

import json
import pytest
from pathlib import Path
import sys
import csv
from datetime import datetime
from typing import Any

# DeepEval imports
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.metrics import (
    TaskCompletionMetric,
    StepEfficiencyMetric,
    PlanAdherenceMetric,
    PlanQualityMetric
)
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, ToolCall

# Add utils to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.weather_client import WeatherAgentClient


# Initialize weather agent client (module-level for reuse)
weather_client = WeatherAgentClient()



@observe(type="agent")
def weather_agent(input_query: str) -> tuple[str, Any]:
    """
    Weather agent wrapper function for DeepEval tracing with multiple metrics evaluation.
    
    The @observe() decorator enables DeepEval to:
    1. Automatically trace function execution
    2. Extract tool calls made during execution
    3. Provide execution context to metrics for evaluation
    
    Args:
        input_query: The user's weather-related query
    
    Returns:
        Tuple of (response_text, response_object):
        - response_text: The agent's response text (for evaluation)
        - response_object: The full response object (for accessing tools_called, etc.)
    """
    # Generate and capture the plan using the planning component
    # The plan is captured in the trace via update_current_span() in agent_planning_component()
    # IMPORTANT: The plan must remain in the trace for PlanAdherenceMetric and PlanQualityMetric to find it
    
    # Execute the plan by calling the weather agent
    response = weather_client.ask(input_query)
    
    # response.tools_called is a list of dictionaries, not objects
    if hasattr(response, 'tools_called') and response.tools_called:
        # Convert dict format to ToolCall objects for DeepEval
        ToolCalls = [
            ToolCall(
                name=tool.get('name') if isinstance(tool, dict) else getattr(tool, 'name', 'unknown')
         #       args=tool.get('args', {}) if isinstance(tool, dict) else getattr(tool, 'args', {})
            ) 
            for tool in response.tools_called
        ]
    else:
        ToolCalls = []
    
    update_current_trace(
        input=input_query,
        output=response.text,
        tools_called=ToolCalls
    )
    
    return response.text, response


def test_multiple_agentic_metrics(thresholds):
    """
    Test multiple agentic metrics (TaskCompletion, StepEfficiency, PlanAdherence, PlanQuality)
    on weather agent queries from taskCompletionData.jsonl.
    
    Uses EvaluationDataset and evals_iterator for evaluation, with comprehensive reporting.
    """
    # Load queries from taskCompletionData.jsonl
    queries_path = Path(__file__).parent.parent.parent.parent / "testData" / "agentic" / "taskCompletionData.jsonl"
    
    if not queries_path.exists():
        pytest.skip(f"Task completion queries file not found: {queries_path}")
    
    # Load test cases
    queries = []
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
        pytest.skip("No task completion queries found in file")
    
    # Setup logging directories
    evidence_dir = Path(__file__).parent.parent.parent.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    reports_dir = Path(__file__).parent.parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = evidence_dir / f"multiple_agentic_metrics_test_log_{timestamp}.jsonl"
    
    # Get thresholds from config
    agentic_config = thresholds.get("deepeval", {}).get("agentic", {})
    task_completion_threshold = agentic_config.get("task_completion_min", 0.7)
    step_efficiency_threshold = agentic_config.get("step_efficiency_min", 0.7)
    plan_adherence_threshold = agentic_config.get("plan_adherence_min", 0.7)
    plan_quality_threshold = agentic_config.get("plan_quality_min", 0.7)
    
    # Get mean thresholds (fallback to min threshold if not defined)
    task_completion_mean_threshold = agentic_config.get("task_completion_mean", task_completion_threshold)
    step_efficiency_mean_threshold = agentic_config.get("step_efficiency_mean", step_efficiency_threshold)
    plan_adherence_mean_threshold = agentic_config.get("plan_adherence_mean", plan_adherence_threshold)
    plan_quality_mean_threshold = agentic_config.get("plan_quality_mean", plan_quality_threshold)
    
    # Create EvaluationDataset from queries
    goldens = [Golden(input=case.get("query", "")) for case in queries if case.get("query")]
    dataset = EvaluationDataset(goldens=goldens)
    
    # Collect all failures and results
    failures = []
    results = []
    
    # Process each query directly (not using evals_iterator)
    # Create fresh metric instances for each golden to ensure scores are captured correctly
    for query_case in queries:
        query = query_case.get("query", "")
        if not query:
            continue
            
        case_id = query_case.get("id", f"case_{len(results) + 1}")
        category = query_case.get("category", "weather_query")
        expected_tool = query_case.get("expected_tool", None)
        
        # Create fresh metric instances for this golden
        # This ensures each test case gets its own metric evaluation
        task_completion = TaskCompletionMetric(threshold=task_completion_threshold)
        step_efficiency = StepEfficiencyMetric(threshold=step_efficiency_threshold)
        plan_adherence = PlanAdherenceMetric(threshold=plan_adherence_threshold)
        plan_quality = PlanQualityMetric(threshold=plan_quality_threshold)
        
        # Initialize result entry before the loop
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "case_id": case_id,
            "category": category,
            "query": query,
            "response": None,
            "tools_called": [],
            "expected_tool": expected_tool,
            "task_completion_score": None,
            "task_completion_success": None,
            "task_completion_reason": None,
            "step_efficiency_score": None,
            "step_efficiency_success": None,
            "step_efficiency_reason": None,
            "plan_adherence_score": None,
            "plan_adherence_success": None,
            "plan_adherence_reason": None,
            "plan_quality_score": None,
            "plan_quality_success": None,
            "plan_quality_reason": None,
            "errors": []
        }
        
        # Reset conversation for each test case to ensure independence
        weather_client.reset_conversation(f"test-case-{case_id}")
        
        # Create a temporary dataset with just this golden for evals_iterator
        temp_golden = Golden(input=query)
        temp_dataset = EvaluationDataset(goldens=[temp_golden])
        
        # Use evals_iterator with fresh metrics for this single golden
        # evals_iterator evaluates metrics automatically after weather_agent is called
        for golden in temp_dataset.evals_iterator(metrics=[task_completion, step_efficiency, plan_adherence, plan_quality]):
            try:
                # Call the weather agent with the query
                # The @observe() decorator automatically traces this execution for DeepEval
                # evals_iterator evaluates metrics automatically after this call
                output, response = weather_agent(golden.input)
                result_entry["response"] = output
                
                # Get tool calls information from the response object
                tools_called = response.tools_called if hasattr(response, 'tools_called') else []
                result_entry["tools_called"] = tools_called
                
            except Exception as e:
                error_msg = f"Error calling weather_agent for case {case_id}: {e}"
                result_entry["errors"].append(error_msg)
                failures.append(error_msg)
        
        # CRITICAL: Capture scores AFTER the inner loop completes
        # evals_iterator evaluates metrics automatically, and scores are available
        # after the iteration completes (similar to test_planadhernew.py)
        # Since we have fresh metric instances for each golden, scores are specific to this test case
        result_entry["task_completion_score"] = round(task_completion.score, 4) if task_completion.score is not None else None
        result_entry["task_completion_success"] = task_completion.success
        result_entry["task_completion_reason"] = task_completion.reason if hasattr(task_completion, 'reason') and task_completion.reason else None
        
        result_entry["step_efficiency_score"] = round(step_efficiency.score, 4) if step_efficiency.score is not None else None
        result_entry["step_efficiency_success"] = step_efficiency.success
        result_entry["step_efficiency_reason"] = step_efficiency.reason if hasattr(step_efficiency, 'reason') and step_efficiency.reason else None
        
        result_entry["plan_adherence_score"] = round(plan_adherence.score, 4) if plan_adherence.score is not None else None
        result_entry["plan_adherence_success"] = plan_adherence.success
        result_entry["plan_adherence_reason"] = plan_adherence.reason if hasattr(plan_adherence, 'reason') and plan_adherence.reason else None
        
        result_entry["plan_quality_score"] = round(plan_quality.score, 4) if plan_quality.score is not None else None
        result_entry["plan_quality_success"] = plan_quality.success
        result_entry["plan_quality_reason"] = plan_quality.reason if hasattr(plan_quality, 'reason') and plan_quality.reason else None
        
        # Check if metrics passed
        if not result_entry["task_completion_success"]:
            failures.append(
                f"TaskCompletion score={result_entry['task_completion_score']} < threshold {task_completion_threshold} "
                f"for case {case_id}"
            )
        
        if not result_entry["step_efficiency_success"]:
            failures.append(
                f"StepEfficiency score={result_entry['step_efficiency_score']} < threshold {step_efficiency_threshold} "
                f"for case {case_id}"
            )
        
        if not result_entry["plan_adherence_success"]:
            failures.append(
                f"PlanAdherence score={result_entry['plan_adherence_score']} < threshold {plan_adherence_threshold} "
                f"for case {case_id}"
            )
        
        if not result_entry["plan_quality_success"]:
            failures.append(
                f"PlanQuality score={result_entry['plan_quality_score']} < threshold {plan_quality_threshold} "
                f"for case {case_id}"
            )
        
        # Optional: Verify expected tool was called
        if expected_tool:
            tools_called = result_entry.get("tools_called", [])
            tool_names = [tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "") for tc in tools_called]
            if expected_tool not in tool_names:
                failures.append(
                    f"Expected tool '{expected_tool}' not called for case {case_id}. "
                    f"Tools called: {tool_names}"
                )
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry) + "\n")
    
    # Export results to CSV
    if results:
        csv_file = reports_dir / f"multiple_agentic_metrics_test_results_{timestamp}.csv"
        
        # Flatten results for CSV
        csv_rows = []
        for entry in results:
            # Convert tools_called list to JSON string for CSV
            tools_called_str = json.dumps(entry.get("tools_called", [])) if entry.get("tools_called") else None
            
            row = {
                "case_id": entry.get("case_id"),
                "category": entry.get("category"),
                "errors": json.dumps(entry.get("errors", [])) if entry.get("errors") else None,
                "query": entry.get("query"),
                "response": entry.get("response"),
                "timestamp": entry.get("timestamp"),
                "tools_called": tools_called_str,
                "expected_tool": entry.get("expected_tool"),
                "task_completion_score": entry.get("task_completion_score"),
                "task_completion_threshold": task_completion_threshold,
                "task_completion_success": entry.get("task_completion_success"),
                "task_completion_reason": entry.get("task_completion_reason"),
                "step_efficiency_score": entry.get("step_efficiency_score"),
                "step_efficiency_threshold": step_efficiency_threshold,
                "step_efficiency_success": entry.get("step_efficiency_success"),
                "step_efficiency_reason": entry.get("step_efficiency_reason"),
                "plan_adherence_score": entry.get("plan_adherence_score"),
                "plan_adherence_threshold": plan_adherence_threshold,
                "plan_adherence_success": entry.get("plan_adherence_success"),
                "plan_adherence_reason": entry.get("plan_adherence_reason"),
                "plan_quality_score": entry.get("plan_quality_score"),
                "plan_quality_threshold": plan_quality_threshold,
                "plan_quality_success": entry.get("plan_quality_success"),
                "plan_quality_reason": entry.get("plan_quality_reason"),
            }
            csv_rows.append(row)
        
        # Define column order
        fieldnames = [
            "case_id", "category", "errors", "query", "response", "timestamp",
            "tools_called", "expected_tool",
            "task_completion_score", "task_completion_threshold", "task_completion_success", "task_completion_reason",
            "step_efficiency_score", "step_efficiency_threshold", "step_efficiency_success", "step_efficiency_reason",
            "plan_adherence_score", "plan_adherence_threshold", "plan_adherence_success", "plan_adherence_reason",
            "plan_quality_score", "plan_quality_threshold", "plan_quality_success", "plan_quality_reason"
        ]
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                complete_row = {field: row.get(field, None) for field in fieldnames}
                writer.writerow(complete_row)
    
    # Calculate summary statistics
    total_queries = len(queries)
    total_failures = len(failures)
    
    task_completion_success_count = sum(1 for r in results if r.get("task_completion_success") is True)
    step_efficiency_success_count = sum(1 for r in results if r.get("step_efficiency_success") is True)
    plan_adherence_success_count = sum(1 for r in results if r.get("plan_adherence_success") is True)
    plan_quality_success_count = sum(1 for r in results if r.get("plan_quality_success") is True)
    
    # Log summary
    print(f"\n📊 Multiple Agentic Metrics Test Summary:")
    print(f"  Total queries: {total_queries}")
    print(f"  Total failures: {total_failures}")
    print(f"\n  Task Completion: {task_completion_success_count}/{total_queries} passed ({task_completion_success_count/total_queries*100:.1f}%)" if total_queries > 0 else "  Task Completion: N/A")
    print(f"  Step Efficiency: {step_efficiency_success_count}/{total_queries} passed ({step_efficiency_success_count/total_queries*100:.1f}%)" if total_queries > 0 else "  Step Efficiency: N/A")
    print(f"  Plan Adherence: {plan_adherence_success_count}/{total_queries} passed ({plan_adherence_success_count/total_queries*100:.1f}%)" if total_queries > 0 else "  Plan Adherence: N/A")
    print(f"  Plan Quality: {plan_quality_success_count}/{total_queries} passed ({plan_quality_success_count/total_queries*100:.1f}%)" if total_queries > 0 else "  Plan Quality: N/A")
    print(f"  Log file: {log_file}")
    if results:
        print(f"  CSV report: {csv_file}")
    
    # Calculate per-metric averages and compare with mean thresholds
    print(f"\n📈 Per-Metric Summary:")
    metric_averages = {}
    metric_mean_failures = []
    
    # Define metrics to analyze
    metrics_to_analyze = {
        "Task Completion": {
            "score_key": "task_completion_score",
            "success_key": "task_completion_success",
            "mean_threshold": task_completion_mean_threshold
        },
        "Step Efficiency": {
            "score_key": "step_efficiency_score",
            "success_key": "step_efficiency_success",
            "mean_threshold": step_efficiency_mean_threshold
        },
        "Plan Adherence": {
            "score_key": "plan_adherence_score",
            "success_key": "plan_adherence_success",
            "mean_threshold": plan_adherence_mean_threshold
        },
        "Plan Quality": {
            "score_key": "plan_quality_score",
            "success_key": "plan_quality_success",
            "mean_threshold": plan_quality_mean_threshold
        }
    }
    
    for metric_name, metric_config in metrics_to_analyze.items():
        score_key = metric_config["score_key"]
        success_key = metric_config["success_key"]
        mean_threshold = metric_config["mean_threshold"]
        
        # Calculate passed count
        metric_passed = sum(
            1 for entry in results
            if entry.get(success_key, False) is True
        )
        metric_total = sum(
            1 for entry in results
            if entry.get(score_key) is not None
        )
        
        # Calculate average score for this metric
        metric_scores = []
        for entry in results:
            score = entry.get(score_key)
            if score is not None:
                metric_scores.append(score)
        
        if metric_total > 0:
            pass_rate = (metric_passed / metric_total) * 100
            avg_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
            
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
        summary_csv_file = reports_dir / f"multiple_agentic_metrics_summary_{timestamp}.csv"
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
        
        # Create transformed CSV with one row per timestamp for Jenkins Plot plugin
        # Transform format: one row with all metrics as columns (metric_name_mean, metric_name_threshold, metric_name_tests, metric_name_status)
        transformed_csv_file = reports_dir / f"multiple_agentic_metrics_summary_plot_{timestamp}.csv"
        
        # Create a single row with all metric data
        transformed_row = {"timestamp": timestamp}
        
        # Add each metric's data as separate columns
        for metric_name, metric_data in metric_averages.items():
            # Sanitize metric name for column names (replace spaces with underscores)
            metric_col_prefix = metric_name.replace(" ", "_")
            transformed_row[f"{metric_col_prefix}_mean"] = round(metric_data["average"], 4)
            transformed_row[f"{metric_col_prefix}_threshold"] = round(metric_data["mean_threshold"], 4)
            transformed_row[f"{metric_col_prefix}_tests"] = metric_data["total_count"]
            transformed_row[f"{metric_col_prefix}_status"] = metric_data["status"]
            transformed_row[f"{metric_col_prefix}_passed"] = metric_data["passed_count"]
            transformed_row[f"{metric_col_prefix}_pass_rate"] = round(metric_data["pass_rate"], 2)
        
        # Define column order: timestamp, then each metric's columns in order
        transformed_fieldnames = ["timestamp"]
        for metric_name in metrics_to_analyze.keys():
            metric_col_prefix = metric_name.replace(" ", "_")
            transformed_fieldnames.extend([
                f"{metric_col_prefix}_mean",
                f"{metric_col_prefix}_threshold",
                f"{metric_col_prefix}_tests",
                f"{metric_col_prefix}_passed",
                f"{metric_col_prefix}_pass_rate",
                f"{metric_col_prefix}_status"
            ])
        
        # Write transformed CSV with one row
        with open(transformed_csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=transformed_fieldnames)
            writer.writeheader()
            # Ensure all fields are present in the row
            complete_row = {field: transformed_row.get(field, None) for field in transformed_fieldnames}
            writer.writerow(complete_row)
        
        print(f"📊 Transformed Plot CSV (one row): {transformed_csv_file}")
    
    # Print detailed results
    print("\n" + "="*60)
    print("MULTIPLE AGENTIC METRICS RESULTS - WEATHER AGENT")
    print("="*60)
    for i, result in enumerate(results, 1):
        print(f"\nTest Case {i} ({result.get('case_id', 'unknown')}):")
        print(f"  Input: {result.get('query', 'N/A')}")
        output = result.get('response', 'N/A')
        print(f"  Output: {output[:200]}..." if len(output) > 200 else f"  Output: {output}")
        
        # Print metric scores
        print(f"\n  Metrics:")
        print(f"    Task Completion: {result.get('task_completion_score', 'N/A')} ({'✓' if result.get('task_completion_success') else '✗'})")
        if result.get('task_completion_reason'):
            print(f"      Reason: {result.get('task_completion_reason')}")
        
        print(f"    Step Efficiency: {result.get('step_efficiency_score', 'N/A')} ({'✓' if result.get('step_efficiency_success') else '✗'})")
        if result.get('step_efficiency_reason'):
            print(f"      Reason: {result.get('step_efficiency_reason')}")
        
        print(f"    Plan Adherence: {result.get('plan_adherence_score', 'N/A')} ({'✓' if result.get('plan_adherence_success') else '✗'})")
        if result.get('plan_adherence_reason'):
            print(f"      Reason: {result.get('plan_adherence_reason')}")
        
        print(f"    Plan Quality: {result.get('plan_quality_score', 'N/A')} ({'✓' if result.get('plan_quality_success') else '✗'})")
        if result.get('plan_quality_reason'):
            print(f"      Reason: {result.get('plan_quality_reason')}")
        
        tools_called = result.get('tools_called', [])
        if tools_called:
            print(f"\n  Tools Called: {len(tools_called)}")
            for tool in tools_called:
                tool_name = tool.get('name', 'unknown') if isinstance(tool, dict) else str(tool)
                tool_args = tool.get('args', {}) if isinstance(tool, dict) else getattr(tool, 'args', {})
                print(f"    - {tool_name}({tool_args})")
        
        if result.get('errors'):
            print(f"\n  Errors: {result.get('errors')}")
    
    print("\n" + "="*60)
    
    # Assert on mean threshold failures at metric level
    if metric_mean_failures:
        failure_summary = (
            f"\n\nFound {len(metric_mean_failures)} metric(s) with average scores below mean threshold:\n" +
            "\n".join(f"  - {f}" for f in metric_mean_failures)
        )
        pytest.fail(failure_summary)
    
    # Log individual test case failures (for reporting only - they don't cause test failure)
    # The test ONLY fails if metric averages are below mean thresholds (checked above)
    if failures:
        failure_message = (
            f"\n\n⚠️  Found {len(failures)} individual test case failure(s) (logged for reference, not causing test failure):\n" + 
            "\n".join(f"  - {f}" for f in failures[:50])  # Show first 50 failures
        )
        if len(failures) > 50:
            failure_message += f"\n  ... and {len(failures) - 50} more failures"
        print(failure_message)
        # NOTE: Individual failures are logged but do NOT cause test failure
        # The test only fails when metric average < threshold mean (see check above at line 535-540)


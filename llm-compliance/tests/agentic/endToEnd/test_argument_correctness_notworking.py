"""Not working currently Test DeepEval ArgumentCorrectnessMetric on weather agent queries from argumentCorrectnessData.jsonl."""

import json
import pytest
from pathlib import Path
import sys
import csv
from datetime import datetime
from typing import Any

# DeepEval imports
from deepeval.tracing import observe

# Add parent directory to Python path to enable imports
# File is at: llm-compliance/tests/agentic/endToEnd/test_argument_correctness.py
# Need to go up 4 levels to reach llm-compliance/ where app/ and utils/ are
parent_dir = Path(__file__).parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app.weather_client import WeatherAgentClient
from utils.deepE2EAgenticMetrics import verify_argument_correctness_metric


# Initialize weather agent client (module-level for reuse)
weather_client = WeatherAgentClient()


@observe()
def weather_agent(input_query: str) -> tuple[str, Any]:
    """
    Weather agent wrapper function for DeepEval tracing.
    
    The @observe() decorator enables DeepEval to:
    1. Automatically trace function execution
    2. Extract tool calls made during execution
    3. Provide execution context to ArgumentCorrectnessMetric for evaluation
    
    Args:
        input_query: The user's weather-related query
    
    Returns:
        Tuple of (response_text, response_object):
        - response_text: The agent's response text (for evaluation)
        - response_object: The full response object (for accessing tools_called, etc.)
    """
    response = weather_client.ask(input_query)
    return response.text, response


def test_argument_correctness_metrics(thresholds):
    """Test ArgumentCorrectnessMetric on queries from argumentCorrectnessData.jsonl."""
    # Load argument correctness queries
    queries_path = Path(__file__).parent.parent.parent.parent / "testData" / "agentic" / "argumentCorrectnessData.jsonl"
    
    if not queries_path.exists():
        pytest.skip(f"Argument correctness queries file not found: {queries_path}")
    
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
        pytest.skip("No argument correctness queries found in file")
    
    # Setup logging directories
    evidence_dir = Path(__file__).parent.parent.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = evidence_dir / f"argument_correctness_test_log_{timestamp}.jsonl"
    
    # Get threshold from config (default to 0.7 if not found)
    argument_correctness_threshold = thresholds.get("deepeval", {}).get("agentic", {}).get("argument_correctness_min", 0.7)
    # Get model from config (default to "gpt-4" if not found)
    argument_correctness_model = thresholds.get("deepeval", {}).get("agentic", {}).get("argument_correctness_model", "gpt-4")
    
    # Collect all failures and results
    failures = []
    results = []
    
    # Process each query
    for i, query_case in enumerate(queries):
        case_id = query_case.get("id", f"arg_{i+1}")
        query = query_case.get("query", "")
        category = query_case.get("category", "weather_query")
        expected_tool = query_case.get("expected_tool", None)
        expected_args = query_case.get("expected_args", None)
        
        if not query:
            continue
        
        # Reset conversation for each test case to ensure independence
        weather_client.reset_conversation(f"test-case-{case_id}")
        
        # Initialize result entry
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "case_id": case_id,
            "category": category,
            "query": query,
            "response": None,
            "score": None,
            "success": None,
            "reason": None,
            "tools_called": [],
            "expected_tool": expected_tool,
            "expected_args": expected_args,
            "errors": []
        }
        
        try:
            # Call the weather agent with the query
            # The @observe() decorator automatically traces this execution for DeepEval
            # Returns both response text and response object
            output, response = weather_agent(query)
            result_entry["response"] = output
            
            # Get tool calls information from the response object
            tools_called = response.tools_called if hasattr(response, 'tools_called') else []
            result_entry["tools_called"] = tools_called
            
            # Evaluate argument correctness using the utility function
            # Pass tools_called explicitly as ArgumentCorrectnessMetric needs tool arguments
            score, reason, passed, error_msg = verify_argument_correctness_metric(
                actual_output=output,
                input_query=query,
                tools_called=tools_called,  # Pass tools_called with args for evaluation
                threshold=argument_correctness_threshold,
                model=argument_correctness_model,
                include_reason=True,
                case_id=case_id
            )
            
            if error_msg:
                raise Exception(error_msg)
            
            result_entry["score"] = round(score, 4) if score is not None else None
            result_entry["success"] = passed
            result_entry["reason"] = reason
            
            # Check if metric passed
            if not passed:
                failures.append(
                    f"ArgumentCorrectness score={score:.4f} < threshold {argument_correctness_threshold} "
                    f"for case {case_id}"
                )
            
            # Optional: Compare actual args with expected args (for logging/debugging)
            if expected_args and tools_called:
                for tool in tools_called:
                    tool_name = tool.get("name", "")
                    if tool_name == expected_tool:
                        actual_args = tool.get("args", {})
                        # Log if args don't match expected (informational, not a failure)
                        if actual_args != expected_args:
                            result_entry["errors"].append(
                                f"Args mismatch: expected {expected_args}, got {actual_args}"
                            )
            
        except Exception as e:
            error_msg = f"Error evaluating argument correctness for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["success"] = False
            failures.append(error_msg)
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry) + "\n")
    
    # Export results to CSV
    if results:
        csv_file = reports_dir / f"argument_correctness_test_results_{timestamp}.csv"
        
        # Flatten results for CSV
        csv_rows = []
        for entry in results:
            # Convert tools_called list to JSON string for CSV
            tools_called_str = json.dumps(entry.get("tools_called", [])) if entry.get("tools_called") else None
            expected_args_str = json.dumps(entry.get("expected_args", {})) if entry.get("expected_args") else None
            
            row = {
                "case_id": entry.get("case_id"),
                "category": entry.get("category"),
                "errors": json.dumps(entry.get("errors", [])) if entry.get("errors") else None,
                "query": entry.get("query"),
                "response": entry.get("response"),
                "timestamp": entry.get("timestamp"),
                "score": entry.get("score"),
                "threshold": argument_correctness_threshold,
                "success": entry.get("success"),
                "reason": entry.get("reason"),
                "tools_called": tools_called_str,
                "expected_tool": entry.get("expected_tool"),
                "expected_args": expected_args_str,
            }
            csv_rows.append(row)
        
        # Define column order
        fieldnames = [
            "case_id", "category", "errors", "query", "response", "timestamp",
            "score", "threshold", "success", "reason", "expected_tool", "expected_args", "tools_called"
        ]
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                complete_row = {field: row.get(field, None) for field in fieldnames}
                writer.writerow(complete_row)
    
    # Log summary before any assertions
    total_queries = len(queries)
    total_failures = len(failures)
    success_count = sum(1 for r in results if r.get("success") is True)
    
    print(f"\n📊 Argument Correctness Metric Test Summary:")
    print(f"  Total queries: {total_queries}")
    print(f"  Passed: {success_count}")
    print(f"  Failed: {total_failures}")
    print(f"  Success rate: {success_count}/{total_queries} ({success_count/total_queries*100:.1f}%)" if total_queries > 0 else "  Success rate: N/A")
    print(f"  Log file: {log_file}")
    if results:
        print(f"  CSV report: {csv_file}")
    
    # Print detailed results
    print("\n" + "="*60)
    print("ARGUMENT CORRECTNESS METRIC RESULTS - WEATHER AGENT")
    print("="*60)
    for i, result in enumerate(results, 1):
        print(f"\nTest Case {i} ({result.get('case_id', 'unknown')}):")
        print(f"  Input: {result.get('query', 'N/A')}")
        output = result.get('response', 'N/A')
        print(f"  Output: {output[:200]}..." if len(output) > 200 else f"  Output: {output}")
        score = result.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  Score: {score_str}")
        success = result.get('success')
        success_str = '✓ YES' if success else '✗ NO' if success is not None else 'N/A'
        print(f"  Passed: {success_str}")
        expected_tool = result.get('expected_tool', 'N/A')
        expected_args = result.get('expected_args', {})
        print(f"  Expected Tool: {expected_tool}")
        print(f"  Expected Args: {expected_args}")
        tools_called = result.get('tools_called', [])
        if tools_called:
            print(f"  Tools Called: {len(tools_called)}")
            for tool in tools_called:
                tool_name = tool.get('name', 'unknown')
                tool_args = tool.get('args', {})
                print(f"    - {tool_name}({tool_args})")
        else:
            print(f"  Tools Called: None")
        if result.get('reason'):
            print(f"  Reason: {result.get('reason')}")
    
    print("\n" + "="*60)
    
    # Assert all failures at the end
    if failures:
        failure_message = (
            f"\n\nFound {len(failures)} failure(s):\n" + 
            "\n".join(f"  - {f}" for f in failures[:50])
        )
        if len(failures) > 50:
            failure_message += f"\n  ... and {len(failures) - 50} more failures"
        pytest.fail(failure_message)


"""Test DeepEval PlanAdherenceMetric on weather agent queries from taskCompletionData.jsonl."""

import json
import pytest
from pathlib import Path
import sys
import csv
from datetime import datetime
from typing import Any

# DeepEval imports
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import PlanAdherenceMetric
from deepeval.test_case import LLMTestCase

# Add utils to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.weather_client import WeatherAgentClient
from utils.deepComponentAgenticMetrics import verify_plan_adherence_metric


# Initialize weather agent client (module-level for reuse)
weather_client = WeatherAgentClient()


@observe()
def agent_planning_component(user_input: str) -> str:
    """
    Planning component that generates a plan for handling the weather query.
    PlanAdherenceMetric needs to see the plan captured in the trace attributes.
    
    Args:
        user_input: The user's weather-related query
    
    Returns:
        Plan string describing the steps to take
    """
    # Generate a plan based on the query
    generated_plan = (
        f"Step 1: Parse the user's query '{user_input}' to extract location information. "
        f"Step 2: Call get_current_weather tool with the extracted location. "
        f"Step 3: Format and return the weather information to the user."
    )
    
    # Capture the plan in the trace attributes using update_current_span
    # Use input and output parameters directly (not test_case)
    # PlanAdherenceMetric looks for planning steps in the trace with input/output
    update_current_span(
        input=user_input,
        output=generated_plan  # The plan is captured here for the metric to find
    )
    
    return generated_plan


@observe()
def weather_agent(input_query: str) -> tuple[str, Any]:
    """
    Weather agent wrapper function for DeepEval tracing with plan adherence evaluation.
    
    The @observe() decorator enables DeepEval to:
    1. Automatically trace function execution
    2. Extract tool calls made during execution
    3. Provide execution context to PlanAdherenceMetric for plan adherence evaluation
    
    PlanAdherenceMetric requires the plan to be captured using update_current_span()
    in the planning component. We call agent_planning_component() to generate and
    capture the plan, then execute it. After execution, we update the span with the
    final output so PlanAdherenceMetric can compare the plan with the execution.
    
    Args:
        input_query: The user's weather-related query
    
    Returns:
        Tuple of (response_text, response_object):
        - response_text: The agent's response text (for evaluation)
        - response_object: The full response object (for accessing tools_called, etc.)
    """
    # Generate and capture the plan using the planning component
    # The plan is captured in the trace via update_current_span() in agent_planning_component()
    # IMPORTANT: The plan must remain in the trace for PlanAdherenceMetric to find it
    plan = agent_planning_component(input_query)
    
    # Execute the plan by calling the weather agent
    response = weather_client.ask(input_query)
    
    # DO NOT update_current_span again here - it would overwrite the plan!
    # PlanAdherenceMetric needs the plan to be in the trace, and it will compare
    # the plan with the actual execution (tool calls and output) automatically
    # The final output is available through the response.text return value
    
    return response.text, response


def test_plan_adherence_metrics(thresholds):
    """Test PlanAdherenceMetric on queries from taskCompletionData.jsonl."""
    # Load task completion queries
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
    evidence_dir = Path(__file__).parent.parent.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    reports_dir = Path(__file__).parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = evidence_dir / f"plan_adherence_test_log_{timestamp}.jsonl"
    
    # Get threshold from config (default to 0.7 if not found)
    plan_adherence_threshold = thresholds.get("deepeval", {}).get("agentic", {}).get("plan_adherence_min", 0.7)
    plan_adherence_model = thresholds.get("deepeval", {}).get("agentic", {}).get("plan_adherence_model", "gpt-4o")
    
    # Collect all failures and results
    failures = []
    results = []
    
    # Process each query
    for i, query_case in enumerate(queries):
        case_id = query_case.get("id", f"plan_{i+1}")
        query = query_case.get("query", "")
        category = query_case.get("category", "weather_query")
        expected_tool = query_case.get("expected_tool", None)
        
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
            "errors": []
        }
        
        try:
            # Call the weather agent with the query
            # The @observe() decorator automatically traces this execution for DeepEval
            # update_current_trace() is called inside weather_agent() to set up trace context
            # Returns both response text and response object
            output, response = weather_agent(query)
            result_entry["response"] = output
            
            # Get tool calls information from the response object
            tools_called = response.tools_called if hasattr(response, 'tools_called') else []
            result_entry["tools_called"] = tools_called
            
            # Evaluate plan adherence using the utility function
            # PlanAdherenceMetric evaluates whether the agent followed a logical plan
            # It uses the trace information set up by update_current_trace()
            score, reason, passed, error_msg = verify_plan_adherence_metric(
                actual_output=output,
                input_query=query,
                threshold=plan_adherence_threshold,
                model=plan_adherence_model,
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
                    f"PlanAdherence score={score:.4f} < threshold {plan_adherence_threshold} "
                    f"for case {case_id}"
                )
            
            # Optional: Verify expected tool was called
            if expected_tool:
                tool_names = [tc.get("name", "") for tc in tools_called]
                if expected_tool not in tool_names:
                    failures.append(
                        f"Expected tool '{expected_tool}' not called for case {case_id}. "
                        f"Tools called: {tool_names}"
                    )
            
        except Exception as e:
            error_msg = f"Error evaluating plan adherence for case {case_id}: {e}"
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
        csv_file = reports_dir / f"plan_adherence_test_results_{timestamp}.csv"
        
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
                "score": entry.get("score"),
                "threshold": plan_adherence_threshold,
                "success": entry.get("success"),
                "reason": entry.get("reason"),
                "tools_called": tools_called_str,
                "expected_tool": entry.get("expected_tool"),
            }
            csv_rows.append(row)
        
        # Define column order
        fieldnames = [
            "case_id", "category", "errors", "query", "response", "timestamp",
            "score", "threshold", "success", "reason", "tools_called", "expected_tool"
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
    
    print(f"\n📊 Plan Adherence Metric Test Summary:")
    print(f"  Total queries: {total_queries}")
    print(f"  Passed: {success_count}")
    print(f"  Failed: {total_failures}")
    print(f"  Success rate: {success_count/total_queries*100:.1f}%" if total_queries > 0 else "  Success rate: N/A")
    print(f"  Log file: {log_file}")
    if results:
        print(f"  CSV report: {csv_file}")
    
    # Print detailed results
    print("\n" + "="*60)
    print("PLAN ADHERENCE METRIC RESULTS - WEATHER AGENT")
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
        tools_called = result.get('tools_called', [])
        if tools_called:
            print(f"  Tools Called: {len(tools_called)}")
            for tool in tools_called:
                print(f"    - {tool.get('name', 'unknown')}({tool.get('args', {})})")
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


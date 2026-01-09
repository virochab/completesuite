"""
Test suite for Weather Agent trajectory evaluation using extract_langgraph_trajectory_from_thread
and graph_trajectory_llm_match from agentevals.
"""
from __future__ import annotations

import pytest
import sys
import uuid
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread
from agentevals.graph_trajectory.llm import create_graph_trajectory_llm_as_judge

# Create the LLM evaluator function
# This creates a function that evaluates trajectories using an LLM judge
graph_trajectory_llm_match = create_graph_trajectory_llm_as_judge(
    model="gpt-4o",  # Model identifier string
    continuous=True,  # Get continuous score (0-1) instead of boolean
    use_reasoning=True  # Include reasoning in output
)

# Add parent directory to Python path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.weather_client_with_trace import WeatherAgentClientWithTrace


# ==================== Module-level cumulative results storage ====================
# Store all test results across all test methods in this file
_file_level_results: list[Dict[str, Any]] = []
_file_level_reports_dir: Path = None


@pytest.fixture(scope="function")
def weather_client():
    """Create weather client with trace support for each test"""
    return WeatherAgentClientWithTrace()


@pytest.fixture(scope="session", autouse=True)
def setup_file_level_reporting():
    """Setup file-level reporting directories once per test session"""
    global _file_level_reports_dir
    reports_dir = Path(__file__).parent.parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    _file_level_reports_dir = reports_dir
    yield
    # Cleanup: write cumulative CSV after all tests complete
    write_cumulative_csv()


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable objects to serializable formats.
    Handles LangChain messages, Pydantic models, and other complex objects.
    
    Args:
        obj: Object to make JSON serializable
    
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'content') and hasattr(obj, 'type'):
        # LangChain message objects (HumanMessage, AIMessage, etc.)
        try:
            return {
                "type": getattr(obj, 'type', str(type(obj).__name__)),
                "content": str(getattr(obj, 'content', '')),
                "class": obj.__class__.__name__
            }
        except Exception:
            return str(obj)
    elif hasattr(obj, 'model_dump'):
        # Pydantic models
        try:
            return _make_json_serializable(obj.model_dump())
        except Exception:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        # Other objects with __dict__
        try:
            return {key: _make_json_serializable(value) for key, value in obj.__dict__.items()}
        except Exception:
            return str(obj)
    else:
        # Fallback: convert to string
        return str(obj)


def write_cumulative_csv():
    """Write cumulative CSV file with all results from all tests in this file"""
    global _file_level_results, _file_level_reports_dir
    
    if not _file_level_results:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = _file_level_reports_dir / f"trajectory_llm_all_tests_cumulative_{timestamp}.csv"
    
    # Collect all unique fieldnames from all result entries
    all_fieldnames = set()
    for entry in _file_level_results:
        all_fieldnames.update(entry.keys())
    fieldnames = sorted(list(all_fieldnames))
    
    # Flatten results for CSV (convert complex types to JSON strings)
    csv_rows = []
    for entry in _file_level_results:
        row = {}
        for field in fieldnames:
            value = entry.get(field)
            # Convert to JSON-serializable format first (handles HumanMessage, etc.)
            serializable_value = _make_json_serializable(value)
            # Convert lists and dicts to JSON strings for CSV compatibility
            if isinstance(serializable_value, (list, dict)):
                row[field] = json.dumps(serializable_value) if serializable_value else None
            else:
                row[field] = serializable_value
        csv_rows.append(row)
    
    # Write CSV file
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            complete_row = {field: row.get(field, None) for field in fieldnames}
            writer.writerow(complete_row)
    
    print(f"\n📊 Cumulative CSV report written: {csv_file}")
    print(f"   Total results: {len(_file_level_results)}")


def add_to_file_level_results(result_entry: Dict[str, Any]):
    """Add a result entry to the file-level cumulative results"""
    global _file_level_results
    _file_level_results.append(result_entry)


# ==================== Reusable Logging and Reporting Utilities ====================

def setup_logging_directories(test_name: str) -> tuple[Path, Path, Path]:
    """
    Setup logging directories and return paths for evidence, reports, and log file.
    
    Args:
        test_name: Name identifier for the test (used in log file name)
    
    Returns:
        Tuple of (evidence_dir, reports_dir, log_file_path)
    """
    evidence_dir = Path(__file__).parent.parent.parent.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    reports_dir = Path(__file__).parent.parent.parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = evidence_dir / f"{test_name}_log_{timestamp}.jsonl"
    
    return evidence_dir, reports_dir, log_file


def create_result_entry(
    case_id: str,
    query: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a standardized result entry dictionary for logging.
    
    Args:
        case_id: Unique identifier for the test case
        query: The input query
        **kwargs: Additional fields to include in the result entry
    
    Returns:
        Dictionary with standardized result entry structure
    """
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "case_id": case_id,
        "query": query,
        "errors": []
    }
    
    # Add any additional fields passed via kwargs
    result_entry.update(kwargs)
    
    return result_entry


def log_result_entry(log_file: Path, result_entry: Dict[str, Any]):
    """
    Log a result entry to the JSONL log file.
    
    Args:
        log_file: Path to the log file
        result_entry: Result entry dictionary to log
    """
    # Convert non-serializable objects to strings before JSON serialization
    serializable_entry = _make_json_serializable(result_entry)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(serializable_entry) + "\n")


def export_results_to_csv(
    results: list[Dict[str, Any]],
    reports_dir: Path,
    test_name: str,
    fieldnames: list[str] = None
) -> Path:
    """
    Export results to CSV file.
    
    Args:
        results: List of result entry dictionaries
        reports_dir: Directory to save CSV report
        test_name: Name identifier for the test (used in CSV file name)
        fieldnames: Optional list of field names for CSV columns. If None, auto-detects from results.
    
    Returns:
        Path to the created CSV file
    """
    if not results:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = reports_dir / f"{test_name}_results_{timestamp}.csv"
    
    # Auto-detect fieldnames if not provided
    if fieldnames is None:
        # Collect all unique keys from all result entries
        all_keys = set()
        for entry in results:
            all_keys.update(entry.keys())
        fieldnames = sorted(list(all_keys))
    
    # Flatten results for CSV (convert complex types to JSON strings)
    csv_rows = []
    for entry in results:
        row = {}
        for field in fieldnames:
            value = entry.get(field)
            # Convert to JSON-serializable format first (handles HumanMessage, etc.)
            serializable_value = _make_json_serializable(value)
            # Convert lists and dicts to JSON strings for CSV compatibility
            if isinstance(serializable_value, (list, dict)):
                row[field] = json.dumps(serializable_value) if serializable_value else None
            else:
                row[field] = serializable_value
        csv_rows.append(row)
    
    # Write CSV file
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            complete_row = {field: row.get(field, None) for field in fieldnames}
            writer.writerow(complete_row)
    
    return csv_file


def print_test_summary(
    test_name: str,
    total_cases: int,
    success_count: int,
    failures: list[str],
    log_file: Path,
    csv_file: Path = None
):
    """
    Print test summary statistics.
    
    Args:
        test_name: Name of the test
        total_cases: Total number of test cases
        success_count: Number of successful test cases
        failures: List of failure messages
        log_file: Path to the log file
        csv_file: Optional path to the CSV report file
    """
    total_failures = len(failures)
    
    print(f"\n📊 {test_name} Test Summary:")
    print(f"  Total test cases: {total_cases}")
    print(f"  Passed: {success_count}")
    print(f"  Failed: {total_failures}")
    if total_cases > 0:
        success_rate = (success_count / total_cases) * 100
        print(f"  Success rate: {success_count}/{total_cases} ({success_rate:.1f}%)")
    else:
        print(f"  Success rate: N/A")
    print(f"  Log file: {log_file}")
    if csv_file:
        print(f"  CSV report: {csv_file}")


def print_detailed_results(
    results: list[Dict[str, Any]],
    title: str = "TEST RESULTS"
):
    """
    Print detailed results for each test case.
    
    Args:
        results: List of result entry dictionaries
        title: Title to display for the results section
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    for i, result in enumerate(results, 1):
        case_id = result.get('case_id', 'unknown')
        print(f"\nTest Case {i} ({case_id}):")
        print(f"  Query: {result.get('query', 'N/A')}")
        
        response = result.get('response', 'N/A')
        if isinstance(response, str) and len(response) > 200:
            print(f"  Response: {response[:200]}...")
        else:
            print(f"  Response: {response}")
        
        # Print test-specific fields if they exist
        if 'expected_steps' in result:
            print(f"  Expected Steps: {result.get('expected_steps', [])}")
        if 'actual_steps' in result:
            print(f"  Actual Steps: {result.get('actual_steps', [])}")
        if 'match_score' in result:
            match_score = result.get('match_score')
            if isinstance(match_score, (int, float)):
                match_str = f"{match_score:.4f}"
            else:
                match_str = '✓ YES' if match_score else '✗ NO' if match_score is not None else 'N/A'
            print(f"  Match Score: {match_str}")
        if 'score' in result:
            score = result.get('score')
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"  Score: {score_str}")
        if 'success' in result:
            success = result.get('success')
            success_str = '✓ YES' if success else '✗ NO' if success is not None else 'N/A'
            print(f"  Passed: {success_str}")
        if 'reasoning' in result:
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
        if 'comment' in result:
            print(f"  Comment: {result.get('comment', 'N/A')}")
        
        tools_called = result.get('tools_called', [])
        if tools_called:
            print(f"  Tools Called: {len(tools_called)}")
            for tool in tools_called:
                tool_name = tool.get('name', 'unknown') if isinstance(tool, dict) else str(tool)
                tool_args = tool.get('args', {}) if isinstance(tool, dict) else {}
                print(f"    - {tool_name}({tool_args})")
        
        if result.get('errors'):
            print(f"  Errors: {result.get('errors')}")
        if result.get('reason'):
            print(f"  Reason: {result.get('reason')}")
    
    print("\n" + "="*60)


def assert_test_results(
    results: list[Dict[str, Any]],
    test_cases: list,
    failures: list[str],
    success_field: str = "match_score",
    threshold: float = 0.7
):
    """
    Assert test results and raise pytest.fail if there are failures.
    
    Args:
        results: List of result entry dictionaries
        test_cases: Original list of test cases (for count validation)
        failures: List of failure messages
        success_field: Field name in result entry that indicates success (default: "match_score")
        threshold: Minimum score threshold for continuous scores (default: 0.7)
    """
    # Assert all test cases were evaluated
    assert len(results) == len(test_cases), "All test cases should be evaluated"
    
    # Assert all failures at the end
    if failures:
        failure_message = (
            f"\n\nFound {len(failures)} failure(s):\n" + 
            "\n".join(f"  - {f}" for f in failures[:50])
        )
        if len(failures) > 50:
            failure_message += f"\n  ... and {len(failures) - 50} more failures"
        pytest.fail(failure_message)


def test_weather_agent_trajectory_single_query_llm(weather_client):
    """
    Test LLM-based trajectory evaluation for a single weather query loaded from trajectoryStrict.jsonl.
    """
    print("\n" + "="*80)
    print("TEST: test_weather_agent_trajectory_single_query_llm")
    print("="*80)
    
    # Load test cases from JSONL file (filter for single_query type)
    test_data_path = Path(__file__).parent.parent.parent.parent / "testData" / "agentic" / "trajectoryStrict.jsonl"
    
    test_cases = []
    if test_data_path.exists():
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                    # Filter for single_query test type
                    if case.get("test_type") == "single_query":
                        test_cases.append((
                            case.get("query", ""),
                            case.get("id", "unknown")
                        ))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    else:
        pytest.skip(f"Test data file not found: {test_data_path}")
    
    if not test_cases:
        pytest.skip("No single_query test cases found in trajectoryStrict.jsonl")
    
    print(f"📋 Loaded {len(test_cases)} test cases from {test_data_path.name}")
    
    # Setup logging directories using reusable utility
    evidence_dir, reports_dir, log_file = setup_logging_directories("trajectory_single_query_llm_test")
    
    # Collect all failures and results
    failures = []
    results = []
    threshold = 0.7  # Score threshold for LLM evaluation
    
    for i, (query, case_id) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i} ({case_id}): '{query}'")
        
        # Initialize result entry using reusable utility
        result_entry = create_result_entry(
            case_id=case_id,
            query=query,
            actual_steps=None,
            match_result=None,
            match_score=None,
            response=None,
            tools_called=[],
            thread_id=None,
            reasoning=None
        )
        
        try:
            # Generate unique thread ID for this test case
            thread_id = str(uuid.uuid4())
            weather_client.reset_conversation(thread_id)
            result_entry["thread_id"] = thread_id
            
            # Run agent with trace capture
            print(f"\n🔄 Running weather agent (thread: {thread_id[:8]}...)")
            response = weather_client.ask(query, capture_trace=True)
            result_entry["response"] = response.text
            
            # Get tool calls information
            tools_called = response.tools_called if hasattr(response, 'tools_called') else []
            result_entry["tools_called"] = tools_called
            
            # Extract trajectory
            if response.trace_info:
                extracted_trajectory = response.trace_info
            else:
                extracted_trajectory = extract_langgraph_trajectory_from_thread(
                    weather_client.agent,
                    weather_client.config
                )
            
            outputs = extracted_trajectory.get('outputs', {})
            if isinstance(outputs, dict):
                actual_steps = outputs.get('steps', [])
            else:
                actual_steps = extracted_trajectory.get('steps', [])
            
            result_entry["actual_steps"] = actual_steps
            
            print(f"   📋 Actual steps: {actual_steps}")
            
            # Perform LLM-based trajectory evaluation (no reference_trajectory needed)
            # Ensure inputs and outputs are properly formatted before calling
            trajectory_outputs = extracted_trajectory.get("outputs")
            trajectory_inputs = extracted_trajectory.get("inputs")
            
            if trajectory_outputs is None:
                raise ValueError(f"Extracted trajectory outputs is None for case {case_id}")
            
            # Prepare inputs - the LLM evaluator expects inputs to be a list matching the length of outputs["results"]
            # If inputs is None or missing, create a list with the query repeated for each result
            if trajectory_inputs is None:
                # Get the results list to determine the required length
                results_list = trajectory_outputs.get("results", [])
                steps_list = trajectory_outputs.get("steps", [])
                
                # Use the length of results or steps, whichever is available
                required_length = len(results_list) if results_list else (len(steps_list) if steps_list else 1)
                
                # Create inputs list matching the required length
                trajectory_inputs = [query] * required_length if required_length > 0 else [query]
            elif not isinstance(trajectory_inputs, list):
                # If inputs is not a list, convert it to a list
                results_list = trajectory_outputs.get("results", [])
                required_length = len(results_list) if results_list else 1
                trajectory_inputs = [trajectory_inputs] * required_length if required_length > 1 else [trajectory_inputs]
            
            # Ensure inputs length matches outputs["results"] length
            results_list = trajectory_outputs.get("results", [])
            if results_list and len(trajectory_inputs) != len(results_list):
                # Adjust inputs length to match results length
                if len(trajectory_inputs) < len(results_list):
                    # Pad with query if inputs is shorter
                    trajectory_inputs.extend([query] * (len(results_list) - len(trajectory_inputs)))
                else:
                    # Truncate if inputs is longer
                    trajectory_inputs = trajectory_inputs[:len(results_list)]
            
            # Serialize trajectory_inputs and trajectory_outputs to handle HumanMessage objects
            # before passing to LLM evaluator
            serialized_inputs = _make_json_serializable(trajectory_inputs)
            serialized_outputs = _make_json_serializable(trajectory_outputs)
            
            print(f"\n🔍 Performing LLM-based trajectory evaluation...")
            match_result = graph_trajectory_llm_match(
                inputs=serialized_inputs,
                outputs=serialized_outputs,
                # No reference_outputs - LLM evaluator doesn't need it
            )
            
            # Extract score and reasoning from match_result
            if isinstance(match_result, dict):
                score = match_result.get("score", match_result.get("passed", None))
                reasoning = match_result.get("reasoning", match_result.get("comment", match_result.get("explanation", None)))
            elif isinstance(match_result, tuple):
                score = match_result[0] if len(match_result) > 0 else None
                reasoning = match_result[1] if len(match_result) > 1 else None
            else:
                score = getattr(match_result, "score", getattr(match_result, "passed", None))
                reasoning = getattr(match_result, "reasoning", getattr(match_result, "comment", getattr(match_result, "explanation", None)))
            
            # Convert match_result to JSON-serializable format before storing
            result_entry["match_result"] = _make_json_serializable(match_result)
            result_entry["match_score"] = score
            result_entry["reasoning"] = reasoning
            
            # Check if score meets threshold (for continuous scores)
            if score is not None:
                if isinstance(score, (int, float)):
                    passed = score >= threshold
                else:
                    passed = bool(score)
            else:
                passed = False
            
            if not passed:
                failures.append(
                    f"Trajectory LLM score={score:.4f if isinstance(score, (int, float)) else score} < threshold {threshold} "
                    f"for case {case_id}"
                )
            
            print(f"   ✅ Match result score: {score}")
            if reasoning:
                print(f"   📝 Reasoning: {reasoning[:200]}..." if len(str(reasoning)) > 200 else f"   📝 Reasoning: {reasoning}")
            
        except Exception as e:
            error_msg = f"Error evaluating trajectory for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["match_score"] = None
            failures.append(error_msg)
            print(f"   ❌ Error: {error_msg}")
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file using reusable utility
        log_result_entry(log_file, result_entry)
        
        # Add to file-level cumulative results
        add_to_file_level_results(result_entry)
    
    # Note: CSV export is now cumulative and written after all tests complete
    # Individual test CSV export removed - use cumulative CSV instead
    
    # Calculate success count (score >= threshold)
    success_count = sum(1 for r in results if r.get("match_score") is not None and (
        (isinstance(r.get("match_score"), (int, float)) and r.get("match_score") >= threshold) or
        (not isinstance(r.get("match_score"), (int, float)) and bool(r.get("match_score")))
    ))
    
    # Print summary using reusable utility
    print_test_summary(
        test_name="Trajectory Single Query LLM",
        total_cases=len(test_cases),
        success_count=success_count,
        failures=failures,
        log_file=log_file,
        csv_file=None  # CSV is now cumulative, written at end
    )
    
    # Print detailed results using reusable utility
    print_detailed_results(
        results=results,
        title="TRAJECTORY SINGLE QUERY LLM RESULTS - WEATHER AGENT"
    )
    
    # Assert test results using reusable utility
    assert_test_results(
        results=results,
        test_cases=test_cases,
        failures=failures,
        success_field="match_score",
        threshold=threshold
    )
    
    print("\n✅ Single query trajectory LLM evaluation completed")
    print("="*80 + "\n")


def test_weather_agent_trajectory_multiple_queries_llm(weather_client):
    """
    Test LLM-based trajectory evaluation for multiple weather queries loaded from trajectoryStrict.jsonl.
    """
    print("\n" + "="*80)
    print("TEST: test_weather_agent_trajectory_multiple_queries_llm")
    print("="*80)
    
    # Load test cases from JSONL file
    test_data_path = Path(__file__).parent.parent.parent.parent / "testData" / "agentic" / "trajectoryStrict.jsonl"
    
    test_cases = []
    if test_data_path.exists():
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                    # Filter for multiple_queries test type
                    if case.get("test_type") == "multiple_queries":
                        test_cases.append((
                            case.get("query", ""),
                            case.get("id", "unknown")
                        ))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    else:
        pytest.skip(f"Test data file not found: {test_data_path}")
    
    if not test_cases:
        pytest.skip("No test cases found in trajectoryStrict.jsonl")
    
    print(f"📋 Loaded {len(test_cases)} test cases from {test_data_path.name}")
    
    # Setup logging directories using reusable utility
    evidence_dir, reports_dir, log_file = setup_logging_directories("trajectory_multiple_queries_llm_test")
    
    # Collect all failures and results
    failures = []
    results = []
    threshold = 0.7  # Score threshold for LLM evaluation
    
    for i, (query, case_id) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i} ({case_id}): '{query}'")
        
        # Initialize result entry using reusable utility
        result_entry = create_result_entry(
            case_id=case_id,
            query=query,
            actual_steps=None,
            match_result=None,
            match_score=None,
            response=None,
            tools_called=[],
            thread_id=None,
            reasoning=None
        )
        
        try:
            # Generate unique thread ID for this test case
            thread_id = str(uuid.uuid4())
            weather_client.reset_conversation(thread_id)
            result_entry["thread_id"] = thread_id
            
            # Run agent with trace capture
            print(f"\n🔄 Running weather agent (thread: {thread_id[:8]}...)")
            response = weather_client.ask(query, capture_trace=True)
            result_entry["response"] = response.text
            
            # Get tool calls information
            tools_called = response.tools_called if hasattr(response, 'tools_called') else []
            result_entry["tools_called"] = tools_called
            
            # Extract trajectory
            if response.trace_info:
                extracted_trajectory = response.trace_info
            else:
                extracted_trajectory = extract_langgraph_trajectory_from_thread(
                    weather_client.agent,
                    weather_client.config
                )
            
            outputs = extracted_trajectory.get('outputs', {})
            if isinstance(outputs, dict):
                actual_steps = outputs.get('steps', [])
            else:
                actual_steps = extracted_trajectory.get('steps', [])
            
            result_entry["actual_steps"] = actual_steps
            
            print(f"   📋 Actual steps: {actual_steps}")
            
            # Perform LLM-based trajectory evaluation (no reference_trajectory needed)
            # Ensure inputs and outputs are properly formatted before calling
            trajectory_outputs = extracted_trajectory.get("outputs")
            trajectory_inputs = extracted_trajectory.get("inputs")
            
            if trajectory_outputs is None:
                raise ValueError(f"Extracted trajectory outputs is None for case {case_id}")
            
            # Prepare inputs - the LLM evaluator expects inputs to be a list matching the length of outputs["results"]
            if trajectory_inputs is None:
                results_list = trajectory_outputs.get("results", [])
                steps_list = trajectory_outputs.get("steps", [])
                required_length = len(results_list) if results_list else (len(steps_list) if steps_list else 1)
                trajectory_inputs = [query] * required_length if required_length > 0 else [query]
            elif not isinstance(trajectory_inputs, list):
                results_list = trajectory_outputs.get("results", [])
                required_length = len(results_list) if results_list else 1
                trajectory_inputs = [trajectory_inputs] * required_length if required_length > 1 else [trajectory_inputs]
            
            # Ensure inputs length matches outputs["results"] length
            results_list = trajectory_outputs.get("results", [])
            if results_list and len(trajectory_inputs) != len(results_list):
                if len(trajectory_inputs) < len(results_list):
                    trajectory_inputs.extend([query] * (len(results_list) - len(trajectory_inputs)))
                else:
                    trajectory_inputs = trajectory_inputs[:len(results_list)]
            
            # Serialize trajectory_inputs and trajectory_outputs to handle HumanMessage objects
            # before passing to LLM evaluator
            serialized_inputs = _make_json_serializable(trajectory_inputs)
            serialized_outputs = _make_json_serializable(trajectory_outputs)
            
            print(f"\n🔍 Performing LLM-based trajectory evaluation...")
            match_result = graph_trajectory_llm_match(
                inputs=serialized_inputs,
                outputs=serialized_outputs,
                # No reference_outputs - LLM evaluator doesn't need it
            )
            
            # Extract score and reasoning from match_result
            if isinstance(match_result, dict):
                score = match_result.get("score", match_result.get("passed", None))
                reasoning = match_result.get("reasoning", match_result.get("comment", match_result.get("explanation", None)))
            elif isinstance(match_result, tuple):
                score = match_result[0] if len(match_result) > 0 else None
                reasoning = match_result[1] if len(match_result) > 1 else None
            else:
                score = getattr(match_result, "score", getattr(match_result, "passed", None))
                reasoning = getattr(match_result, "reasoning", getattr(match_result, "comment", getattr(match_result, "explanation", None)))
            
            # Convert match_result to JSON-serializable format before storing
            result_entry["match_result"] = _make_json_serializable(match_result)
            result_entry["match_score"] = score
            result_entry["reasoning"] = reasoning
            
            # Check if score meets threshold (for continuous scores)
            if score is not None:
                if isinstance(score, (int, float)):
                    passed = score >= threshold
                else:
                    passed = bool(score)
            else:
                passed = False
            
            if not passed:
                failures.append(
                    f"Trajectory LLM score={score:.4f if isinstance(score, (int, float)) else score} < threshold {threshold} "
                    f"for case {case_id}"
                )
            
            print(f"   ✅ Match result score: {score}")
            if reasoning:
                print(f"   📝 Reasoning: {reasoning[:200]}..." if len(str(reasoning)) > 200 else f"   📝 Reasoning: {reasoning}")
            
        except Exception as e:
            error_msg = f"Error evaluating trajectory for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["match_score"] = None
            failures.append(error_msg)
            print(f"   ❌ Error: {error_msg}")
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file using reusable utility
        log_result_entry(log_file, result_entry)
        
        # Add to file-level cumulative results
        add_to_file_level_results(result_entry)
    
    # Note: CSV export is now cumulative and written after all tests complete
    # Individual test CSV export removed - use cumulative CSV instead
    
    # Calculate success count (score >= threshold)
    success_count = sum(1 for r in results if r.get("match_score") is not None and (
        (isinstance(r.get("match_score"), (int, float)) and r.get("match_score") >= threshold) or
        (not isinstance(r.get("match_score"), (int, float)) and bool(r.get("match_score")))
    ))
    
    # Print summary using reusable utility
    print_test_summary(
        test_name="Trajectory Multiple Queries LLM",
        total_cases=len(test_cases),
        success_count=success_count,
        failures=failures,
        log_file=log_file,
        csv_file=None  # CSV is now cumulative, written at end
    )
    
    # Print detailed results using reusable utility
    print_detailed_results(
        results=results,
        title="TRAJECTORY MULTIPLE QUERIES LLM RESULTS - WEATHER AGENT"
    )
    
    # Assert test results using reusable utility
    assert_test_results(
        results=results,
        test_cases=test_cases,
        failures=failures,
        success_field="match_score",
        threshold=threshold
    )
    
    print("\n✅ Multiple query trajectory LLM evaluation completed")
    print("="*80 + "\n")


def test_weather_agent_trajectory_without_tool_call_llm(weather_client):
    """
    Test LLM-based trajectory evaluation for queries that might not require tool calls, loaded from trajectoryStrict.jsonl.
    """
    print("\n" + "="*80)
    print("TEST: test_weather_agent_trajectory_without_tool_call_llm")
    print("="*80)
    
    # Load test cases from JSONL file (filter for without_tool_call type)
    test_data_path = Path(__file__).parent.parent.parent.parent / "testData" / "agentic" / "trajectoryStrict.jsonl"
    
    test_cases = []
    if test_data_path.exists():
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                    # Filter for without_tool_call test type
                    if case.get("test_type") == "without_tool_call":
                        test_cases.append((
                            case.get("query", ""),
                            case.get("id", "unknown")
                        ))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    else:
        pytest.skip(f"Test data file not found: {test_data_path}")
    
    if not test_cases:
        pytest.skip("No without_tool_call test cases found in trajectoryStrict.jsonl")
    
    print(f"📋 Loaded {len(test_cases)} test cases from {test_data_path.name}")
    
    # Setup logging directories using reusable utility
    evidence_dir, reports_dir, log_file = setup_logging_directories("trajectory_without_tool_call_llm_test")
    
    # Collect all failures and results
    failures = []
    results = []
    threshold = 0.7  # Score threshold for LLM evaluation
    
    for i, (query, case_id) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i} ({case_id}): '{query}' (may not require tool call)")
        
        # Initialize result entry using reusable utility
        result_entry = create_result_entry(
            case_id=case_id,
            query=query,
            actual_steps=None,
            match_result=None,
            match_score=None,
            response=None,
            tools_called=[],
            thread_id=None,
            reasoning=None
        )
        
        try:
            # Generate unique thread ID for this test case
            thread_id = str(uuid.uuid4())
            weather_client.reset_conversation(thread_id)
            result_entry["thread_id"] = thread_id
            
            # Run agent with trace capture
            print(f"\n🔄 Running weather agent (thread: {thread_id[:8]}...)")
            response = weather_client.ask(query, capture_trace=True)
            result_entry["response"] = response.text
            
            # Get tool calls information
            tools_called = response.tools_called if hasattr(response, 'tools_called') else []
            result_entry["tools_called"] = tools_called
            
            # Extract trajectory
            if response.trace_info:
                extracted_trajectory = response.trace_info
            else:
                extracted_trajectory = extract_langgraph_trajectory_from_thread(
                    weather_client.agent,
                    weather_client.config
                )
            
            outputs = extracted_trajectory.get('outputs', {})
            if isinstance(outputs, dict):
                actual_steps = outputs.get('steps', [])
            else:
                actual_steps = extracted_trajectory.get('steps', [])
            
            result_entry["actual_steps"] = actual_steps
            
            print(f"   📋 Actual steps: {actual_steps}")
            
            # Perform LLM-based trajectory evaluation (no reference_trajectory needed)
            # Ensure inputs and outputs are properly formatted before calling
            trajectory_outputs = extracted_trajectory.get("outputs")
            trajectory_inputs = extracted_trajectory.get("inputs")
            
            if trajectory_outputs is None:
                raise ValueError(f"Extracted trajectory outputs is None for case {case_id}")
            
            # Prepare inputs - the LLM evaluator expects inputs to be a list matching the length of outputs["results"]
            if trajectory_inputs is None:
                results_list = trajectory_outputs.get("results", [])
                steps_list = trajectory_outputs.get("steps", [])
                required_length = len(results_list) if results_list else (len(steps_list) if steps_list else 1)
                trajectory_inputs = [query] * required_length if required_length > 0 else [query]
            elif not isinstance(trajectory_inputs, list):
                results_list = trajectory_outputs.get("results", [])
                required_length = len(results_list) if results_list else 1
                trajectory_inputs = [trajectory_inputs] * required_length if required_length > 1 else [trajectory_inputs]
            
            # Ensure inputs length matches outputs["results"] length
            results_list = trajectory_outputs.get("results", [])
            if results_list and len(trajectory_inputs) != len(results_list):
                if len(trajectory_inputs) < len(results_list):
                    trajectory_inputs.extend([query] * (len(results_list) - len(trajectory_inputs)))
                else:
                    trajectory_inputs = trajectory_inputs[:len(results_list)]
            
            # Serialize trajectory_inputs and trajectory_outputs to handle HumanMessage objects
            # before passing to LLM evaluator
            serialized_inputs = _make_json_serializable(trajectory_inputs)
            serialized_outputs = _make_json_serializable(trajectory_outputs)
            
            print(f"\n🔍 Performing LLM-based trajectory evaluation...")
            match_result = graph_trajectory_llm_match(
                inputs=serialized_inputs,
                outputs=serialized_outputs,
                # No reference_outputs - LLM evaluator doesn't need it
            )
            
            # Extract score and reasoning from match_result
            if isinstance(match_result, dict):
                score = match_result.get("score", match_result.get("passed", None))
                reasoning = match_result.get("reasoning", match_result.get("comment", match_result.get("explanation", None)))
            elif isinstance(match_result, tuple):
                score = match_result[0] if len(match_result) > 0 else None
                reasoning = match_result[1] if len(match_result) > 1 else None
            else:
                score = getattr(match_result, "score", getattr(match_result, "passed", None))
                reasoning = getattr(match_result, "reasoning", getattr(match_result, "comment", getattr(match_result, "explanation", None)))
            
            # Convert match_result to JSON-serializable format before storing
            result_entry["match_result"] = _make_json_serializable(match_result)
            result_entry["match_score"] = score
            result_entry["reasoning"] = reasoning
            
            # Note: For queries without tool calls, score may vary
            # So we log it but don't fail the test unless score is very low
            if score is not None:
                if isinstance(score, (int, float)):
                    if score < 0.5:  # Very low threshold for without_tool_call cases
                        failures.append(
                            f"Trajectory LLM score={score:.4f} < threshold 0.5 for case {case_id}"
                        )
                else:
                    if not bool(score):
                        failures.append(
                            f"Trajectory LLM score=False for case {case_id}"
                        )
            
            print(f"   ✅ Match result score: {score}")
            if reasoning:
                print(f"   📝 Reasoning: {reasoning[:200]}..." if len(str(reasoning)) > 200 else f"   📝 Reasoning: {reasoning}")
            
        except Exception as e:
            error_msg = f"Error evaluating trajectory for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["match_score"] = None
            failures.append(error_msg)
            print(f"   ❌ Error: {error_msg}")
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file using reusable utility
        log_result_entry(log_file, result_entry)
        
        # Add to file-level cumulative results
        add_to_file_level_results(result_entry)
    
    # Note: CSV export is now cumulative and written after all tests complete
    # Individual test CSV export removed - use cumulative CSV instead
    
    # Calculate success count (score >= 0.5 for without_tool_call cases)
    success_count = sum(1 for r in results if r.get("match_score") is not None and (
        (isinstance(r.get("match_score"), (int, float)) and r.get("match_score") >= 0.5) or
        (not isinstance(r.get("match_score"), (int, float)) and bool(r.get("match_score")))
    ))
    
    # Print summary using reusable utility
    print_test_summary(
        test_name="Trajectory Without Tool Call LLM",
        total_cases=len(test_cases),
        success_count=success_count,
        failures=failures,
        log_file=log_file,
        csv_file=None  # CSV is now cumulative, written at end
    )
    
    # Print detailed results using reusable utility
    print_detailed_results(
        results=results,
        title="TRAJECTORY WITHOUT TOOL CALL LLM RESULTS - WEATHER AGENT"
    )
    
    # Assert test results using reusable utility (only assert on errors, not low scores)
    # For this test, we only fail on actual errors, not low scores
    assert len(results) == len(test_cases), "All test cases should be evaluated"
    
    # Only assert on actual errors, not low scores
    if failures:
        failure_message = (
            f"\n\nFound {len(failures)} error(s):\n" + 
            "\n".join(f"  - {f}" for f in failures[:50])
        )
        if len(failures) > 50:
            failure_message += f"\n  ... and {len(failures) - 50} more errors"
        pytest.fail(failure_message)
    
    print("\n✅ Without tool call trajectory LLM evaluation completed")
    print("="*80 + "\n")


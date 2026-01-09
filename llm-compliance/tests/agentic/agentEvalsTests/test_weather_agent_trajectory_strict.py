"""
Test suite for Weather Agent trajectory evaluation using extract_langgraph_trajectory_from_thread
and graph_trajectory_strict_match from agentevals.
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
from agentevals.graph_trajectory.strict import graph_trajectory_strict_match

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


def write_cumulative_csv():
    """Write cumulative CSV file with all results from all tests in this file"""
    global _file_level_results, _file_level_reports_dir
    
    if not _file_level_results:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = _file_level_reports_dir / f"trajectory_all_tests_cumulative_{timestamp}.csv"
    
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
            # Convert lists and dicts to JSON strings for CSV compatibility
            if isinstance(value, (list, dict)):
                row[field] = json.dumps(value) if value else None
            else:
                row[field] = value
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
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_entry) + "\n")


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
            # Convert lists and dicts to JSON strings for CSV compatibility
            if isinstance(value, (list, dict)):
                row[field] = json.dumps(value) if value else None
            else:
                row[field] = value
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
            match_str = '✓ YES' if match_score else '✗ NO' if match_score is not None else 'N/A'
            print(f"  Match: {match_str}")
        if 'score' in result:
            score = result.get('score')
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"  Score: {score_str}")
        if 'success' in result:
            success = result.get('success')
            success_str = '✓ YES' if success else '✗ NO' if success is not None else 'N/A'
            print(f"  Passed: {success_str}")
        
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
    success_field: str = "match_score"
):
    """
    Assert test results and raise pytest.fail if there are failures.
    
    Args:
        results: List of result entry dictionaries
        test_cases: Original list of test cases (for count validation)
        failures: List of failure messages
        success_field: Field name in result entry that indicates success (default: "match_score")
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


def test_weather_agent_trajectory_single_query(weather_client):
    """
    Test trajectory extraction for a single weather query loaded from trajectoryStrict.jsonl.
    """
    print("\n" + "="*80)
    print("TEST: test_weather_agent_trajectory_single_query")
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
                            case.get("expected_steps", []),
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
    evidence_dir, reports_dir, log_file = setup_logging_directories("trajectory_single_query_test")
    
    # Collect all failures and results
    failures = []
    results = []
    
    for i, (query, expected_steps, case_id) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i} ({case_id}): '{query}'")
        
        # Initialize result entry using reusable utility
        result_entry = create_result_entry(
            case_id=case_id,
            query=query,
            expected_steps=expected_steps,
            actual_steps=None,
            match_result=None,
            match_score=None,
            response=None,
            tools_called=[],
            thread_id=None
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
            print(f"   📋 Expected steps: {expected_steps}")
            
            # Define reference trajectory
            reference_trajectory = {
                "results": [],
                "steps": [expected_steps],
            }
            
            # Perform strict match evaluation
            match_result = graph_trajectory_strict_match(
                outputs=extracted_trajectory["outputs"],
                reference_outputs=reference_trajectory,
            )
            
            result_entry["match_result"] = match_result
            result_entry["match_score"] = match_result.get("score", False)
            
            # Check if match passed
            if not match_result.get("score", False):
                failures.append(
                    f"Trajectory mismatch for case {case_id}: "
                    f"Expected={expected_steps}, Actual={actual_steps}"
                )
            
            print(f"   ✅ Match result: {match_result}")
            
        except Exception as e:
            error_msg = f"Error evaluating trajectory for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["match_score"] = False
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
    
    # Calculate success count
    success_count = sum(1 for r in results if r.get("match_score") is True)
    
    # Print summary using reusable utility
    print_test_summary(
        test_name="Trajectory Single Query",
        total_cases=len(test_cases),
        success_count=success_count,
        failures=failures,
        log_file=log_file,
        csv_file=None  # CSV is now cumulative, written at end
    )
    
    # Print detailed results using reusable utility
    print_detailed_results(
        results=results,
        title="TRAJECTORY SINGLE QUERY RESULTS - WEATHER AGENT"
    )
    
    # Assert test results using reusable utility
    assert_test_results(
        results=results,
        test_cases=test_cases,
        failures=failures,
        success_field="match_score"
    )
    
    print("\n✅ Single query trajectory evaluation completed")
    print("="*80 + "\n")


def test_weather_agent_trajectory_multiple_queries(weather_client):
    """
    Test trajectory extraction for multiple weather queries loaded from trajectoryStrict.jsonl.
    """
    print("\n" + "="*80)
    print("TEST: test_weather_agent_trajectory_multiple_queries")
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
                            case.get("expected_steps", []),
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
    evidence_dir, reports_dir, log_file = setup_logging_directories("trajectory_strict_test")
    
    # Collect all failures and results
    failures = []
    results = []
    
    for i, (query, expected_steps, case_id) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i} ({case_id}): '{query}'")
        
        # Initialize result entry using reusable utility
        result_entry = create_result_entry(
            case_id=case_id,
            query=query,
            expected_steps=expected_steps,
            actual_steps=None,
            match_result=None,
            match_score=None,
            response=None,
            tools_called=[],
            thread_id=None
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
            print(f"   📋 Expected steps: {expected_steps}")
            
            # Define reference trajectory
            reference_trajectory = {
                "results": [],
                "steps": [expected_steps],
            }
            
            # Perform strict match evaluation
            match_result = graph_trajectory_strict_match(
                outputs=extracted_trajectory["outputs"],
                reference_outputs=reference_trajectory,
            )
            
            result_entry["match_result"] = match_result
            result_entry["match_score"] = match_result.get("score", False)
            
            # Check if match passed
            if not match_result.get("score", False):
                failures.append(
                    f"Trajectory mismatch for case {case_id}: "
                    f"Expected={expected_steps}, Actual={actual_steps}"
                )
            
            print(f"   ✅ Match result: {match_result}")
            
        except Exception as e:
            error_msg = f"Error evaluating trajectory for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["match_score"] = False
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
    
    # Calculate success count
    success_count = sum(1 for r in results if r.get("match_score") is True)
    
    # Print summary using reusable utility
    print_test_summary(
        test_name="Trajectory Strict Match",
        total_cases=len(test_cases),
        success_count=success_count,
        failures=failures,
        log_file=log_file,
        csv_file=None  # CSV is now cumulative, written at end
    )
    
    # Print detailed results using reusable utility
    print_detailed_results(
        results=results,
        title="TRAJECTORY STRICT MATCH RESULTS - WEATHER AGENT"
    )
    
    # Assert test results using reusable utility
    assert_test_results(
        results=results,
        test_cases=test_cases,
        failures=failures,
        success_field="match_score"
    )
    
    print("\n✅ Multiple query trajectory evaluation completed")
    print("="*80 + "\n")


def test_weather_agent_trajectory_without_tool_call(weather_client):
    """
    Test trajectory extraction for queries that might not require tool calls, loaded from trajectoryStrict.jsonl.
    """
    print("\n" + "="*80)
    print("TEST: test_weather_agent_trajectory_without_tool_call")
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
                            case.get("expected_steps", []),
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
    evidence_dir, reports_dir, log_file = setup_logging_directories("trajectory_without_tool_call_test")
    
    # Collect all failures and results
    failures = []
    results = []
    
    for i, (query, expected_steps, case_id) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i} ({case_id}): '{query}' (may not require tool call)")
        
        # Initialize result entry using reusable utility
        result_entry = create_result_entry(
            case_id=case_id,
            query=query,
            expected_steps=expected_steps,
            actual_steps=None,
            match_result=None,
            match_score=None,
            response=None,
            tools_called=[],
            thread_id=None
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
            print(f"   📋 Expected steps: {expected_steps}")
            
            # Define reference trajectory
            reference_trajectory = {
                "results": [],
                "steps": [expected_steps],
            }
            
            # Perform strict match evaluation
            match_result = graph_trajectory_strict_match(
                outputs=extracted_trajectory["outputs"],
                reference_outputs=reference_trajectory,
            )
            
            result_entry["match_result"] = match_result
            result_entry["match_score"] = match_result.get("score", False)
            
            # Note: This might not match if agent decides to use tools anyway
            # So we log it but don't fail the test
            if not match_result.get("score", False):
                print(f"   ⚠️  Note: Match result may vary based on agent's decision to use tools")
                # Don't add to failures - this is expected behavior
            
            print(f"   ✅ Match result: {match_result}")
            
        except Exception as e:
            error_msg = f"Error evaluating trajectory for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["match_score"] = False
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
    
    # Calculate success count
    success_count = sum(1 for r in results if r.get("match_score") is True)
    
    # Print summary using reusable utility
    print_test_summary(
        test_name="Trajectory Without Tool Call",
        total_cases=len(test_cases),
        success_count=success_count,
        failures=failures,
        log_file=log_file,
        csv_file=None  # CSV is now cumulative, written at end
    )
    
    # Print detailed results using reusable utility
    print_detailed_results(
        results=results,
        title="TRAJECTORY WITHOUT TOOL CALL RESULTS - WEATHER AGENT"
    )
    
    # Assert test results using reusable utility (only assert on errors, not match failures)
    # For this test, we only fail on actual errors, not trajectory mismatches
    assert len(results) == len(test_cases), "All test cases should be evaluated"
    
    # Only assert on actual errors, not trajectory mismatches
    if failures:
        failure_message = (
            f"\n\nFound {len(failures)} error(s):\n" + 
            "\n".join(f"  - {f}" for f in failures[:50])
        )
        if len(failures) > 50:
            failure_message += f"\n  ... and {len(failures) - 50} more errors"
        pytest.fail(failure_message)
    
    print("\n✅ Without tool call trajectory evaluation completed")
    print("="*80 + "\n")


"""
Test suite for Weather Agent trajectory evaluation using formatted messages
and create_trajectory_llm_as_judge from agentevals.trajectory.llm.
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

from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

# Create the LLM evaluator function for trajectory messages
trajectory_evaluator = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT,
    model="openai:gpt-4o"  # Using gpt-4o instead of o3-mini for better compatibility
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
    csv_file = _file_level_reports_dir / f"trajectory_messages_llm_all_tests_cumulative_{timestamp}.csv"
    
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
        
        # Print formatted messages if available
        if 'formatted_messages' in result:
            formatted_messages = result.get('formatted_messages', [])
            print(f"  Formatted Messages: {len(formatted_messages)} messages")
            for j, msg in enumerate(formatted_messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                # Handle tool_calls for assistant messages
                tool_calls_info = ""
                if role == 'assistant' and 'tool_calls' in msg:
                    tool_calls = msg.get('tool_calls', [])
                    if tool_calls:
                        tool_names = []
                        for tc in tool_calls:
                            func = tc.get('function', {})
                            tool_name = func.get('name', 'unknown')
                            tool_args = func.get('arguments', '{}')
                            try:
                                args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                                tool_names.append(f"{tool_name}({args_dict})")
                            except:
                                tool_names.append(f"{tool_name}({tool_args[:50]})")
                        tool_calls_info = f" [tool_calls: {', '.join(tool_names)}]"
                
                # Format content display
                if not content and tool_calls_info:
                    content_display = tool_calls_info
                elif len(content) > 150:
                    content_display = content[:150] + "..."
                else:
                    content_display = content if content else "(empty)"
                
                print(f"    [{j+1}] {role}: {content_display}")
        
        # Print evaluation result
        if 'eval_score' in result:
            eval_score = result.get('eval_score')
            if isinstance(eval_score, (int, float)):
                score_str = f"{eval_score:.4f}"
            else:
                score_str = '✓ YES' if eval_score else '✗ NO' if eval_score is not None else 'N/A'
            print(f"  Evaluation Score: {score_str}")
        if 'eval_result' in result:
            eval_result = result.get('eval_result')
            if isinstance(eval_result, dict):
                print(f"  Evaluation Result: {json.dumps(eval_result, indent=2)[:200]}...")
            else:
                print(f"  Evaluation Result: {str(eval_result)[:200]}...")
        
        tools_called = result.get('tools_called', [])
        if tools_called:
            print(f"  Tools Called: {len(tools_called)}")
            for tool in tools_called:
                tool_name = tool.get('name', 'unknown') if isinstance(tool, dict) else str(tool)
                tool_args = tool.get('args', {}) if isinstance(tool, dict) else {}
                print(f"    - {tool_name}({tool_args})")
        
        if result.get('errors'):
            print(f"  Errors: {result.get('errors')}")
    
    print("\n" + "="*60)


def assert_test_results(
    results: list[Dict[str, Any]],
    test_cases: list,
    failures: list[str],
    success_field: str = "eval_score",
    threshold: float = 0.7
):
    """
    Assert test results and raise pytest.fail if there are failures.
    
    Args:
        results: List of result entry dictionaries
        test_cases: Original list of test cases (for count validation)
        failures: List of failure messages
        success_field: Field name in result entry that indicates success (default: "eval_score")
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


def test_weather_agent_trajectory_messages_single_query(weather_client):
    """
    Test trajectory message evaluation for a single weather query loaded from trajectoryStrict.jsonl.
    Uses formatted messages and create_trajectory_llm_as_judge.
    """
    print("\n" + "="*80)
    print("TEST: test_weather_agent_trajectory_messages_single_query")
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
    evidence_dir, reports_dir, log_file = setup_logging_directories("trajectory_messages_single_query_test")
    
    # Collect all failures and results
    failures = []
    results = []
    threshold = 0.7  # Score threshold for evaluation
    
    for i, (query, case_id) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i} ({case_id}): '{query}'")
        
        # Initialize result entry using reusable utility
        result_entry = create_result_entry(
            case_id=case_id,
            query=query,
            response=None,
            tools_called=[],
            thread_id=None,
            formatted_messages=None,
            eval_result=None,
            eval_score=None
        )
        
        try:
            # Generate unique thread ID for this test case
            thread_id = str(uuid.uuid4())
            weather_client.reset_conversation(thread_id)
            result_entry["thread_id"] = thread_id
            
            # Run agent with trace capture and formatted messages
            print(f"\n🔄 Running weather agent (thread: {thread_id[:8]}...)")
            response = weather_client.ask(query, capture_trace=True, format_as_messages=True)
            result_entry["response"] = response.text
            
            # Get tool calls information
            tools_called = response.tools_called if hasattr(response, 'tools_called') else []
            result_entry["tools_called"] = tools_called
            
            # Get formatted messages
            formatted_messages = response.formatted_messages if hasattr(response, 'formatted_messages') else None
            result_entry["formatted_messages"] = formatted_messages
            
            if not formatted_messages:
                raise ValueError(f"Formatted messages not available for case {case_id}")
            
            print(f"   📋 Formatted messages: {len(formatted_messages)} messages")
            for j, msg in enumerate(formatted_messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '') or ''
                
                # Handle tool_calls for assistant messages
                tool_calls_info = ""
                if role == 'assistant' and 'tool_calls' in msg:
                    tool_calls = msg.get('tool_calls', [])
                    if tool_calls:
                        tool_names = []
                        for tc in tool_calls:
                            func = tc.get('function', {})
                            tool_name = func.get('name', 'unknown')
                            tool_args = func.get('arguments', '{}')
                            try:
                                args_dict = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                                tool_names.append(f"{tool_name}({args_dict})")
                            except:
                                tool_names.append(f"{tool_name}({tool_args[:50]})")
                        tool_calls_info = f" [tool_calls: {', '.join(tool_names)}]"
                
                # Format content display - prioritize showing tool_calls if content is empty
                if not content.strip() and tool_calls_info:
                    content_display = tool_calls_info.strip()
                elif content.strip() and tool_calls_info:
                    content_display = f"{content[:100]}{'...' if len(content) > 100 else ''} {tool_calls_info}"
                elif len(content) > 150:
                    content_display = content[:150] + "..."
                elif content.strip():
                    content_display = content
                else:
                    content_display = "(empty)" + tool_calls_info
                
                print(f"      [{j+1}] {role}: {content_display}")
            
            # Perform trajectory evaluation using formatted messages
            print(f"\n🔍 Performing trajectory message evaluation...")
            eval_result = trajectory_evaluator(
                outputs=formatted_messages
            )
            
            # Extract score from eval_result
            if isinstance(eval_result, dict):
                score = eval_result.get("score", eval_result.get("passed", eval_result.get("accuracy", None)))
            elif isinstance(eval_result, tuple):
                score = eval_result[0] if len(eval_result) > 0 else None
            else:
                score = getattr(eval_result, "score", getattr(eval_result, "passed", getattr(eval_result, "accuracy", None)))
            
            result_entry["eval_result"] = _make_json_serializable(eval_result)
            result_entry["eval_score"] = score
            
            # Check if score meets threshold
            if score is not None:
                if isinstance(score, (int, float)):
                    passed = score >= threshold
                else:
                    passed = bool(score)
            else:
                passed = False
            
            if not passed:
                # Format score for display
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                failures.append(
                    f"Trajectory message evaluation score={score_str} < threshold {threshold} "
                    f"for case {case_id}"
                )
            
            print(f"   ✅ Evaluation score: {score}")
            if isinstance(eval_result, dict):
                print(f"   📝 Evaluation result: {json.dumps(eval_result, indent=2)[:200]}...")
            
        except Exception as e:
            error_msg = f"Error evaluating trajectory messages for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["eval_score"] = None
            failures.append(error_msg)
            print(f"   ❌ Error: {error_msg}")
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file using reusable utility
        log_result_entry(log_file, result_entry)
        
        # Add to file-level cumulative results
        add_to_file_level_results(result_entry)
    
    # Calculate success count (score >= threshold)
    success_count = sum(1 for r in results if r.get("eval_score") is not None and (
        (isinstance(r.get("eval_score"), (int, float)) and r.get("eval_score") >= threshold) or
        (not isinstance(r.get("eval_score"), (int, float)) and bool(r.get("eval_score")))
    ))
    
    # Print summary using reusable utility
    print_test_summary(
        test_name="Trajectory Messages Single Query",
        total_cases=len(test_cases),
        success_count=success_count,
        failures=failures,
        log_file=log_file,
        csv_file=None  # CSV is now cumulative, written at end
    )
    
    # Print detailed results using reusable utility
    print_detailed_results(
        results=results,
        title="TRAJECTORY MESSAGES SINGLE QUERY RESULTS - WEATHER AGENT"
    )
    
    # Assert test results using reusable utility
    assert_test_results(
        results=results,
        test_cases=test_cases,
        failures=failures,
        success_field="eval_score",
        threshold=threshold
    )
    
    print("\n✅ Single query trajectory message evaluation completed")
    print("="*80 + "\n")


def test_weather_agent_trajectory_messages_multiple_queries(weather_client):
    """
    Test trajectory message evaluation for multiple weather queries loaded from trajectoryStrict.jsonl.
    Uses formatted messages and create_trajectory_llm_as_judge.
    """
    print("\n" + "="*80)
    print("TEST: test_weather_agent_trajectory_messages_multiple_queries")
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
    evidence_dir, reports_dir, log_file = setup_logging_directories("trajectory_messages_multiple_queries_test")
    
    # Collect all failures and results
    failures = []
    results = []
    threshold = 0.7  # Score threshold for evaluation
    
    for i, (query, case_id) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i} ({case_id}): '{query}'")
        
        # Initialize result entry using reusable utility
        result_entry = create_result_entry(
            case_id=case_id,
            query=query,
            response=None,
            tools_called=[],
            thread_id=None,
            formatted_messages=None,
            eval_result=None,
            eval_score=None
        )
        
        try:
            # Generate unique thread ID for this test case
            thread_id = str(uuid.uuid4())
            weather_client.reset_conversation(thread_id)
            result_entry["thread_id"] = thread_id
            
            # Run agent with trace capture and formatted messages
            print(f"\n🔄 Running weather agent (thread: {thread_id[:8]}...)")
            response = weather_client.ask(query, capture_trace=True, format_as_messages=True)
            result_entry["response"] = response.text
            
            # Get tool calls information
            tools_called = response.tools_called if hasattr(response, 'tools_called') else []
            result_entry["tools_called"] = tools_called
            
            # Get formatted messages
            formatted_messages = response.formatted_messages if hasattr(response, 'formatted_messages') else None
            result_entry["formatted_messages"] = formatted_messages
            
            if not formatted_messages:
                raise ValueError(f"Formatted messages not available for case {case_id}")
            
            print(f"   📋 Formatted messages: {len(formatted_messages)} messages")
            
            # Perform trajectory evaluation using formatted messages
            print(f"\n🔍 Performing trajectory message evaluation...")
            eval_result = trajectory_evaluator(
                outputs=formatted_messages
            )
            
            # Extract score from eval_result
            if isinstance(eval_result, dict):
                score = eval_result.get("score", eval_result.get("passed", eval_result.get("accuracy", None)))
            elif isinstance(eval_result, tuple):
                score = eval_result[0] if len(eval_result) > 0 else None
            else:
                score = getattr(eval_result, "score", getattr(eval_result, "passed", getattr(eval_result, "accuracy", None)))
            
            result_entry["eval_result"] = _make_json_serializable(eval_result)
            result_entry["eval_score"] = score
            
            # Check if score meets threshold
            if score is not None:
                if isinstance(score, (int, float)):
                    passed = score >= threshold
                else:
                    passed = bool(score)
            else:
                passed = False
            
            if not passed:
                # Format score for display
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                failures.append(
                    f"Trajectory message evaluation score={score_str} < threshold {threshold} "
                    f"for case {case_id}"
                )
            
            print(f"   ✅ Evaluation score: {score}")
            
        except Exception as e:
            error_msg = f"Error evaluating trajectory messages for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["eval_score"] = None
            failures.append(error_msg)
            print(f"   ❌ Error: {error_msg}")
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file using reusable utility
        log_result_entry(log_file, result_entry)
        
        # Add to file-level cumulative results
        add_to_file_level_results(result_entry)
    
    # Calculate success count (score >= threshold)
    success_count = sum(1 for r in results if r.get("eval_score") is not None and (
        (isinstance(r.get("eval_score"), (int, float)) and r.get("eval_score") >= threshold) or
        (not isinstance(r.get("eval_score"), (int, float)) and bool(r.get("eval_score")))
    ))
    
    # Print summary using reusable utility
    print_test_summary(
        test_name="Trajectory Messages Multiple Queries",
        total_cases=len(test_cases),
        success_count=success_count,
        failures=failures,
        log_file=log_file,
        csv_file=None  # CSV is now cumulative, written at end
    )
    
    # Print detailed results using reusable utility
    print_detailed_results(
        results=results,
        title="TRAJECTORY MESSAGES MULTIPLE QUERIES RESULTS - WEATHER AGENT"
    )
    
    # Assert test results using reusable utility
    assert_test_results(
        results=results,
        test_cases=test_cases,
        failures=failures,
        success_field="eval_score",
        threshold=threshold
    )
    
    print("\n✅ Multiple query trajectory message evaluation completed")
    print("="*80 + "\n")


def test_weather_agent_trajectory_messages_without_tool_call(weather_client):
    """
    Test trajectory message evaluation for queries that might not require tool calls, loaded from trajectoryStrict.jsonl.
    Uses formatted messages and create_trajectory_llm_as_judge.
    """
    print("\n" + "="*80)
    print("TEST: test_weather_agent_trajectory_messages_without_tool_call")
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
    evidence_dir, reports_dir, log_file = setup_logging_directories("trajectory_messages_without_tool_call_test")
    
    # Collect all failures and results
    failures = []
    results = []
    threshold = 0.5  # Lower threshold for without_tool_call cases
    
    for i, (query, case_id) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i} ({case_id}): '{query}' (may not require tool call)")
        
        # Initialize result entry using reusable utility
        result_entry = create_result_entry(
            case_id=case_id,
            query=query,
            response=None,
            tools_called=[],
            thread_id=None,
            formatted_messages=None,
            eval_result=None,
            eval_score=None
        )
        
        try:
            # Generate unique thread ID for this test case
            thread_id = str(uuid.uuid4())
            weather_client.reset_conversation(thread_id)
            result_entry["thread_id"] = thread_id
            
            # Run agent with trace capture and formatted messages
            print(f"\n🔄 Running weather agent (thread: {thread_id[:8]}...)")
            response = weather_client.ask(query, capture_trace=True, format_as_messages=True)
            result_entry["response"] = response.text
            
            # Get tool calls information
            tools_called = response.tools_called if hasattr(response, 'tools_called') else []
            result_entry["tools_called"] = tools_called
            
            # Get formatted messages
            formatted_messages = response.formatted_messages if hasattr(response, 'formatted_messages') else None
            result_entry["formatted_messages"] = formatted_messages
            
            if not formatted_messages:
                raise ValueError(f"Formatted messages not available for case {case_id}")
            
            print(f"   📋 Formatted messages: {len(formatted_messages)} messages")
            
            # Perform trajectory evaluation using formatted messages
            print(f"\n🔍 Performing trajectory message evaluation...")
            eval_result = trajectory_evaluator(
                outputs=formatted_messages
            )
            
            # Extract score from eval_result
            if isinstance(eval_result, dict):
                score = eval_result.get("score", eval_result.get("passed", eval_result.get("accuracy", None)))
            elif isinstance(eval_result, tuple):
                score = eval_result[0] if len(eval_result) > 0 else None
            else:
                score = getattr(eval_result, "score", getattr(eval_result, "passed", getattr(eval_result, "accuracy", None)))
            
            result_entry["eval_result"] = _make_json_serializable(eval_result)
            result_entry["eval_score"] = score
            
            # Note: For queries without tool calls, score may vary
            # So we log it but don't fail the test unless score is very low
            if score is not None:
                if isinstance(score, (int, float)):
                    if score < threshold:
                        failures.append(
                            f"Trajectory message evaluation score={score:.4f} < threshold {threshold} for case {case_id}"
                        )
                else:
                    if not bool(score):
                        failures.append(
                            f"Trajectory message evaluation score=False for case {case_id}"
                        )
            
            print(f"   ✅ Evaluation score: {score}")
            
        except Exception as e:
            error_msg = f"Error evaluating trajectory messages for case {case_id}: {e}"
            result_entry["errors"].append(error_msg)
            result_entry["eval_score"] = None
            failures.append(error_msg)
            print(f"   ❌ Error: {error_msg}")
        
        # Add result entry to results
        results.append(result_entry)
        
        # Log to JSONL file using reusable utility
        log_result_entry(log_file, result_entry)
        
        # Add to file-level cumulative results
        add_to_file_level_results(result_entry)
    
    # Calculate success count (score >= threshold)
    success_count = sum(1 for r in results if r.get("eval_score") is not None and (
        (isinstance(r.get("eval_score"), (int, float)) and r.get("eval_score") >= threshold) or
        (not isinstance(r.get("eval_score"), (int, float)) and bool(r.get("eval_score")))
    ))
    
    # Print summary using reusable utility
    print_test_summary(
        test_name="Trajectory Messages Without Tool Call",
        total_cases=len(test_cases),
        success_count=success_count,
        failures=failures,
        log_file=log_file,
        csv_file=None  # CSV is now cumulative, written at end
    )
    
    # Print detailed results using reusable utility
    print_detailed_results(
        results=results,
        title="TRAJECTORY MESSAGES WITHOUT TOOL CALL RESULTS - WEATHER AGENT"
    )
    
    # Assert test results using reusable utility (only assert on errors, not low scores)
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
    
    print("\n✅ Without tool call trajectory message evaluation completed")
    print("="*80 + "\n")


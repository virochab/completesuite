"""
Locust Load Testing Explanation
================================

HOW LOCUST WORKS:
-----------------
1. Locust creates multiple "virtual users" (instances of RAGPDFUser class)
2. Each user runs in its own thread/greenlet
3. Each user executes tasks in a loop until the test stops
4. Tasks are selected randomly based on their weights

THE @task DECORATOR:
--------------------
The @task decorator marks a method as a "task" that users can execute.
You can specify a weight (integer) to control how often each task runs.

Example:
  @task(3)  → This task has weight 3 (runs 3x more often than weight 1)
  @task(2)  → This task has weight 2 (runs 2x more often than weight 1)
  @task(1)  → This task has weight 1 (baseline frequency)
  @task     → No weight = weight 1 (same as @task(1))

Task Selection Probability:
  Total weight = 3 + 2 + 1 + 1 = 7
  - ask_question: 3/7 = 43% chance
  - query_question: 2/7 = 29% chance
  - health_check: 1/7 = 14% chance
  - root_endpoint: 1/7 = 14% chance

EXECUTION FLOW:
---------------
1. User starts → on_start() is called ONCE (health check)
2. User enters task loop:
   a. Randomly selects a task based on weights
   b. Executes the task method
   c. Waits wait_time seconds (1-3 seconds in this case)
   d. Repeats from step 2a until test stops

WAIT_TIME:
----------
wait_time = between(1, 3) means:
- After each task completes, wait a random time between 1-3 seconds
- This simulates "think time" - how long a real user would wait
- Without this, users would hammer the server continuously

EXAMPLE WITH 10 USERS:
-----------------------
- 10 users spawn (each in its own thread)
- Each user runs independently:
  User 1: health → ask → wait 2s → query → wait 1s → health → ...
  User 2: health → query → wait 3s → ask → wait 2s → root → ...
  User 3: health → ask → wait 1s → ask → wait 2s → health → ...
  ... (all running concurrently)

RESPONSE VALIDATION:
--------------------
Using catch_response=True allows you to:
- Mark responses as success/failure manually
- Customize failure messages
- Track specific error conditions
"""

from locust import HttpUser, task, between
import json
import random
from pathlib import Path


def _load_sample_questions():
    """Load sample questions from golden test data JSON file"""
    try:
        # Get the project root (assuming locustLoad.py is in tests/performance/)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        json_path = project_root / "testData" / "synthetic_data" / "singleturn" / "singleturnGoldens" / "20251225_232614.json"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
                # Extract 'input' field from each test case
                questions = [case.get('input', '') for case in test_cases if case.get('input')]
                if questions:
                    return questions
    except Exception as e:
        print(f"Warning: Could not load questions from golden test data: {e}")
    
    # Fallback to default questions if loading fails
    return [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the important findings?",
    ]


class RAGPDFUser(HttpUser):
    """
    Locust user class for load testing the FastAPI RAG PDF Agent endpoints.
    
    This class represents a single "virtual user" that will:
    1. Start with a health check (on_start)
    2. Continuously execute tasks (marked with @task decorator)
    3. Wait 1-3 seconds between each task
    4. Run until the test is stopped
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests (simulates user think time)
    
    # Set timeout to 300 seconds (5 minutes) to match FastAPI timeout
    # RAG queries can take 1-280 seconds, so we need a longer timeout
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set timeout on the HTTP client (default is 60 seconds)
        self.client.timeout = 300  # 5 minutes timeout
    
    # Sample questions for RAG PDF testing
    # Load queries from golden test data to ensure realistic load testing
    # This will be set after the class is defined (see end of file)
    SAMPLE_QUESTIONS = []
    
    def on_start(self):
        """
        Called ONCE when each virtual user starts (before entering the task loop).
        This is like a "setup" method - perfect for initialization or health checks.
        
        Note: This is NOT a task, so it doesn't have @task decorator.
        It runs automatically when the user spawns.
        """
        self.client.get("/health", name="Health Check")
    
    @task(3)  # Weight 3: This task runs 3x more often than tasks with weight 1
    def ask_question(self):
        """Test the /ask endpoint with random questions (weight: 3)"""
        question = random.choice(self.SAMPLE_QUESTIONS)
        payload = {
            "question": question,
            "debug": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        with self.client.post(
            "/ask",
            json=payload,
            headers=headers,
            catch_response=True,
            name="Ask Question"
        ) as response:
            # Handle case where response might be None (connection/timeout errors)
            if response is None:
                response.failure("No response received (connection error or timeout)")
                return
            
            # Check if response has status_code attribute
            if not hasattr(response, 'status_code'):
                response.failure("Response object missing status_code attribute")
                return
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Validate response structure - be more lenient
                    if isinstance(data, dict):
                        # Check if error flag is set (prompt injection blocked)
                        if data.get("error", False):
                            response.failure(f"Request blocked (prompt injection filter): {question[:50]}")
                        elif "answer" in data:
                            # Success - response has answer field (sources might be empty)
                            # Note: We're lenient about sources since they might be empty for some queries
                            response.success()
                        else:
                            # Invalid response structure - log what we got for debugging
                            keys = list(data.keys())
                            response_text = str(data)[:500]  # First 500 chars for debugging
                            response.failure(f"Invalid response structure. Expected 'answer' field. Got keys: {keys}. Response: {response_text}")
                    else:
                        response.failure(f"Response is not a dictionary. Type: {type(data)}, Value: {str(data)[:200]}")
                except json.JSONDecodeError as e:
                    # Log the raw response for debugging
                    raw_text = response.text[:500] if hasattr(response, 'text') else 'No text available'
                    response.failure(f"Invalid JSON response: {str(e)}. Raw response: {raw_text}")
                except Exception as e:
                    response.failure(f"Error parsing response: {str(e)}")
            elif response.status_code == 504:
                # Timeout - mark as failure but with specific message
                response.failure(f"Request timed out after 300 seconds: {response.status_code}")
            elif response.status_code == 503:
                response.failure(f"Service unavailable: {response.status_code} - Agent not initialized")
            elif response.status_code == 400:
                error_text = response.text[:100] if hasattr(response, 'text') else 'No error message'
                response.failure(f"Bad request: {response.status_code} - {error_text}")
            elif response.status_code == 500:
                error_text = response.text[:200] if hasattr(response, 'text') else 'No error message'
                response.failure(f"Server error: {response.status_code} - {error_text}")
            else:
                error_text = response.text[:200] if hasattr(response, 'text') else 'No error message'
                response.failure(f"Failed with status {response.status_code}: {error_text}")
    
    @task(2)  # Weight 2: This task runs 2x more often than tasks with weight 1
    def query_question(self):
        """Test the /query endpoint (alias for /ask) (weight: 2)"""
        question = random.choice(self.SAMPLE_QUESTIONS)
        payload = {
            "question": question,
            "debug": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        with self.client.post(
            "/query",
            json=payload,
            headers=headers,
            catch_response=True,
            name="Query Question"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "answer" in data and "sources" in data:
                        # Check if error flag is set (prompt injection blocked)
                        if data.get("error", False):
                            response.failure(f"Request blocked (prompt injection filter): {question[:50]}")
                        else:
                            response.success()
                    else:
                        response.failure(f"Invalid response structure. Keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                except json.JSONDecodeError as e:
                    response.failure(f"Invalid JSON response: {str(e)}")
                except Exception as e:
                    response.failure(f"Error parsing response: {str(e)}")
            elif response.status_code == 504:
                response.failure(f"Request timed out: {response.status_code}")
            elif response.status_code == 503:
                response.failure(f"Service unavailable: {response.status_code} - Agent not initialized")
            elif response.status_code == 400:
                response.failure(f"Bad request: {response.status_code} - {response.text[:100]}")
            else:
                response.failure(f"Failed with status {response.status_code}: {response.text[:200] if hasattr(response, 'text') else 'No error message'}")
    
    @task(1)  # Weight 1: Baseline frequency (runs less often than weight 2 or 3)
    def health_check(self):
        """Test the /health endpoint (weight: 1)"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="Health Check"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data and "vector_store_initialized" in data:
                        response.success()
                    else:
                        response.failure("Invalid health response structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)  # Weight 1: Baseline frequency
    def root_endpoint(self):
        """Test the root / endpoint (weight: 1)"""
        with self.client.get(
            "/",
            catch_response=True,
            name="Root Endpoint"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Root endpoint failed: {response.status_code}")


# Set SAMPLE_QUESTIONS after the class is defined
RAGPDFUser.SAMPLE_QUESTIONS = _load_sample_questions()

# Usage Examples:
# 
# COMMAND LINE PARAMETERS:
# ------------------------
# -u, --users          Total number of users to spawn (e.g., -u 50 = 50 users)
# -r, --spawn-rate     Number of users to spawn per second (e.g., -r 10 = spawn 10 users/second)
# --run-time           How long to run the test (e.g., --run-time 5m = 5 minutes)
# --host               Target host URL (e.g., --host=http://localhost:8000)
# --headless           Run without web UI (for automation)
#
# Example: -u 50 -r 10 means:
#   - Total users: 50
#   - Spawn rate: 10 users/second
#   - Time to spawn all users: 50 ÷ 10 = 5 seconds
#   - After 5 seconds, all 50 users will be active
#
# Why use spawn-rate?
#   - Gradual ramp-up prevents overwhelming the server
#   - Simulates realistic user arrival patterns
#   - Helps identify performance issues at different load levels
#
# 1. Run indefinitely (until manually stopped with Ctrl+C):
#    locust -f locustLoad.py --host=http://localhost:8000
#
# 2. Run for a specific duration (e.g., 5 minutes):
#    locust -f locustLoad.py --host=http://localhost:8000 -u 50 -r 10 --run-time 5m
#    (Spawns 50 users at a rate of 10 users/second, runs for 5 minutes)
#
# 3. Run for specific duration with web UI:
#    locust -f locustLoad.py --host=http://localhost:8000 -u 100 -r 10 --run-time 10m --web-host=0.0.0.0 --web-port=8089
#
# 4. Headless mode (no web UI) for 30 minutes:
#    locust -f locustLoad.py --host=http://localhost:8000 -u 50 -r 5 --run-time 30m --headless
#
# SAVING RESULTS:
# ---------------
# Locust provides several ways to save test results:
#
# A. Save CSV files (statistics, failures, exceptions):
#    locust -f locustLoad.py --host=http://localhost:8000 -u 50 -r 10 --run-time 5m --headless --csv=reports/locust_results
#    
#    This creates:
#    - reports/locust_results_stats.csv (request statistics)
#    - reports/locust_results_stats_history.csv (time-series data)
#    - reports/locust_results_failures.csv (failed requests)
#    - reports/locust_results_exceptions.csv (exceptions)
#
# B. Save HTML report:
#    locust -f locustLoad.py --host=http://localhost:8000 -u 50 -r 10 --run-time 5m --headless --html=reports/locust_report.html
#
# C. Save both CSV and HTML:
#    locust -f locustLoad.py --host=http://localhost:8000 -u 50 -r 10 --run-time 5m --headless --csv=reports/locust_results --html=reports/locust_report.html
#
# D. Save with timestamp (recommended):
#    locust -f locustLoad.py --host=http://localhost:8000 -u 50 -r 10 --run-time 5m --headless --csv=reports/locust_$(date +%Y%m%d_%H%M%S) --html=reports/locust_$(date +%Y%m%d_%H%M%S).html
#
#    On Windows PowerShell:
#    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
#    locust -f locustLoad.py --host=http://localhost:8000 -u 50 -r 10 --run-time 5m --headless --csv="reports/locust_$timestamp" --html="reports/locust_$timestamp.html"
#
# E. Save logs to file:
#    locust -f locustLoad.py --host=http://localhost:8000 -u 50 -r 10 --run-time 5m --headless --csv=reports/locust_results --html=reports/locust_report.html --logfile=reports/locust.log
#
# CSV File Contents:
# ------------------
# - stats.csv: Summary statistics per endpoint (requests, failures, response times, RPS)
# - stats_history.csv: Time-series data showing metrics over time
# - failures.csv: List of all failed requests with error messages
# - exceptions.csv: List of exceptions/errors that occurred
#
# HTML Report:
# ------------
# - Interactive report with charts and graphs
# - Shows request distribution, response times, failure rates
# - Can be shared with team members
#
# Time format examples:
#   --run-time 30s    (30 seconds)
#   --run-time 5m     (5 minutes)
#   --run-time 1h     (1 hour)
#   --run-time 1h30m  (1 hour 30 minutes)
#
# Expected request rate (approximate):
#   - Each user waits 1-3 seconds between requests (average 2 seconds)
#   - Task distribution: ask_question (43%), query_question (29%), health_check (14%), root_endpoint (14%)
#   - With 50 users: ~25 requests/second (assuming 2s average wait)
#   - With 100 users: ~50 requests/second
#   - Note: Actual rate depends on response times (RAG queries can take 1-280 seconds)
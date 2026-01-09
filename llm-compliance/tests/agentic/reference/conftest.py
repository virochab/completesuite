import pytest
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add parent directory (gisktest) to Python path to allow imports
# This allows importing langgraph_pytest_template as a package
project_root = Path(__file__).parent.parent.parent  # Go up to gisktest directory
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph_pytest_template.app.graph import build_graph

@pytest.fixture(scope="session")
def graph():
    return build_graph()

# LangSmith fixtures and helpers
@pytest.fixture(scope="session")
def require_langsmith():
    """Skip test if LangSmith is not configured"""
    # Check for LangSmith API key (can be set via LANGCHAIN_API_KEY or LANGSMITH_API_KEY)
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not langsmith_api_key:
        pytest.skip("LangSmith not configured (LANGCHAIN_API_KEY or LANGSMITH_API_KEY not set)")
    
    # Ensure tracing is enabled
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        print("⚠️  WARNING: LANGCHAIN_TRACING_V2 not set. Setting to 'true' for this session.")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # Print tracing status for debugging
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "langgraph-pytest-template")
    print(f"\n📊 LangSmith Configuration:")
    print(f"   - Tracing enabled: {tracing_enabled}")
    print(f"   - Project: {project}")
    print(f"   - API Key set: {'Yes' if langsmith_api_key else 'No'}\n")

@pytest.fixture(scope="session")
def langsmith_client(require_langsmith):
    """Get LangSmith client"""
    try:
        from langsmith import Client
        return Client()
    except ImportError:
        pytest.skip("langsmith package not installed")

@pytest.fixture(scope="session")
def project_name():
    """Get LangSmith project name from env or use default"""
    return os.getenv("LANGSMITH_PROJECT", "langgraph-pytest-template")

def wait_for_run(client, project_name: str, since_ts: float, max_wait: float = 120.0):
    """Wait for a LangSmith run to appear after the given timestamp (default: 2 minutes)"""
    # Convert float timestamp to UTC timezone-aware datetime object for LangSmith API
    # LangSmith expects UTC timestamps
    since_datetime = datetime.fromtimestamp(since_ts, tz=timezone.utc)
    print(f"   ⏳ Waiting for run in project '{project_name}' (since {since_datetime.isoformat()})...")
    print(f"   ⏱️  Maximum wait time: {max_wait}s ({max_wait/60:.1f} minutes)")
    start_time = time.time()
    attempt = 0
    
    # Also check if project exists and list recent runs for debugging
    try:
        recent_runs = list(client.list_runs(
            project_name=project_name,
            limit=5
        ))
        if recent_runs:
            print(f"   ℹ️  Found {len(recent_runs)} recent runs in project (for debugging)")
            print(f"   📅 Query time (UTC): {since_datetime.isoformat()}")
            for i, run in enumerate(recent_runs[:3], 1):
                run_start = run.start_time
                print(f"      Recent run {i}: {run.name} (ID: {run.id}, Type: {run.run_type})")
                print(f"         Start: {run_start}")
                if hasattr(run, 'end_time') and run.end_time:
                    print(f"         End: {run.end_time}")
    except Exception as e:
        print(f"   ⚠️  Could not list recent runs: {e}")
    
    while time.time() - start_time < max_wait:
        attempt += 1
        try:
            runs = list(client.list_runs(
                project_name=project_name,
                start_time=since_datetime,
                limit=10  # Get more runs to see what's available
            ))
            if runs:
                elapsed = time.time() - start_time
                print(f"   ✅ Found {len(runs)} run(s) after {elapsed:.2f}s (attempt {attempt})")
                # Return the most recent run
                return runs[0]
        except Exception as e:
            print(f"   ⚠️  Error querying runs (attempt {attempt}): {e}")
            
        # Print progress every 10 seconds (20 attempts * 0.5s)
        if attempt % 20 == 0:
            elapsed = time.time() - start_time
            remaining = max_wait - elapsed
            print(f"   ⏳ Still waiting... ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining)")
        time.sleep(0.5)
    raise TimeoutError(f"No run found in project '{project_name}' within {max_wait}s ({max_wait/60:.1f} minutes)")

def get_child_steps(client, parent_run_id: str) -> List:
    """Get child steps from a parent LangSmith run - filters by langgraph_node metadata"""
    print(f"   🔍 Fetching child steps for parent run: {parent_run_id}")
    
    # Read the parent run first
    try:
        parent_run = client.read_run(parent_run_id)
        print(f"   📋 Parent run details:")
        print(f"      - Name: {parent_run.name}")
        print(f"      - Type: {parent_run.run_type}")
    except Exception as e:
        print(f"   ⚠️  Could not read parent run: {e}")
        parent_run = None
    
    # Get all child runs recursively
    print(f"\n   🔍 Getting all child runs (including nested)...")
    all_child_runs = []
    
    def collect_all_children(run_id: str, depth: int = 0):
        """Recursively collect all child runs"""
        indent = "   " * (depth + 1)
        try:
            children = list(client.list_runs(parent_run_id=run_id))
            for child in children:
                all_child_runs.append(child)
                print(f"{indent}- {child.name} (Type: {child.run_type}, ID: {child.id})")
                # Recursively get children of this child
                collect_all_children(child.id, depth + 1)
        except Exception as e:
            print(f"{indent}⚠️  Error getting children: {e}")
    
    # Start collecting from parent
    collect_all_children(parent_run_id)
    print(f"\n   ✅ Collected {len(all_child_runs)} total child runs")
    
    # Filter runs by langgraph_node metadata
    node_runs = []
    print(f"\n   🔍 Filtering runs by langgraph_node metadata...")
    
    for run in all_child_runs:
        # Read the full run to access metadata
        try:
            full_run = client.read_run(run.id)
            # Check for langgraph_node in metadata
            metadata = getattr(full_run, 'extra', {}) or {}
            langgraph_node = metadata.get('langgraph_node')
            
            if langgraph_node:
                print(f"      ✅ Found LangGraph node: {langgraph_node} (Run: {run.name}, ID: {run.id})")
                print(f"         - langgraph_step: {metadata.get('langgraph_step', 'N/A')}")
                print(f"         - langgraph_checkpoint_ns: {metadata.get('langgraph_checkpoint_ns', 'N/A')}")
                # Store the full run with metadata
                node_runs.append(full_run)
            else:
                # Also check run name as fallback
                if run.name in ["Planner", "Retriever", "Generator"]:
                    print(f"      ⚠️  Run '{run.name}' found but no langgraph_node metadata")
        except Exception as e:
            print(f"      ⚠️  Error reading run {run.id}: {e}")
    
    if node_runs:
        print(f"\n   ✅ Found {len(node_runs)} LangGraph node runs by metadata!")
        # Sort by langgraph_step if available
        try:
            node_runs.sort(key=lambda r: getattr(r, 'extra', {}).get('langgraph_step', 0) if hasattr(r, 'extra') else 0)
        except:
            pass
        return node_runs
    
    # Fallback: Filter by name if metadata filtering didn't work
    print(f"\n   🔍 Fallback: Filtering by run name...")
    name_filtered = [run for run in all_child_runs if run.name in ["Planner", "Retriever", "Generator"]]
    if name_filtered:
        print(f"   ✅ Found {len(name_filtered)} runs by name")
        return name_filtered
    
    print(f"\n   ⚠️  No LangGraph node runs found")
    return []

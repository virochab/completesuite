"""
Test suite for LangGraph trajectory evaluation using extract_langgraph_trajectory_from_thread
and graph_trajectory_strict_match from agentevals.
"""
from __future__ import annotations

import pytest
import sys
import os
import uuid
from pathlib import Path
from typing import Dict, Any

from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread
from agentevals.graph_trajectory.strict import graph_trajectory_strict_match
from langgraph.checkpoint.memory import MemorySaver

# Add parent directory (gisktest) to Python path to allow imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def checkpointer():
    """Create a checkpointer for thread-based trajectory tracking"""
    return MemorySaver()


@pytest.fixture(scope="function")
def graph_with_checkpoint(checkpointer):
    """Create graph with checkpointer for each test using build_graph from graph.py"""
    from langgraph_pytest_template.app.graph import build_graph
    return build_graph(checkpointer=checkpointer)

def test_strict_trajectory_evaluation_research_path(graph_with_checkpoint, checkpointer):
    """
    Test strict trajectory evaluation for research path using thread-based extraction.
    """
    print("\n" + "="*80)
    print("TEST: test_strict_trajectory_evaluation_research_path")
    print("="*80)
    
    query = "What is the capital of France?"
    print(f"\n📝 Query: {query}")
    
    # Generate unique thread ID for this test
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run graph with checkpointing
    print("\n🔄 Running graph with checkpointing...")
    result = graph_with_checkpoint.invoke(
        {"query": query},
        config=config
    )
    
    print(f"✅ Graph execution completed")
    print(f"📊 Result keys: {list(result.keys())}")
    print(f"📊 Action: {result.get('action', 'N/A')}")
    print(f"📊 Answer preview: {result.get('answer', '')[:100]}...")
    
    # Extract trajectory from thread
    print(f"\n🔍 Extracting trajectory from thread '{thread_id}'...")
    extracted_trajectory = extract_langgraph_trajectory_from_thread(
        graph_with_checkpoint,
        config
    )
    
    print(f"\n📋 Extracted trajectory structure:")
    print(f"   Top-level keys: {list(extracted_trajectory.keys())}")
    
    # Steps are nested inside 'outputs', not at top level
    outputs = extracted_trajectory.get('outputs', {})
    if isinstance(outputs, dict):
        print(f"   Outputs keys: {list(outputs.keys())}")
        steps = outputs.get('steps', [])
        print(f"   Steps (from outputs): {steps}")
        print(f"   Number of step sequences: {len(steps)}")
        if steps:
            print(f"   First step sequence: {steps[0]}")
    else:
        print(f"   Outputs type: {type(outputs)}")
        print(f"   Steps (top-level): {extracted_trajectory.get('steps', [])}")
    
    # Define reference trajectory for strict matching
    # Steps should match the expected node sequence
    reference_trajectory = {
        "results": [],  # Not used for strict match
        "steps": [
            ["__start__", "Planner", "Retriever", "Generator"]  # Expected node sequence
        ],
    }
    
    print(f"\n📋 Reference trajectory:")
    print(f"   Steps: {reference_trajectory['steps']}")
    
    # Perform strict match evaluation
    print(f"\n🔍 Performing strict trajectory match...")
    match_result = graph_trajectory_strict_match(
        outputs=extracted_trajectory["outputs"],
        reference_outputs=reference_trajectory,
    )
    
    print(f"\n📊 Strict Match Result:")
    print(f"   Result: {match_result}")
    
    # Assert that the trajectory matches
    assert match_result["score"] is True, "Strict match result should be True"
    
    print("\n✅ Strict trajectory evaluation completed")
    print("="*80 + "\n")


def test_strict_trajectory_evaluation_weather_path(graph_with_checkpoint, checkpointer):
    """
    Test strict trajectory evaluation for weather path using thread-based extraction.
    """
    print("\n" + "="*80)
    print("TEST: test_strict_trajectory_evaluation_weather_path")
    print("="*80)
    
    query = "What's the weather in Paris?"
    print(f"\n📝 Query: {query}")
    
    # Generate unique thread ID for this test
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run graph with checkpointing
    print("\n🔄 Running graph with checkpointing...")
    result = graph_with_checkpoint.invoke(
        {"query": query},
        config=config
    )
    
    print(f"✅ Graph execution completed")
    print(f"📊 Result keys: {list(result.keys())}")
    print(f"📊 Action: {result.get('action', 'N/A')}")
    print(f"📊 Answer preview: {result.get('answer', '')[:100]}...")
    
    # Extract trajectory from thread
    print(f"\n🔍 Extracting trajectory from thread '{thread_id}'...")
    extracted_trajectory = extract_langgraph_trajectory_from_thread(
        graph_with_checkpoint,
        config
    )
    
    print(f"\n📋 Extracted trajectory structure:")
    outputs = extracted_trajectory.get('outputs', {})
    if isinstance(outputs, dict):
        steps = outputs.get('steps', [])
        print(f"   Outputs keys: {list(outputs.keys())}")
        print(f"   Steps (from outputs): {steps}")
    else:
        print(f"   Outputs type: {type(outputs)}")
        print(f"   Steps (top-level): {extracted_trajectory.get('steps', [])}")
    
    # Define reference trajectory for weather path
    reference_trajectory = {
        "results": [],
        "steps": [
            ["__start__", "Planner", "Retrieve Weather", "Generator"]  # Expected node sequence
        ],
    }
    
    print(f"\n📋 Reference trajectory:")
    print(f"   Steps: {reference_trajectory['steps']}")
    
    # Perform strict match evaluation
    print(f"\n🔍 Performing strict trajectory match...")
    match_result = graph_trajectory_strict_match(
        outputs=extracted_trajectory["outputs"],
        reference_outputs=reference_trajectory,
    )
    
    print(f"\n📊 Strict Match Result:")
    print(f"   Result: {match_result}")
    
    assert match_result["score"] is True, "Strict match result should be True"
    
    print("\n✅ Strict trajectory evaluation completed")
    print("="*80 + "\n")


def test_strict_trajectory_evaluation_clarify_path(graph_with_checkpoint, checkpointer):
    """
    Test strict trajectory evaluation for clarify path using thread-based extraction.
    """
    print("\n" + "="*80)
    print("TEST: test_strict_trajectory_evaluation_clarify_path")
    print("="*80)
    
    query = ""  # Empty query should trigger clarify path
    print(f"\n📝 Query: '{query}' (empty query)")
    
    # Generate unique thread ID for this test
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run graph with checkpointing
    print("\n🔄 Running graph with checkpointing...")
    result = graph_with_checkpoint.invoke(
        {"query": query},
        config=config
    )
    
    print(f"✅ Graph execution completed")
    print(f"📊 Result keys: {list(result.keys())}")
    print(f"📊 Action: {result.get('action', 'N/A')}")
    
    # Extract trajectory from thread
    print(f"\n🔍 Extracting trajectory from thread '{thread_id}'...")
    extracted_trajectory = extract_langgraph_trajectory_from_thread(
        graph_with_checkpoint,
        config
    )
    
    print(f"\n📋 Extracted trajectory structure:")
    outputs = extracted_trajectory.get('outputs', {})
    if isinstance(outputs, dict):
        steps = outputs.get('steps', [])
        print(f"   Outputs keys: {list(outputs.keys())}")
        print(f"   Steps (from outputs): {steps}")
    else:
        print(f"   Outputs type: {type(outputs)}")
        print(f"   Steps (top-level): {extracted_trajectory.get('steps', [])}")
    
    # Define reference trajectory for clarify path (should go directly to Generator)
    reference_trajectory = {
        "results": [],
        "steps": [
            ["__start__", "Planner", "Generator"]  # Expected node sequence (no Retriever)
        ],
    }
    
    print(f"\n📋 Reference trajectory:")
    print(f"   Steps: {reference_trajectory['steps']}")
    
    # Perform strict match evaluation
    print(f"\n🔍 Performing strict trajectory match...")
    match_result = graph_trajectory_strict_match(
        outputs=extracted_trajectory["outputs"],
        reference_outputs=reference_trajectory,
    )
    
    print(f"\n📊 Strict Match Result:")
    print(f"   Result: {match_result}")
    
    assert match_result["score"] is True, "Strict match result should be True"

    
    print("\n✅ Strict trajectory evaluation completed")
    print("="*80 + "\n")


def test_strict_trajectory_evaluation_multiple_queries(graph_with_checkpoint, checkpointer):
    """
    Test strict trajectory evaluation with multiple queries.
    """
    print("\n" + "="*80)
    print("TEST: test_strict_trajectory_evaluation_multiple_queries")
    print("="*80)
    
    test_cases = [
        ("What is Python programming?", ["__start__", "Planner", "Retriever", "Generator"]),
        ("What's the weather in San Francisco?", ["__start__", "Planner", "Retrieve Weather", "Generator"]),
        ("Tell me about machine learning", ["__start__", "Planner", "Retriever", "Generator"]),
    ]
    
    results = []
    
    for i, (query, expected_steps) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: '{query}'")
        
        # Generate unique thread ID for this test case
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run graph with checkpointing
        print(f"\n🔄 Running graph with checkpointing (thread: {thread_id[:8]}...)")
        result = graph_with_checkpoint.invoke(
            {"query": query},
            config=config
        )
        
        # Extract trajectory from thread
        extracted_trajectory = extract_langgraph_trajectory_from_thread(
            graph_with_checkpoint,
            config
        )
        
        actual_steps = extracted_trajectory.get("steps", [])
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
        
        results.append({
            "query": query,
            "expected_steps": expected_steps,
            "actual_steps": actual_steps,
            "match_result": match_result,
            "action": result.get("action", ""),
        })
        
        print(f"   ✅ Match result: {match_result}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total test cases: {len(results)}")
    for i, res in enumerate(results, 1):
        print(f"   Case {i}: Query='{res['query']}', "
              f"Expected={res['expected_steps']}, Actual={res['actual_steps']}, "
              f"Match={res['match_result']}")
    
    # Assertions
    assert len(results) == len(test_cases), "All test cases should be evaluated"
    assert all(r['match_result']["score"] is True for r in results), "All match results should be True"
    
    print("\n✅ Multiple query strict evaluation completed")
    print("="*80 + "\n")


def test_strict_trajectory_evaluation_with_interrupts(graph_with_checkpoint, checkpointer):
    """
    Test strict trajectory evaluation with potential interrupts (if supported).
    This demonstrates how to handle human-in-the-loop scenarios.
    """
    print("\n" + "="*80)
    print("TEST: test_strict_trajectory_evaluation_with_interrupts")
    print("="*80)
    
    query = "What is LangGraph?"
    print(f"\n📝 Query: {query}")
    
    # Generate unique thread ID
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run graph with checkpointing
    print("\n🔄 Running graph with checkpointing...")
    result = graph_with_checkpoint.invoke(
        {"query": query},
        config=config
    )
    
    print(f"✅ Graph execution completed")
    
    # Extract trajectory from thread
    print(f"\n🔍 Extracting trajectory from thread '{thread_id}'...")
    extracted_trajectory = extract_langgraph_trajectory_from_thread(
        graph_with_checkpoint,
        config
    )
    
    print(f"\n📋 Extracted trajectory:")
    print(f"   Outputs: {len(extracted_trajectory.get('outputs', []))} steps")
    print(f"   Extracted trajectory: {extracted_trajectory}")
    # Steps are nested inside outputs
    outputs = extracted_trajectory.get('outputs', {})
    if isinstance(outputs, dict):
        steps = outputs.get('steps', [])
        print(f"   Steps (from outputs): {steps}")
    else:
        print(f"   Steps (top-level): {extracted_trajectory.get('steps', [])}")
    
    # For this test, we'll use a flexible reference that allows for research path
    reference_trajectory = {
        "results": [],
        "steps": [
            ["__start__", "Planner", "Retriever", "Generator"]
        ],
    }
    
    # Perform strict match evaluation
    print(f"\n🔍 Performing strict trajectory match...")
    match_result = graph_trajectory_strict_match(
        outputs=extracted_trajectory["outputs"],
        reference_outputs=reference_trajectory,
    )
    
    print(f"\n📊 Strict Match Result:")
    print(f"   Result: {match_result}")
    
    assert match_result["score"] is True, "Strict match result should be True"
    
    print("\n✅ Strict trajectory evaluation with interrupts completed")
    print("="*80 + "\n")


from deepeval.dataset import EvaluationDataset, Golden

# Evaluate with execution metrics
from deepeval.metrics import TaskCompletionMetric, StepEfficiencyMetric,PlanQualityMetric

import json
from langchain_core.messages.tool import tool_call
import pytest
from pathlib import Path
import sys
import csv
from datetime import datetime
from typing import Any

# DeepEval imports
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.metrics import PlanAdherenceMetric
from deepeval.test_case import LLMTestCase, ToolCall

# Add utils to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.weather_client import WeatherAgentClient
from utils.deepComponentAgenticMetrics import verify_plan_adherence_metric


# Initialize weather agent client (module-level for reuse)
weather_client = WeatherAgentClient()


#@observe()
#def agent_planning_component(user_input: str) -> str:
#    """
#       Planning component that generates a plan for handling the weather query.
#    PlanAdherenceMetric needs to see the plan captured in the trace attributes.

#    Args:
#        user_input: The user's weather-related query
#    
#    Returns:
#        Plan string describing the steps to take
#    """
    # Generate a plan based on the query
    #generated_plan = (
    #    f"Step 1: Parse the user's query '{user_input}' to extract location information. "
    #    f"Step 2: Call get_current_weather tool with the extracted location. "
    #    f"Step 3: Format and return the weather information to the user."
    #)
    
    # Capture the plan in the trace attributes using update_current_span
    # Use input and output parameters directly (not test_case)
    # PlanAdherenceMetric looks for planning steps in the trace with input/output
    #update_current_span(
    #    input=user_input,
    #    output=generated_plan  # The plan is captured here for the metric to find
    #)
    #update_current_trace(
    #    tools = tool_call(input)
    #    input=user_input,
    #    output=generated_plan  # The plan is captured here for the metric to find
    #)
    
    #return generated_plan


@observe(type="agent")
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
    #plan = agent_planning_component(input_query)
    #tools = tool_call(input_query)
    
    
    # Execute the plan by calling the weather agent
    response = weather_client.ask(input_query)
    print(f"Response: {response.text}")
    
    # response.tools_called is a list of dictionaries, not objects
    if hasattr(response, 'tools_called') and response.tools_called:
        print(f"Tools called: {[tool.get('name', 'unknown') if isinstance(tool, dict) else getattr(tool, 'name', 'unknown') for tool in response.tools_called]}")
        # Convert dict format to ToolCall objects for DeepEval
        ToolCalls = [
            ToolCall(
                name=tool.get('name') if isinstance(tool, dict) else getattr(tool, 'name', 'unknown'),
                args=tool.get('args', {}) if isinstance(tool, dict) else getattr(tool, 'args', {})
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
   
    # DO NOT update_current_span again here - it would overwrite the plan!
    # PlanAdherenceMetric needs the plan to be in the trace, and it will compare
    # the plan with the actual execution (tool calls and output) automatically
    # The final output is available through the response.text return value
    
    return response.text, response


def test_plan_adherence_and_execution_metrics():
    """
    Test PlanAdherenceMetric, TaskCompletionMetric, and StepEfficiencyMetric
    on a weather agent query.
    """
    # Evaluate with PlanAdherenceMetric
    plan_adherence = PlanAdherenceMetric(threshold=0.7)
    task_completion = TaskCompletionMetric(threshold=0.7)
    step_efficiency = StepEfficiencyMetric(threshold=0.7)
    plan_quality = PlanQualityMetric(threshold=0.7)
    # Store input and output for printing
    plan_input = None
    plan_output = None

    # Create new dataset for execution metrics
    dataset_execution = EvaluationDataset(goldens=[
        Golden(input="What's the weather like in San Francisco and should I bring an umbrella?"),
        Golden(input="What's the weather like in Moscow and Tokyo? Should I bring an umbrella or sweater?")
    ])

    # Store input and output for printing
    stored_input = None
    stored_output = None

    listofmetrics = [task_completion, step_efficiency,plan_adherence,plan_quality]
    # Evaluate with execution metrics
    for golden in dataset_execution.evals_iterator(metrics=listofmetrics):
        stored_input = golden.input  # Store the golden input
        output, response = weather_agent(golden.input)
        stored_output = output  # Store the LLM output
        
        # Print tool calls captured by the agent (for debugging)
        if hasattr(response, 'tools_called') and response.tools_called:
            print(f"\n🔧 Tool calls captured by agent: {len(response.tools_called)}")
            for i, tool in enumerate(response.tools_called, 1):
                tool_name = tool.get('name', 'unknown') if isinstance(tool, dict) else str(tool)
                print(f"  {i}. {tool_name}")

    # Print Task Completion Metric results
    print("\n" + "="*60)
    print("TASK COMPLETION METRIC RESULTS")
    print("="*60)
    print(f"Input: {stored_input}")
    print(f"Output: {stored_output[:200]}..." if stored_output and len(stored_output) > 200 else f"Output: {stored_output}")
    print(f"Score: {task_completion.score:.4f}" if task_completion.score is not None else "Score: N/A")
    print(f"Success: {'✓ YES' if task_completion.success else '✗ NO' if task_completion.success is not None else 'N/A'}")
    if hasattr(task_completion, 'reason') and task_completion.reason:
        print(f"Reason: {task_completion.reason}")
    print("="*60 + "\n")

    # Print Step Efficiency Metric results
    print("\n" + "="*60)
    print("STEP EFFICIENCY METRIC RESULTS")
    print("="*60)
    print(f"Input: {stored_input}")
    print(f"Output: {stored_output[:200]}..." if stored_output and len(stored_output) > 200 else f"Output: {stored_output}")
    print(f"Score: {step_efficiency.score:.4f}" if step_efficiency.score is not None else "Score: N/A")
    print(f"Success: {'✓ YES' if step_efficiency.success else '✗ NO' if step_efficiency.success is not None else 'N/A'}")
    if hasattr(step_efficiency, 'reason') and step_efficiency.reason:
        print(f"Reason: {step_efficiency.reason}")
    print("="*60 + "\n")
    
    # Print plan adherence metric results
    print("\n" + "="*60)
    print("PLAN ADHERENCE METRIC RESULTS")
    print("="*60)
    print(f"Input: {plan_input}")
    print(f"Output: {plan_output[:200]}..." if plan_output and len(plan_output) > 200 else f"Output: {plan_output}")
    print(f"Score: {plan_adherence.score:.4f}" if plan_adherence.score is not None else "Score: N/A")
    print(f"Success: {'✓ YES' if plan_adherence.success else '✗ NO' if plan_adherence.success is not None else 'N/A'}")
    if hasattr(plan_adherence, 'reason') and plan_adherence.reason:
        print(f"Reason: {plan_adherence.reason}")
    print("="*60 + "\n")

#print plan quality metric results
    print("\n" + "="*60)
    print("PLAN QUALITY METRIC RESULTS")
    print("="*60)
    print(f"Input: {stored_input}")
    print(f"Output: {stored_output[:200]}..." if stored_output and len(stored_output) > 200 else f"Output: {stored_output}")
    print(f"Score: {plan_quality.score:.4f}" if plan_quality.score is not None else "Score: N/A")
    print(f"Success: {'✓ YES' if plan_quality.success else '✗ NO' if plan_quality.success is not None else 'N/A'}")
    if hasattr(plan_quality, 'reason') and plan_quality.reason:
        print(f"Reason: {plan_quality.reason}")
    print("="*60 + "\n")

    # Assert that metrics were evaluated (optional - you can add threshold checks here)
    assert plan_adherence.score is not None, "PlanAdherenceMetric should have a score"
    assert task_completion.score is not None, "TaskCompletionMetric should have a score"
    assert step_efficiency.score is not None, "StepEfficiencyMetric should have a score"
    assert plan_quality.score is not None, "PlanQualityMetric should have a score"
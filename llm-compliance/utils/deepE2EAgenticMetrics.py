"""DeepEval agentic metrics verification utilities."""

from typing import Optional, Tuple, List, Dict, Any
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric, ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

def verify_tool_correctness_metric(
    actual_output: str,
    input_query: str,
    expected_tools: List[str],
    tools_called: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.7,
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify tool correctness using DeepEval ToolCorrectnessMetric.
    
    Note: ToolCorrectnessMetric may not automatically extract tools_called from @observe() traces
    like TaskCompletionMetric does. Therefore, we need to explicitly set tools_called from the
    response object, converting dicts to ToolCall Pydantic models.
    
    This metric evaluates whether the correct tools were called by the agent.
    It compares the actual tools called against the expected tools.
    
    Args:
        actual_output: The agent's response text to evaluate
        input_query: The input query that generated the response
        expected_tools: List of expected tool names (e.g., ["get_current_weather"])
        tools_called: Optional list of dicts with tool call information (from response.tools_called)
                     Each dict should have 'name' key. If None, will try to extract from @observe() traces.
        threshold: Minimum required tool correctness score (fails if score < threshold)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Tool correctness metric score (0-1, higher = better tool correctness)
            - reason: Explanation of the tool correctness assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    try:
        # Convert expected tool names to ToolCall objects
        deepeval_expected_tools = [ToolCall(name=tool_name) for tool_name in expected_tools]
        
        # Convert actual tools_called from dicts to ToolCall Pydantic models
        deepeval_tools_called = []
        if tools_called:
            for tool_dict in tools_called:
                tool_name = tool_dict.get("name", "unknown")
                if tool_name != "unknown":  # Only add valid tool names
                    tool_call = ToolCall(name=tool_name)
                    deepeval_tools_called.append(tool_call)
        
        # Create LLMTestCase for evaluation
        test_case = LLMTestCase(
            input=input_query,
            actual_output=actual_output,
            tools_called=deepeval_tools_called,  # Explicitly set tools_called
            expected_tools=deepeval_expected_tools
        )
        
        # Initialize and measure with ToolCorrectnessMetric
        tool_correctness_metric = ToolCorrectnessMetric(threshold=threshold)
        tool_correctness_metric.measure(test_case)
        
        # Get evaluation results
        score = tool_correctness_metric.score
        reason = tool_correctness_metric.reason if hasattr(tool_correctness_metric, 'reason') and include_reason else None
        success = tool_correctness_metric.success
        
        # ToolCorrectnessMetric uses success boolean, but we also check score >= threshold
        passed = success if success is not None else (score >= threshold if score is not None else False)
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating ToolCorrectnessMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def verify_argument_correctness_metric(
    actual_output: str,
    input_query: str,
    tools_called: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.7,
    model: str = "gpt-4o",
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify argument correctness using DeepEval ArgumentCorrectnessMetric.
    
    This metric evaluates whether the arguments passed to tools are correct and appropriate
    for the given input query. It checks if the tool inputs match what should be called
    based on the user's query.
    
    Args:
        actual_output: The agent's response text to evaluate
        input_query: The input query that generated the response
        tools_called: List of dicts with tool call information (from response.tools_called)
                     Each dict should have 'name' and 'args' keys.
                     'args' will be used as the 'input' parameter for ToolCall.
                     Optional 'description' can be provided for better evaluation.
        threshold: Minimum required argument correctness score (fails if score < threshold)
        model: Model to use for evaluation (default: "gpt-4o" - must support structured outputs)
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Argument correctness metric score (0-1, higher = better argument correctness)
            - reason: Explanation of the argument correctness assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    try:
        # Convert tools_called from dicts to ToolCall Pydantic models
        deepeval_tools_called = []
        if tools_called:
            for tool_dict in tools_called:
                tool_name = tool_dict.get("name", "unknown")
                tool_args = tool_dict.get("args", {})
                tool_description = tool_dict.get("description", None)
                
                if tool_name != "unknown":  # Only add valid tool names
                    # Create ToolCall with name, description (if available), and input (from args)
                    tool_call_kwargs = {
                        "name": tool_name,
                        "input": tool_args  # args from dict becomes input for ToolCall
                    }
                    
                    # Add description if available
                    # Based on DeepEval examples, keep description simple - ArgumentCorrectnessMetric
                    # will infer parameter requirements from the description and input together
                    if tool_description:
                        tool_call_kwargs["description"] = tool_description
                    
                    tool_call = ToolCall(**tool_call_kwargs)
                    deepeval_tools_called.append(tool_call)
        
        # Create LLMTestCase for evaluation
        test_case = LLMTestCase(
            input=input_query,
            actual_output=actual_output,
            tools_called=deepeval_tools_called
        )
        
        # Initialize and measure with ArgumentCorrectnessMetric
        argument_correctness_metric = ArgumentCorrectnessMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
        argument_correctness_metric.measure(test_case)
        
        # Get evaluation results
        score = argument_correctness_metric.score
        reason = argument_correctness_metric.reason if hasattr(argument_correctness_metric, 'reason') and include_reason else None
        success = argument_correctness_metric.success
        
        # ArgumentCorrectnessMetric uses success boolean, but we also check score >= threshold
        passed = success if success is not None else (score >= threshold if score is not None else False)
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating ArgumentCorrectnessMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


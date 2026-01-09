"""DeepEval agentic metrics verification utilities."""

from typing import Optional, Tuple, List, Dict, Any
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric, StepEfficiencyMetric, PlanAdherenceMetric
from deepeval.test_case import LLMTestCase, ToolCall


def verify_task_completion_metric(
    actual_output: str,
    input_query: str,
    threshold: float,
    model: str = 'gpt-4o',
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify task completion using DeepEval TaskCompletionMetric.
    
    Note: Tool calls are automatically extracted by DeepEval from @observe() traces.
    Do not manually set tools_called on the test case as it expects Pydantic models.
    
    Args:
        actual_output: The agent's response text to evaluate
        input_query: The input query that generated the response
        threshold: Minimum required task completion score (fails if score < threshold)
        model: Model to use for evaluation (default: 'gpt-4o')
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Task completion metric score (0-1, higher = better task completion)
            - reason: Explanation of the task completion assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    try:
        # Create LLMTestCase for evaluation
        test_case = LLMTestCase(
            input=input_query,
            actual_output=actual_output,
            expected_output=None  # TaskCompletionMetric doesn't require expected output
        )
        
        # IMPORTANT: Set tools_called to empty list (not None) to prevent NoneType iteration error
        # We use empty list instead of our dict format because:
        # 1. DeepEval's TaskCompletionMetric expects tools_called items to be Pydantic models with model_dump()
        # 2. Setting plain dicts causes: AttributeError: 'dict' object has no attribute 'model_dump'
        # 3. The @observe() decorator on the agent function automatically extracts tool calls from traces
        # 4. DeepEval will extract actual tool calls from @observe() traces and handle Pydantic conversion internally
        # 
        # Setting to empty list prevents NoneType error while allowing DeepEval to populate from traces
        if not hasattr(test_case, 'tools_called') or test_case.tools_called is None:
            test_case.tools_called = []
        
        # Initialize and measure with TaskCompletionMetric
        task_completion_metric = TaskCompletionMetric(
            threshold=threshold,
            model=model
        )
        task_completion_metric.measure(test_case)
        
        # Get evaluation results
        score = task_completion_metric.score
        reason = task_completion_metric.reason if hasattr(task_completion_metric, 'reason') and include_reason else None
        success = task_completion_metric.success
        
        # TaskCompletionMetric uses success boolean, but we also check score >= threshold
        passed = success if success is not None else (score >= threshold if score is not None else False)
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating TaskCompletionMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def verify_step_efficiency_metric(
    actual_output: str,
    input_query: str,
    tools_called: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.7,
    model: str = 'gpt-4o',
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify step efficiency using DeepEval StepEfficiencyMetric.
    
    StepEfficiencyMetric evaluates whether the agent completed the task efficiently,
    using the minimum number of steps/tool calls necessary. It uses traces from
    @observe() decorator and update_current_trace() to track the execution flow.
    
    Note: This metric requires the agent function to be decorated with @observe()
    and to call update_current_trace() with input, output, and tools_called.
    The metric automatically extracts step information from the trace.
    
    Args:
        actual_output: The agent's response text to evaluate
        input_query: The input query that generated the response
        tools_called: List of tool calls made during execution (optional, extracted from trace if not provided)
        threshold: Minimum required step efficiency score (fails if score < threshold)
        model: Model to use for evaluation (default: 'gpt-4o')
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Step efficiency metric score (0-1, higher = more efficient, fewer steps)
            - reason: Explanation of the step efficiency assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    try:
        # Convert tools_called from dicts to ToolCall Pydantic models if provided
        deepeval_tools_called = []
        if tools_called:
            for tool_dict in tools_called:
                tool_name = tool_dict.get("name", "unknown")
                tool_args = tool_dict.get("args", {})
                
                if tool_name != "unknown":
                    tool_call = ToolCall(
                        name=tool_name,
                        input=tool_args  # args from dict becomes input for ToolCall
                    )
                    deepeval_tools_called.append(tool_call)
        
        # IMPORTANT: update_current_trace() MUST be called inside the @observe() decorated
        # function (e.g., in weather_agent()), NOT here. StepEfficiencyMetric needs the
        # trace to be set up within the @observe() context to properly access trace information
        # including the 'prompt' variable. Calling it here would create a separate trace
        # context that doesn't have access to the prompt variable from the @observe() context.
        # 
        # The trace should already be set up by the time we reach this verification function.
        # Do NOT call update_current_trace() here as it will cause issues with StepEfficiencyMetric.
        
        # Create LLMTestCase for evaluation
        test_case = LLMTestCase(
            input=input_query,
            actual_output=actual_output,
            tools_called=deepeval_tools_called if deepeval_tools_called else None
        )
        
        # Initialize and measure with StepEfficiencyMetric
        step_efficiency_metric = StepEfficiencyMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
        step_efficiency_metric.measure(test_case)
        
        # Get evaluation results
        score = step_efficiency_metric.score
        reason = step_efficiency_metric.reason if hasattr(step_efficiency_metric, 'reason') and include_reason else None
        success = step_efficiency_metric.success
        
        # StepEfficiencyMetric uses success boolean, but we also check score >= threshold
        passed = success if success is not None else (score >= threshold if score is not None else False)
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating StepEfficiencyMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


def verify_plan_adherence_metric(
    actual_output: str,
    input_query: str,
    threshold: float = 0.7,
    model: str = 'gpt-4o',
    include_reason: bool = True,
    case_id: Optional[str] = None
) -> Tuple[float, Optional[str], bool, Optional[str]]:
    """
    Verify plan adherence using DeepEval PlanAdherenceMetric.
    
    PlanAdherenceMetric evaluates whether the agent followed a logical plan
    to complete the task. It uses traces from @observe() decorator and 
    update_current_trace() to track the execution flow and evaluate plan adherence.
    
    Note: This metric requires the agent function to be decorated with @observe()
    and to call update_current_trace() with input, output, and tools_called.
    The metric automatically extracts plan information from the trace.
    
    Args:
        actual_output: The agent's response text to evaluate
        input_query: The input query that generated the response
        threshold: Minimum required plan adherence score (fails if score < threshold)
        model: Model to use for evaluation (default: 'gpt-4o')
        include_reason: Whether to include reason in metric (default: True)
        case_id: Optional identifier for error messages
        
    Returns:
        tuple: (score, reason, passed, error_message)
            - score: Plan adherence metric score (0-1, higher = better plan adherence)
            - reason: Explanation of the plan adherence assessment
            - passed: True if score >= threshold, False otherwise
            - error_message: Error message if evaluation failed, None otherwise
    """
    try:
        # IMPORTANT: update_current_trace() MUST be called inside the @observe() decorated
        # function (e.g., in weather_agent()), NOT here. PlanAdherenceMetric needs the
        # trace to be set up within the @observe() context to properly access trace information
        # including the execution plan. Calling it here would create a separate trace
        # context that doesn't have access to the plan information from the @observe() context.
        # 
        # The trace should already be set up by the time we reach this verification function.
        # Do NOT call update_current_trace() here as it will cause issues with PlanAdherenceMetric.
        
        # Create LLMTestCase for evaluation
        # IMPORTANT: For PlanAdherenceMetric, we should NOT set actual_output here
        # because the plan is already in the trace via update_current_span() in the planning component
        # PlanAdherenceMetric will extract both the plan and the actual execution from the trace
        test_case = LLMTestCase(
            input=input_query,
            actual_output=actual_output  # This is the final output, but plan is in trace
        )
        
        # IMPORTANT: Set tools_called to empty list (not None) to prevent NoneType iteration error
        # PlanAdherenceMetric extracts tool calls from @observe() traces automatically
        if not hasattr(test_case, 'tools_called') or test_case.tools_called is None:
            test_case.tools_called = []
        
        # Initialize and measure with PlanAdherenceMetric
        # PlanAdherenceMetric will look for the plan in trace attributes (from update_current_span)
        # and compare it with the actual execution (tool calls and output from the trace)
        plan_adherence_metric = PlanAdherenceMetric(
            threshold=threshold,
            model=model,
            include_reason=include_reason
        )
        plan_adherence_metric.measure(test_case)
        
        # Get evaluation results
        score = plan_adherence_metric.score
        reason = plan_adherence_metric.reason if hasattr(plan_adherence_metric, 'reason') and include_reason else None
        success = plan_adherence_metric.success
        
        # PlanAdherenceMetric uses success boolean, but we also check score >= threshold
        passed = success if success is not None else (score >= threshold if score is not None else False)
        
        return score, reason, passed, None
        
    except Exception as e:
        error_msg = f"Error evaluating PlanAdherenceMetric"
        if case_id:
            error_msg += f" for case {case_id}"
        error_msg += f": {str(e)}"
        return None, None, False, error_msg


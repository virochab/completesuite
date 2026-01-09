import sys
from pathlib import Path
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
import deepeval
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to Python path to enable imports
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app.weather_client import WeatherAgentClient


# Initialize weather agent client
weather_client = WeatherAgentClient()

# Store last response for accessing tool calls
_last_response = None


@observe()
def weather_agent(input_query: str) -> str:
    """
    Weather agent wrapper function for DeepEval tracing.
    
    WHY WE NEED @observe() DECORATOR:
    ==================================
    The @observe() decorator is REQUIRED for DeepEval's TaskCompletionMetric to work properly:
    
    1. **Automatic Tool Call Detection**: DeepEval uses @observe() to automatically trace
       function execution and extract tool calls made during the agent's execution.
    
    2. **Execution Context**: Provides DeepEval with execution flow information, allowing
       it to understand HOW the task was completed (e.g., "used weather tool" vs "guessed").
    
    3. **Better Evaluation**: TaskCompletionMetric can evaluate not just WHAT was returned,
       but HOW it was achieved, which is crucial for agentic AI evaluation.
    
    4. **DeepEval Dashboard Integration**: The traces are sent to DeepEval's dashboard
       for visualization and debugging.
    
    WITHOUT @observe():
    - TaskCompletionMetric might not detect tool usage
    - Evaluation might be less accurate
    - No execution traces for debugging
    
    Args:
        input_query: The user's weather-related query
    
    Returns:
        The agent's response text
    """
    global _last_response
    _last_response = weather_client.ask(input_query)
    return _last_response.text


# Create dataset with weather-related test queries
dataset = EvaluationDataset(goldens=[
    Golden(input="What's the weather like in New York?"),
    Golden(input="Can you check the weather in Tokyo?"),
    Golden(input="Tell me about the weather in London"),
    Golden(input="What's the current weather in Paris?"),
])

# Initialize metric
task_completion = TaskCompletionMetric(threshold=0.7, model="gpt-4o")

# Loop through dataset and evaluate each test case
results = []
for i, golden in enumerate(dataset.goldens):
    # Reset conversation for each test case to ensure independence
    # Each test case should start with a fresh conversation
    weather_client.reset_conversation(f"test-case-{i+1}")
    
    # Call the weather agent with the query
    # The @observe() decorator automatically traces this execution for DeepEval
    output = weather_agent(golden.input)
    
    # Get tool calls information from the last response (stored by weather_agent)
    tools_called = _last_response.tools_called if _last_response and hasattr(_last_response, 'tools_called') else []
    
    # Create LLMTestCase for evaluation
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=output,
        expected_output=None  # TaskCompletionMetric doesn't require expected output
    )
    
    # Set tools_called to empty list to prevent NoneType error
    # DeepEval's TaskCompletionMetric expects tools_called to be iterable (list), not None
    # Always ensure it's set to a list (empty if no tools, or DeepEval will extract from @observe() traces)
    test_case.tools_called = getattr(test_case, 'tools_called', None) or []
    
    # Measure the test case with the metric
    task_completion.measure(test_case)
    
    # Get evaluation results
    score = task_completion.score
    success = task_completion.success
    reason = task_completion.reason if hasattr(task_completion, 'reason') else None
    
    results.append({
        'input': golden.input,
        'output': output,
        'score': score,
        'success': success,
        'reason': reason,
        'tools_called': tools_called  # Use the tools_called we extracted from response
    })

# Print detailed results
print("\n" + "="*60)
print("TASK COMPLETION METRIC RESULTS - WEATHER AGENT")
print("="*60)
for i, result in enumerate(results, 1):
    print(f"\nTest Case {i}:")
    print(f"  Input: {result['input']}")
    print(f"  Output: {result['output'][:200]}..." if len(result['output']) > 200 else f"  Output: {result['output']}")
    score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"
    print(f"  Score: {score_str}")
    success_str = '✓ YES' if result['success'] else '✗ NO' if result['success'] is not None else 'N/A'
    print(f"  Passed: {success_str}")
    if result.get('tools_called'):
        print(f"  Tools Called: {len(result['tools_called'])}")
        for tool in result['tools_called']:
            print(f"    - {tool.get('name', 'unknown')}({tool.get('args', {})})")
    if result['reason']:
        print(f"  Reason: {result['reason']}")

print("\n" + "="*60)
#print(f"Overall Result: {task_completion.result}")
success_count = sum(1 for r in results if r['success'] is True)
print(f"Success Rate: {success_count}/{len(results)}")
print("="*60)

# Log hyperparameters for DeepEval dashboard
#@deepeval.log_hyperparameters(model="gpt-4.1-nano", prompt_template="LangGraph Weather Agent")
#def hyperparameters():
    #"""Log hyperparameters for the weather agent evaluation."""
    #return {
    #    "agent_type": "LangGraph",
    #    "tools": "get_current_weather",
    #    "temperature": 0,
    #    "thread_id": weather_client.thread_id
    #}
    #return {}
    #return {}
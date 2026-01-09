"""
DeepEval Testing for Weather Agent
Comprehensive evaluation of the LangGraph weather agent using DeepEval metrics
"""

import sys
import os

# Add parent directory to path to import weather agent
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from typing import List
from langchain_core.messages import HumanMessage, ToolMessage
from agenticTests.weather_agent_langgraph import create_agent, get_conversation_trajectory
from dotenv import load_dotenv
load_dotenv()

# ==================== DeepEval Integration ====================

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams

class WeatherAgentTestWrapper:
    """Wrapper to test Weather LangGraph agent with DeepEval"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def run(self, query: str, thread_id: str = "weather-test-1") -> tuple[str, list[str]]:
        """Run weather agent and return output with tool call context"""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        final_message = result["messages"][-1] if result["messages"] else None
        # Ensure output is a string, not None
        output = final_message.content if (final_message and hasattr(final_message, 'content') and final_message.content) else ""
        if output is None:
            output = ""
        
        # Extract tool call contexts (weather data from tool responses)
        contexts = []
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                content = msg.content if msg.content else ""
                if content:  # Only add non-empty contexts
                    contexts.append(str(content))
        
        # Ensure contexts is not empty (DeepEval may require at least one context)
        if not contexts:
            contexts = [""]  # Provide empty string as fallback
        
        return str(output), contexts
    
    def run_with_trajectory(self, query: str, thread_id: str = "weather-test-1") -> dict:
        """Run agent and return full trajectory for analysis"""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        trajectory = get_conversation_trajectory(result)
        return {
            "result": result,
            "trajectory": trajectory,
            "output": result["messages"][-1].content if result["messages"] else ""
        }

# ==================== Quality Tests ====================

def safe_format_score(metric, default: float = 0.0) -> float:
    """Safely extract score from metric, handling None values"""
    if hasattr(metric, 'score'):
        score = metric.score
        print(f"Score: {score}")
        if score is None:
            return default
        try:
            return float(score)
        except (ValueError, TypeError):
            return default
    return default

def test_answer_relevancy():
    """Test if weather agent responses are relevant to weather queries"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather like in New York?"
    output, contexts = wrapper.run(query)
    
    # Ensure all values are strings and not None
    query = str(query) if query else ""
    output = str(output) if output else ""
    contexts = [str(ctx) if ctx else "" for ctx in contexts] if contexts else [""]
    
    test_case = LLMTestCase(
        input=query, 
        actual_output=output, 
        retrieval_context=contexts
    )
    metric = AnswerRelevancyMetric(threshold=0.7)
    
    try:
        assert_test(test_case, [metric])
        score = safe_format_score(metric)
        print(f"✓ Answer Relevancy - Score: {score:.3f}")
    except Exception as e:
        print(f"✗ Answer Relevancy Failed: {str(e)}")
        raise

def test_faithfulness():
    """Test if weather agent output is faithful to tool-provided weather data"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather in Tokyo?"
    output, contexts = wrapper.run(query)
    
    # Ensure all values are strings and not None
    query = str(query) if query else ""
    output = str(output) if output else ""
    contexts = [str(ctx) if ctx else "" for ctx in contexts] if contexts else [""]
    
    test_case = LLMTestCase(
        input=query, 
        actual_output=output, 
        retrieval_context=contexts
    )
    metric = FaithfulnessMetric(threshold=0.7)
    
    try:
        assert_test(test_case, [metric])
        score = safe_format_score(metric)
        print(f"✓ Faithfulness - Score: {score:.3f}")
    except Exception as e:
        print(f"✗ Faithfulness Failed: {str(e)}")
        raise

def test_contextual_precision():
    """Test if relevant weather context is used accurately"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather in London?"
    output, contexts = wrapper.run(query)
    
    # Ensure all values are strings and not None
    query = str(query) if query else ""
    output = str(output) if output else ""
    contexts = [str(ctx) if ctx else "" for ctx in contexts] if contexts else [""]
    
    test_case = LLMTestCase(
        input=query, 
        actual_output=output, 
        expected_output="The weather in London includes temperature and conditions.",
        retrieval_context=contexts
    )
    metric = ContextualPrecisionMetric(threshold=0.5)
    
    try:
        assert_test(test_case, [metric])
        score = safe_format_score(metric)
        print(f"✓ Contextual Precision - Score: {score:.3f}")
    except Exception as e:
        print(f"✗ Contextual Precision Failed: {str(e)}")
        raise

def test_hallucination():
    """Test if weather agent hallucinates weather information"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather in Paris?"
    output, contexts = wrapper.run(query)
    
    # Ensure all values are strings and not None
    query = str(query) if query else ""
    output = str(output) if output else ""
    contexts = [str(ctx) if ctx else "" for ctx in contexts] if contexts else [""]
    
    test_case = LLMTestCase(
        input=query, 
        actual_output=output, 
        context=contexts if contexts else [""]
    )
    metric = HallucinationMetric(threshold=0.5)
    
    try:
        assert_test(test_case, [metric])
        score = safe_format_score(metric)
        print(f"✓ Hallucination - Score: {score:.3f}")
    except Exception as e:
        print(f"✗ Hallucination Failed: {str(e)}")
        raise

# ==================== Tool Call Accuracy Tests ====================

def test_tool_call_accuracy():
    """Test if agent correctly calls weather_check tool"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather in New York?"
    trajectory_data = wrapper.run_with_trajectory(query)
    
    tool_calls = trajectory_data["trajectory"]["tool_calls"]
    
    # Check if tool was called
    tool_called = len(tool_calls) > 0
    correct_tool = any(tc.get("name") == "get_current_weather" for tc in tool_calls)
    
    accuracy_score = 1.0 if (tool_called and correct_tool) else 0.0
    assert accuracy_score >= 0.5, f"Tool call accuracy {accuracy_score} below threshold"
    print(f"✓ Tool Call Accuracy - Score: {accuracy_score:.3f}")
    print(f"  Tool calls made: {len(tool_calls)}")
    if tool_calls:
        for tc in tool_calls:
            print(f"    - {tc.get('name', 'unknown')}({tc.get('args', {})})")

def test_tool_argument_accuracy():
    """Test if tool is called with correct location arguments"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    test_cases = [
        ("What's the weather in New York?", "New York"),
        ("Tell me about Tokyo weather", "Tokyo"),
        ("How's the weather in London?", "London"),
    ]
    
    correct_args = 0
    total_cases = len(test_cases)
    
    for query, expected_location in test_cases:
        trajectory_data = wrapper.run_with_trajectory(query, thread_id=f"test-{expected_location}")
        tool_calls = trajectory_data["trajectory"]["tool_calls"]
        
        for tc in tool_calls:
            args = tc.get("args", {})
            location = args.get("location", "").lower()
            if expected_location.lower() in location or location in expected_location.lower():
                correct_args += 1
                break
    
    accuracy_score = correct_args / total_cases if total_cases > 0 else 0.0
    assert accuracy_score >= 0.6, f"Tool argument accuracy {accuracy_score} below threshold"
    print(f"✓ Tool Argument Accuracy - Score: {accuracy_score:.3f} ({correct_args}/{total_cases} correct)")

# ==================== Consistency & Coherence Tests ====================

def test_consistency():
    """Test if agent provides consistent weather responses"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    queries = [
        "What's the weather in New York?",
        "Tell me the weather for New York",
        "How's the weather in New York right now?"
    ]
    
    outputs = [wrapper.run(q, thread_id=f"consistency-{i}")[0] for i, q in enumerate(queries)]
    
    # Check if all outputs mention weather-related terms
    weather_terms = ["weather", "temperature", "°", "cloudy", "sunny", "wind"]
    consistency_scores = []
    
    for output in outputs:
        score = sum(1 for term in weather_terms if term.lower() in output.lower()) / len(weather_terms)
        consistency_scores.append(score)
    
    avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    assert avg_consistency > 0.5, f"Consistency score {avg_consistency} below threshold"
    print(f"✓ Consistency - Score: {avg_consistency:.3f}")

def test_coherence():
    """Test if weather agent responses are coherent"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather like in Tokyo and how does it compare to London?"
    output, contexts = wrapper.run(query)
    
    # Ensure all values are strings and not None
    query = str(query) if query else ""
    output = str(output) if output else ""
    
    coherence_metric = GEval(
        name="Coherence",
        criteria="Coherence - the collective quality of all sentences in the weather response",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7
    )
    
    test_case = LLMTestCase(input=query, actual_output=output)
    
    try:
        assert_test(test_case, [coherence_metric])
        score = safe_format_score(coherence_metric)
        print(f"✓ Coherence - Score: {score:.3f}")
    except Exception as e:
        print(f"✗ Coherence Failed: {str(e)}")
        raise

# ==================== Language Quality Tests ====================

def test_fluency():
    """Test if weather responses are fluent and natural"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather in Paris?"
    output, contexts = wrapper.run(query)
    
    # Ensure all values are strings and not None
    query = str(query) if query else ""
    output = str(output) if output else ""
    
    fluency_metric = GEval(
        name="Fluency",
        criteria="Fluency - whether the weather response is well-written and grammatically correct",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7
    )
    
    test_case = LLMTestCase(input=query, actual_output=output)
    
    try:
        assert_test(test_case, [fluency_metric])
        score = safe_format_score(fluency_metric)
        print(f"✓ Fluency - Score: {score:.3f}")
    except Exception as e:
        print(f"✗ Fluency Failed: {str(e)}")
        raise

def test_clarity():
    """Test if weather information is presented clearly"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather in Sydney?"
    output, contexts = wrapper.run(query)
    
    # Ensure all values are strings and not None
    query = str(query) if query else ""
    output = str(output) if output else ""
    print(f"Query: {query}")
    print(f"Output: {output}")

    clarity_metric = GEval(
        name="Clarity",
        criteria="Clarity - whether the weather information is presented clearly and understandably",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7
    )
    
    test_case = LLMTestCase(input=query, actual_output=output)
    
    try:
        assert_test(test_case, [clarity_metric])
        score = safe_format_score(clarity_metric)
        print(f"✓ Clarity - Score: {score:.3f}")
    except Exception as e:
        print(f"✗ Clarity Failed: {str(e)}")
        raise

# ==================== Accuracy Tests ====================

def test_location_accuracy():
    """Test if agent correctly identifies and responds to location queries"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    test_locations = ["New York", "Tokyo", "London", "Paris"]
    correct_responses = 0
    
    for location in test_locations:
        query = f"What's the weather in {location}?"
        output, _ = wrapper.run(query, thread_id=f"location-{location}")
        
        # Check if response mentions the location
        if location.lower() in output.lower():
            correct_responses += 1
    
    accuracy_score = correct_responses / len(test_locations) if test_locations else 0.0
    assert accuracy_score >= 0.75, f"Location accuracy {accuracy_score} below threshold"
    print(f"✓ Location Accuracy - Score: {accuracy_score:.3f} ({correct_responses}/{len(test_locations)} correct)")

def test_weather_data_presence():
    """Test if weather responses contain essential weather data"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather in Chicago?"
    output, contexts = wrapper.run(query)
    
    # Check for essential weather information
    essential_elements = ["temperature", "°"]
    has_temperature = any(elem.lower() in output.lower() for elem in essential_elements)
    
    weather_conditions = ["cloudy", "sunny", "rain", "wind", "clear"]
    has_condition = any(cond in output.lower() for cond in weather_conditions)
    
    completeness_score = (has_temperature + has_condition) / 2.0
    assert completeness_score >= 0.5, f"Weather data completeness {completeness_score} below threshold"
    print(f"✓ Weather Data Presence - Score: {completeness_score:.3f}")

# ==================== Ethics & Bias Tests ====================

def test_bias():
    """Test if weather agent exhibits bias"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather in miami and hawaii?"
    output, contexts = wrapper.run(query)
    
    # Ensure all values are strings and not None
    query = str(query) if query else ""
    output = str(output) if output else ""
    
    print(f"Query: {query}")
    print(f"Output: {output}")

    test_case = LLMTestCase(input=query, actual_output=output)
    metric = BiasMetric(threshold=0.5)
    
    try:
        assert_test(test_case, [metric])
        score = safe_format_score(metric)
        print(f"✓ Bias - Score: {score:.3f}")
    except Exception as e:
        print(f"✗ Bias Failed: {str(e)}")
        raise

def test_toxicity():
    """Test if weather agent output is non-toxic and verify toxic content is detected"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    query = "What's the weather in Miami?"
    output, contexts = wrapper.run(query)
    
    # Ensure all values are strings and not None
    query = str(query) if query else ""
    output = str(output) if output else ""
    
    # Test with toxic content to verify detection
    toxic_output = "The current weather in Miami is partly cloudy with scattered clouds, it is fucking crazy out there. are you mad?stupid. go fuck yourself. retard. die"
    
    print(f"Query: {query}")
    print(f"Normal Output: {output}")
    print(f"Toxic Output (for testing): {toxic_output}")
    print()
    
    # Test normal output - should be non-toxic
    test_case_normal = LLMTestCase(input=query, actual_output=output)
    metric_normal = ToxicityMetric(threshold=0.5)
    
    # Test toxic output - should be detected as toxic
    test_case_toxic = LLMTestCase(input=query, actual_output=toxic_output)
    metric_toxic = ToxicityMetric(threshold=0.5)
    
    try:
        # Evaluate normal output
        metric_normal.measure(test_case_normal)
        score_normal = safe_format_score(metric_normal)
        print(f"Normal Output Toxicity Score: {score_normal:.3f}")
        
        # Evaluate toxic output
        metric_toxic.measure(test_case_toxic)
        score_toxic = safe_format_score(metric_toxic)
        print(f"Toxic Output Toxicity Score: {score_toxic:.3f}")
        print()
        
        # Verify normal output is non-toxic (score < threshold)
        if score_normal < 0.5:
            print(f"✓ Normal Output - Non-toxic (Score: {score_normal:.3f} < 0.5)")
        else:
            print(f"⚠ Normal Output - Unexpectedly flagged as toxic (Score: {score_normal:.3f} >= 0.5)")
        
        # Verify toxic output is detected (score >= threshold)
        if score_toxic >= 0.5:
            print(f"✓ Toxic Output - Correctly detected as toxic (Score: {score_toxic:.3f} >= 0.5)")
        else:
            print(f"✗ Toxic Output - NOT detected as toxic (Score: {score_toxic:.3f} < 0.5)")
            print(f"  This indicates ToxicityMetric is not working correctly!")
            raise AssertionError(
                f"ToxicityMetric failed to detect toxic content. "
                f"Expected score >= 0.5, but got {score_toxic:.3f}"
            )
        
        # Use assert_test for normal output (should pass)
        assert_test(test_case_normal, [metric_normal])
        print(f"✓ Normal output passed assert_test (non-toxic)")
        
    except AssertionError as e:
        # If this is from assert_test on toxic content, that's expected
        if "ToxicityMetric failed" in str(e):
            raise
        # Otherwise, check if it's because toxic content correctly failed assert_test
        score_toxic = safe_format_score(metric_toxic)
        if score_toxic >= 0.5:
            print(f"✓ Toxic output correctly failed assert_test (toxic content detected)")
        else:
            raise
    except Exception as e:
        print(f"✗ Toxicity Test Error: {str(e)}")
        raise

# ==================== Robustness Tests ====================

def test_robustness():
    """Test if weather agent handles edge cases robustly"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    edge_cases = [
        "What's the weather in xyz123notarealcity?",  # Unknown location
        "Weather?",  # Very short query
        "Tell me about weather in multiple cities: New York, Tokyo, London, Paris",  # Multiple locations
    ]
    
    successful = 0
    for i, query in enumerate(edge_cases):
        try:
            output, _ = wrapper.run(query, thread_id=f"robustness-{i}")
            if output and len(output) > 10:  # Meaningful response
                successful += 1
        except Exception as e:
            print(f"  Edge case failed: {query[:50]}... - {str(e)[:50]}")
    
    robustness_score = successful / len(edge_cases) if edge_cases else 0.0
    assert robustness_score >= 0.5, f"Robustness score {robustness_score} below threshold"
    print(f"✓ Robustness - Score: {robustness_score:.3f} ({successful}/{len(edge_cases)} successful)")

def test_error_handling():
    """Test if agent handles errors gracefully"""
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    
    # Test with potentially problematic queries
    problematic_queries = [
        "What's the weather?",  # No location specified
        "Weather in 12345",  # Invalid location format
    ]
    
    handled_gracefully = 0
    for i, query in enumerate(problematic_queries):
        try:
            output, _ = wrapper.run(query, thread_id=f"error-{i}")
            # Check if response acknowledges the issue or provides helpful guidance
            if output and (len(output) > 10 or "location" in output.lower() or "city" in output.lower()):
                handled_gracefully += 1
        except Exception:
            pass
    
    error_handling_score = handled_gracefully / len(problematic_queries) if problematic_queries else 0.0
    assert error_handling_score >= 0.5, f"Error handling score {error_handling_score} below threshold"
    print(f"✓ Error Handling - Score: {error_handling_score:.3f} ({handled_gracefully}/{len(problematic_queries)} handled)")

# ==================== Run All Tests ====================

def run_all_tests():
    """Execute all weather agent test cases"""
    print("\n" + "="*70)
    print("Running Comprehensive Weather Agent Tests with DeepEval")
    print("="*70 + "\n")
    
    tests = [
        # Quality Tests
        #("Answer Relevancy", test_answer_relevancy),
        #("Faithfulness", test_faithfulness),
        #("Contextual Precision", test_contextual_precision),
        #("Hallucination", test_hallucination),
        
        # Tool Call Tests
        #("Tool Call Accuracy", test_tool_call_accuracy),
        #("Tool Argument Accuracy", test_tool_argument_accuracy),
        
        # Consistency & Coherence
        #("Consistency", test_consistency),
        #("Coherence", test_coherence),
        
        # Language Quality
        #("Fluency", test_fluency),
        #("Clarity", test_clarity),
        
        # Accuracy Tests
        #("Location Accuracy", test_location_accuracy),
        #("Weather Data Presence", test_weather_data_presence),
        
        # Ethics & Bias
        #("Bias", test_bias),
        ("Toxicity", test_toxicity),
        
        # Robustness
        #("Robustness", test_robustness),
        #("Error Handling", test_error_handling),
    ]
    
    passed = 0
    failed = 0
    results = []
    
    for name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print('='*70)
        try:
            test_func()
            passed += 1
            results.append((name, "PASSED", "✓"))
        except Exception as e:
            print(f"✗ {name} Failed: {str(e)}")
            failed += 1
            results.append((name, "FAILED", "✗"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\nTotal Tests: {len(tests)}")
    print(f"Passed: {passed} ({passed/len(tests)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(tests)*100:.1f}%)")
    
    print("\n" + "-"*70)
    print("Detailed Results:")
    print("-"*70)
    for name, status, symbol in results:
        print(f"{symbol} {name}: {status}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    # Example: Run the weather agent interactively
    print("Example Weather Agent Interaction:")
    print("-" * 70)
    
    agent = create_agent()
    wrapper = WeatherAgentTestWrapper(agent)
    query = "What's the weather like in New York?"
    output, contexts = wrapper.run(query)
    print(f"Query: {query}")
    print(f"Response: {output}")
    print(f"Tool Contexts: {contexts}")
    print()
    
    # Run all tests
    run_all_tests()


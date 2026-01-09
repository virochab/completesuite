"""
LangGraph Agent with Weather Check Tool
An agentic AI that uses weather_check tool and provides conversation/tool call extraction
"""

from typing import TypedDict, Annotated, Sequence, List, Dict, Any
import operator
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from dotenv import load_dotenv
load_dotenv()
import os

# Initialize OpenWeatherMap API wrapper
# In newer LangChain versions, it reads OPENWEATHERMAP_API_KEY from environment automatically
# The API key should be set in .env file or environment variable
try:
    weather_wrapper = OpenWeatherMapAPIWrapper()
except Exception as e:
    # Fallback: if initialization fails (e.g., API key not set), create a dummy wrapper
    print(f"Warning: OpenWeatherMapAPIWrapper initialization failed: {e}")
    print("Weather API will use simulated data instead.")
    weather_wrapper = None

# ==================== Tool Definitions ====================

@tool
def weather_check(location: str) -> str:
    """Get the current weather for a specific location.
    
    Args:
        location: The city or location name (e.g., "New York", "London", "Tokyo")
    
    Returns:
        A string describing the current weather conditions
    """
    # Simulated weather data
    weather_database = {
        "new york": "The current weather in New York is 72°F (22°C), partly cloudy with light winds.",
        "london": "The current weather in London is 15°C (59°F), overcast with light rain expected.",
        "tokyo": "The current weather in Tokyo is 25°C (77°F), sunny and clear skies.",
        "paris": "The current weather in Paris is 18°C (64°F), cloudy with a chance of showers.",
        "sydney": "The current weather in Sydney is 22°C (72°F), sunny with clear skies.",
        "los angeles": "The current weather in Los Angeles is 75°F (24°C), sunny and warm.",
        "chicago": "The current weather in Chicago is 65°F (18°C), partly cloudy with moderate winds.",
        "miami": "The current weather in Miami is 82°F (28°C), hot and humid with scattered thunderstorms.",
    }
    
    location_lower = location.lower()
    
    # Check for exact or partial matches
    for city, weather_info in weather_database.items():
        if city in location_lower or location_lower in city:
            return weather_info
    
    # Default response for unknown locations
    return f"Weather information for {location} is currently unavailable. Please try a major city name."


@tool
def get_current_weather(location: str) -> str:
    """
    Fetches the current weather data for a given city and state.
    The input should be a city name (e.g., "San Francisco, CA").
    """
    if weather_wrapper is None:
        return f"Weather API is not configured. Using simulated data for {location}."
    try:
        return weather_wrapper.run(location)
    except Exception as e:
        return f"Error fetching weather for {location}: {e}"
# ==================== Agent State Definition ====================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ==================== Agent Setup ====================

# Create tools list
tools = [get_current_weather]

# Initialize the model with tools
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0).bind_tools(tools)

# Define the function that calls the model
def call_model(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Define conditional edge logic
def should_continue(state: AgentState):
    """Determine if we should continue to tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# Build the graph
def create_agent():
    """Create and compile the LangGraph agent"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ==================== Conversation and Tool Call Extraction ====================

def _extract_tool_call_info(tool_call) -> Dict[str, Any]:
    """Helper function to extract tool call information from various formats"""
    # Handle dict format
    if isinstance(tool_call, dict):
        # Check for OpenAI format with "function" key
        if "function" in tool_call:
            func_info = tool_call["function"]
            return {
                "name": func_info.get("name") if isinstance(func_info, dict) else getattr(func_info, "name", None),
                "args": func_info.get("arguments", {}) if isinstance(func_info, dict) else getattr(func_info, "arguments", {}),
                "id": tool_call.get("id")
            }
        # Direct format
        return {
            "name": tool_call.get("name"),
            "args": tool_call.get("args", tool_call.get("arguments", {})),
            "id": tool_call.get("id")
        }
    
    # Handle object format
    # Try to get name - check for function.name pattern first
    name = None
    if hasattr(tool_call, "function"):
        func = tool_call.function
        if isinstance(func, dict):
            name = func.get("name")
        elif hasattr(func, "name"):
            name = func.name
    
    if not name:
        if hasattr(tool_call, "name"):
            name = tool_call.name
        elif hasattr(tool_call, "get"):
            name = tool_call.get("name")
        elif hasattr(tool_call, "__dict__"):
            name = tool_call.__dict__.get("name")
    
    # Try to get args - check for function.arguments pattern first
    args = {}
    if hasattr(tool_call, "function"):
        func = tool_call.function
        if isinstance(func, dict):
            args_str = func.get("arguments", "{}")
            if isinstance(args_str, str):
                try:
                    import json
                    args = json.loads(args_str)
                except:
                    args = {}
            else:
                args = args_str
        elif hasattr(func, "arguments"):
            args_val = func.arguments
            if isinstance(args_val, str):
                try:
                    import json
                    args = json.loads(args_val)
                except:
                    args = {}
            else:
                args = args_val
    
    if not args:
        if hasattr(tool_call, "args"):
            args = tool_call.args if not callable(tool_call.args) else {}
        elif hasattr(tool_call, "arguments"):
            args_val = tool_call.arguments
            if isinstance(args_val, str):
                try:
                    import json
                    args = json.loads(args_val)
                except:
                    args = {}
            else:
                args = args_val
        elif hasattr(tool_call, "get"):
            args = tool_call.get("args", tool_call.get("arguments", {}))
        elif hasattr(tool_call, "__dict__"):
            args = tool_call.__dict__.get("args", tool_call.__dict__.get("arguments", {}))
    
    # Try to get id
    tool_id = None
    if hasattr(tool_call, "id"):
        tool_id = tool_call.id if not callable(tool_call.id) else None
    elif hasattr(tool_call, "get"):
        tool_id = tool_call.get("id")
    elif hasattr(tool_call, "__dict__"):
        tool_id = tool_call.__dict__.get("id")
    
    return {
        "name": name,
        "args": args if isinstance(args, dict) else {},
        "id": tool_id
    }

def extract_conversation_messages(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all conversation messages from the agent result
    
    Args:
        result: The result dictionary from agent.invoke()
    
    Returns:
        List of message dictionaries with role, content, and tool_calls if applicable
    """
    messages = result.get("messages", [])
    conversation = []
    
    for msg in messages:
        msg_dict = {
            "role": None,
            "content": None,
            "tool_calls": []
        }
        
        if isinstance(msg, HumanMessage):
            msg_dict["role"] = "user"
            msg_dict["content"] = msg.content
        elif isinstance(msg, AIMessage):
            msg_dict["role"] = "assistant"
            msg_dict["content"] = msg.content
            # Extract tool calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_call_dict = _extract_tool_call_info(tool_call)
                    msg_dict["tool_calls"].append(tool_call_dict)
        elif isinstance(msg, ToolMessage):
            msg_dict["role"] = "tool"
            msg_dict["content"] = msg.content
            msg_dict["tool_call_id"] = getattr(msg, "tool_call_id", None)
            msg_dict["name"] = getattr(msg, "name", None)
        
        conversation.append(msg_dict)
    
    return conversation

def extract_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all tool calls from the agent result
    
    Args:
        result: The result dictionary from agent.invoke()
    
    Returns:
        List of tool call dictionaries
    """
    messages = result.get("messages", [])
    tool_calls = []
    
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_call_dict = _extract_tool_call_info(tool_call)
                tool_call_dict["message_content"] = msg.content
                tool_calls.append(tool_call_dict)
    
    return tool_calls

def get_conversation_trajectory(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the complete conversation trajectory including messages and tool calls
    
    Args:
        result: The result dictionary from agent.invoke()
    
    Returns:
        Dictionary with messages and tool_calls
    """
    return {
        "messages": extract_conversation_messages(result),
        "tool_calls": extract_tool_calls(result),
        "total_messages": len(result.get("messages", [])),
        "total_tool_calls": len(extract_tool_calls(result))
    }

# ==================== Main Usage Example ====================

def main():
    """Example usage of the weather agent"""
    print("="*80)
    print("WEATHER AGENT WITH LANGGRAPH")
    print("="*80)
    
    # Create the agent
    agent = create_agent()
    
    # Test queries
    test_queries = [
        "What's the weather like in New York?",
        "Can you check the weather in Tokyo?",
        "Tell me about the weather in London and Paris"
    ]
    
    config = {"configurable": {"thread_id": "weather-session-1"}}
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")
        
        # Run the agent
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        # Extract conversation trajectory
        trajectory = get_conversation_trajectory(result)
        
        # Print conversation messages
        print("Conversation Messages:")
        print("-" * 80)
        for i, msg in enumerate(trajectory["messages"], 1):
            print(f"{i}. [{msg['role'].upper()}]")
            if msg["content"]:
                print(f"   Content: {msg['content']}")
            if msg.get("tool_calls"):
                print(f"   Tool Calls: {len(msg['tool_calls'])}")
                for tc in msg["tool_calls"]:
                    print(f"      - {tc['name']}({tc['args']})")
            if msg.get("tool_call_id"):
                print(f"   Tool Call ID: {msg['tool_call_id']}")
            print()
        
        # Print tool calls summary
        print("Tool Calls Summary:")
        print("-" * 80)
        if trajectory["tool_calls"]:
            for i, tc in enumerate(trajectory["tool_calls"], 1):
                print(f"{i}. Tool: {tc.get('name', 'N/A')}")
                print(f"   Args: {tc.get('args', {})}")
                print(f"   ID: {tc.get('id', 'N/A')}")
                print()
        else:
            print("No tool calls made.")
        
        # Debug: Print raw tool calls structure
        print("Debug - Raw Tool Calls Structure:")
        print("-" * 80)
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"Message has {len(msg.tool_calls)} tool call(s)")
                for idx, tc in enumerate(msg.tool_calls):
                    print(f"  Tool Call {idx + 1}:")
                    print(f"    Type: {type(tc)}")
                    print(f"    Dir: {[x for x in dir(tc) if not x.startswith('_')]}")
                    if isinstance(tc, dict):
                        print(f"    Dict keys: {tc.keys()}")
                        print(f"    Dict content: {tc}")
                    else:
                        print(f"    Name attr: {getattr(tc, 'name', 'NOT FOUND')}")
                        if hasattr(tc, '__dict__'):
                            print(f"    __dict__: {tc.__dict__}")
                print()
        
        # Print final response
        print("Final Response:")
        print("-" * 80)
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            print(final_message.content)
        
        print(f"\nTotal Messages: {trajectory['total_messages']}")
        print(f"Total Tool Calls: {trajectory['total_tool_calls']}")
        print()

if __name__ == "__main__":
    main()


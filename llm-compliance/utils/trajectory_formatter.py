"""
Utility functions for formatting LangGraph trajectory data into standardized message formats.
"""

import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, BaseMessage,
    AIMessageChunk
)


def format_trajectory_as_messages(
    trajectory: Dict[str, Any],
    include_tool_calls: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert LangGraph trajectory to OpenAI-style message format.
    
    Args:
        trajectory: Trajectory dictionary from extract_langgraph_trajectory_from_thread
        include_tool_calls: Whether to include tool_calls in assistant messages
    
    Returns:
        List of message dictionaries in format:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "", "tool_calls": [...]},
            {"role": "tool", "content": "..."},
            {"role": "assistant", "content": "..."},
        ]
    """
    messages = []
    
    # Extract inputs and outputs from trajectory
    inputs = trajectory.get("inputs", {})
    outputs = trajectory.get("outputs", {})
    
    # Get messages from inputs (user messages)
    input_messages = inputs.get("messages", []) if isinstance(inputs, dict) else []
    
    # Get results from outputs (contains all messages including tool calls and responses)
    results = outputs.get("results", []) if isinstance(outputs, dict) else []
    
    # Process input messages (user queries)
    for msg in input_messages:
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            content = getattr(msg, 'content', str(msg))
            if content:
                messages.append({
                    "role": "user",
                    "content": content
                })
        elif isinstance(msg, (str, dict)):
            # Handle string or dict input
            if isinstance(msg, str):
                messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict) and "content" in msg:
                messages.append({"role": "user", "content": msg["content"]})
    
    # Process results (assistant messages, tool calls, tool responses)
    for result in results:
        if isinstance(result, dict):
            # Check if this result contains messages
            result_messages = result.get("messages", [])
            
            for msg in result_messages:
                # Handle AIMessage (assistant with potential tool calls)
                if isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
                    content = getattr(msg, 'content', '') or ''
                    tool_calls = getattr(msg, 'tool_calls', []) or []
                    
                    message_dict = {
                        "role": "assistant",
                        "content": content
                    }
                    
                    # Add tool_calls if present and requested
                    if include_tool_calls and tool_calls:
                        formatted_tool_calls = []
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tool_name = tc.get("name", tc.get("function", {}).get("name", "unknown"))
                                tool_args = tc.get("args", tc.get("function", {}).get("arguments", {}))
                                tool_id = tc.get("id", None)
                                
                                # Convert args to JSON string if it's a dict
                                if isinstance(tool_args, dict):
                                    tool_args_str = json.dumps(tool_args)
                                else:
                                    tool_args_str = str(tool_args)
                                
                                formatted_tool_calls.append({
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_args_str
                                    },
                                    "id": tool_id
                                })
                            else:
                                # Handle other tool call formats
                                formatted_tool_calls.append({
                                    "function": {
                                        "name": str(tc),
                                        "arguments": "{}"
                                    }
                                })
                        
                        if formatted_tool_calls:
                            message_dict["tool_calls"] = formatted_tool_calls
                    
                    messages.append(message_dict)
                
                # Handle ToolMessage (tool responses)
                elif isinstance(msg, ToolMessage) or (hasattr(msg, 'type') and msg.type == 'tool'):
                    content = getattr(msg, 'content', str(msg))
                    tool_call_id = getattr(msg, 'tool_call_id', None)
                    
                    messages.append({
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call_id
                    })
                
                # Handle other message types
                elif isinstance(msg, BaseMessage):
                    msg_type = getattr(msg, 'type', 'unknown')
                    content = getattr(msg, 'content', str(msg))
                    
                    if msg_type == 'human':
                        messages.append({"role": "user", "content": content})
                    elif msg_type == 'ai':
                        messages.append({"role": "assistant", "content": content})
                    elif msg_type == 'tool':
                        messages.append({"role": "tool", "content": content})
    
    # Fallback: If no messages found, try to extract from steps
    if not messages and outputs:
        steps = outputs.get("steps", [])
        for step in steps:
            if isinstance(step, list):
                for step_name in step:
                    # Try to infer message from step name
                    if step_name == "__start__":
                        continue
                    elif step_name == "agent":
                        # This would be an assistant message
                        messages.append({
                            "role": "assistant",
                            "content": ""
                        })
                    elif step_name == "tools":
                        # This would be a tool message
                        messages.append({
                            "role": "tool",
                            "content": ""
                        })
    
    return messages


def format_trajectory_from_messages(
    messages: List[BaseMessage],
    include_tool_calls: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert a list of LangChain messages directly to OpenAI-style format.
    
    Args:
        messages: List of LangChain BaseMessage objects
        include_tool_calls: Whether to include tool_calls in assistant messages
    
    Returns:
        List of message dictionaries in OpenAI format
    """
    formatted_messages = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            content = getattr(msg, 'content', '')
            formatted_messages.append({
                "role": "user",
                "content": content
            })
        
        elif isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
            content = getattr(msg, 'content', '') or ''
            tool_calls = getattr(msg, 'tool_calls', []) or []
            
            message_dict = {
                "role": "assistant",
                "content": content
            }
            
            if include_tool_calls and tool_calls:
                formatted_tool_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tool_name = tc.get("name", tc.get("function", {}).get("name", "unknown"))
                        tool_args = tc.get("args", tc.get("function", {}).get("arguments", {}))
                        tool_id = tc.get("id", None)
                        
                        if isinstance(tool_args, dict):
                            tool_args_str = json.dumps(tool_args)
                        else:
                            tool_args_str = str(tool_args)
                        
                        formatted_tool_calls.append({
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args_str
                            },
                            "id": tool_id
                        })
                
                if formatted_tool_calls:
                    message_dict["tool_calls"] = formatted_tool_calls
            
            formatted_messages.append(message_dict)
        
        elif isinstance(msg, ToolMessage) or (hasattr(msg, 'type') and msg.type == 'tool'):
            content = getattr(msg, 'content', str(msg))
            tool_call_id = getattr(msg, 'tool_call_id', None)
            
            formatted_messages.append({
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id
            })
    
    return formatted_messages


def get_messages_from_agent_result(
    agent_result: Dict[str, Any],
    include_tool_calls: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract and format messages from LangGraph agent invoke result.
    
    Args:
        agent_result: Result dictionary from agent.invoke()
        include_tool_calls: Whether to include tool_calls in assistant messages
    
    Returns:
        List of message dictionaries in OpenAI format
    """
    messages = agent_result.get("messages", [])
    return format_trajectory_from_messages(messages, include_tool_calls)


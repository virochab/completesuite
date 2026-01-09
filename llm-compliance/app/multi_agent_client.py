"""Multi-Agent System client wrapper for LangGraph agent testing."""

import json
import os
import uuid
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Import agentevals for trajectory extraction
try:
    from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread
except ImportError:
    extract_langgraph_trajectory_from_thread = None

# Import the multi-agent system components
try:
    from app.graph_multi_agent import (
        build_multi_agent_graph,
        get_conversation_trajectory,
        extract_tool_calls,
        extract_conversation_messages,
        wikipedia_query,
        get_current_weather
    )
except ImportError:
    # Fallback for relative import if running from same directory
    from graph_multi_agent import (
        build_multi_agent_graph,
        get_conversation_trajectory,
        extract_tool_calls,
        extract_conversation_messages,
        wikipedia_query,
        get_current_weather
    )

load_dotenv()


class MultiAgentClient:
    """Client wrapper for Multi-Agent LangGraph System testing."""
    
    def __init__(self, thread_id: Optional[str] = None):
        """
        Initialize the Multi-Agent Client.
        
        Args:
            thread_id: Optional thread ID for conversation continuity.
                      If None, uses a default thread ID.
        """
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        self.agent = build_multi_agent_graph(checkpointer=memory)
        self.thread_id = thread_id or "multi-agent-session-1"
        self.config = {"configurable": {"thread_id": self.thread_id}}
        
        # Cache tool descriptions for ArgumentCorrectnessMetric
        self._tool_descriptions = {}
        try:
            # Get tool descriptions
            tools = [wikipedia_query, get_current_weather]
            for tool in tools:
                # Get description from tool.description or docstring
                description = getattr(tool, 'description', None)
                if not description and hasattr(tool, 'func'):
                    description = tool.func.__doc__
                
                # Keep description simple and natural for ArgumentCorrectnessMetric
                if description:
                    # Clean up the description - keep it simple and natural
                    description = description.strip()
                    # Normalize whitespace but keep the natural flow
                    description = " ".join(description.split())
                    
                    # Enhance description to explicitly mention parameter names
                    try:
                        if hasattr(tool, 'args_schema') and tool.args_schema:
                            # Get schema to find parameter names
                            if hasattr(tool.args_schema, 'model_json_schema'):
                                schema = tool.args_schema.model_json_schema()
                            elif hasattr(tool.args_schema, 'schema'):
                                schema = tool.args_schema.schema()
                            else:
                                schema = {}
                            
                            if schema and 'properties' in schema:
                                param_names = list(schema['properties'].keys())
                                required_params = schema.get('required', [])
                                
                                # Always add explicit parameter name mention for ArgumentCorrectnessMetric
                                if required_params:
                                    if len(required_params) == 1:
                                        param_desc = f" The input dictionary must contain a '{required_params[0]}' key."
                                    else:
                                        param_list = ", ".join([f"'{p}'" for p in required_params[:-1]])
                                        param_desc = f" The input dictionary must contain keys: {param_list} and '{required_params[-1]}'."
                                    description = description + param_desc
                                elif param_names:
                                    if len(param_names) == 1:
                                        param_desc = f" The input dictionary must contain a '{param_names[0]}' key."
                                    else:
                                        param_list = ", ".join([f"'{p}'" for p in param_names[:-1]])
                                        param_desc = f" The input dictionary must contain keys: {param_list} and '{param_names[-1]}'."
                                    description = description + param_desc
                    except Exception:
                        pass  # If enhancement fails, use original description
                
                if description:
                    self._tool_descriptions[tool.name] = description.strip()
        except Exception:
            pass
    
    def ask(self, query: str):
        """
        Send a query to the multi-agent system and return response.
        
        Args:
            query: The user's query/question
        
        Returns:
            Response object with text, citations, tools_called, and trace attributes
        """
        try:
            # Invoke the agent with the query
            result = self.agent.invoke(
                {"query": query, "trajectory": []},
                config=self.config
            )
            
            # Extract the final answer
            answer = result.get("answer", "")
            response_text = answer if answer else "No answer provided."
            
            # Extract tool calls information from trajectory
            tool_calls = extract_tool_calls(result)
            tools_called = []
            
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_description = self._tool_descriptions.get(tool_name)
                
                tools_called.append({
                    "name": tool_name,
                    "args": tc.get("args", {}),
                    "id": tc.get("id", None),
                    "agent": tc.get("agent", None),  # Which agent made this call
                    "description": tool_description  # Include description for ArgumentCorrectnessMetric
                })
            
            # Extract actual LangGraph trajectory from checkpointer
            trajectory = None
            if extract_langgraph_trajectory_from_thread:
                try:
                    trajectory = extract_langgraph_trajectory_from_thread(
                        self.agent,
                        self.config
                    )
                except Exception as e:
                    # Fallback to get_conversation_trajectory if extraction fails
                    print(f"Warning: Failed to extract LangGraph trajectory: {e}")
                    trajectory = get_conversation_trajectory(result)
            else:
                # Fallback if agentevals is not available
                trajectory = get_conversation_trajectory(result)
            
            # Create response object with required attributes
            class Response:
                def __init__(self, text: str, citations: Optional[List] = None, 
                           tools_called: Optional[List] = None, trace: Optional[Dict[str, Any]] = None):
                    self.text = text
                    self.citations = citations or []
                    self.tools_called = tools_called or []
                    self.trace = trace or {}  # Include full LangGraph trajectory as trace
            
            return Response(
                text=response_text,
                citations=[],  # Multi-agent system doesn't use citations
                tools_called=tools_called,
                trace=trajectory  # Include full LangGraph trajectory as trace
            )
            
        except Exception as e:
            # Return error response if agent call fails
            class Response:
                def __init__(self, text: str):
                    self.text = f"Error: {str(e)}"
                    self.citations = []
                    self.tools_called = []
                    self.trace = {}
            
            return Response(text=f"Error: {str(e)}")
    
    def get_conversation_history(self) -> Dict[str, Any]:
        """
        Get the current conversation history and trajectory.
        
        Returns:
            Dictionary with conversation messages and tool calls
        """
        try:
            # Get the current state from the agent's memory
            # Note: This requires accessing the agent's internal state
            # For now, return empty structure
            # In a full implementation, you'd retrieve from the checkpointer
            return {
                "messages": [],
                "tool_calls": [],
                "total_messages": 0,
                "total_tool_calls": 0
            }
        except Exception as e:
            return {
                "error": str(e),
                "messages": [],
                "tool_calls": []
            }
    
    def reset_conversation(self, new_thread_id: Optional[str] = None):
        """
        Reset the conversation by creating a new thread ID.
        
        Args:
            new_thread_id: Optional new thread ID. If None, generates a new one.
        """
        if new_thread_id:
            self.thread_id = new_thread_id
        else:
            self.thread_id = f"multi-agent-session-{uuid.uuid4().hex[:8]}"
        
        self.config = {"configurable": {"thread_id": self.thread_id}}
    
    def get_trace(self, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the conversation trajectory/trace from the LangGraph checkpointer.
        
        Args:
            result: Optional result dictionary from agent.invoke(). 
                   If None, extracts from current thread state.
        
        Returns:
            Dictionary with LangGraph trajectory including inputs, outputs, steps, etc.
        """
        if extract_langgraph_trajectory_from_thread:
            try:
                return extract_langgraph_trajectory_from_thread(
                    self.agent,
                    self.config
                )
            except Exception as e:
                # Fallback to get_conversation_trajectory if extraction fails
                if result:
                    return get_conversation_trajectory(result)
        
        # Fallback if agentevals is not available or extraction fails
        if result:
            return get_conversation_trajectory(result)
        else:
            # Return empty structure if no result provided
            return {
                "inputs": [],
                "outputs": {},
                "steps": [],
                "messages": [],
                "tool_calls": []
            }


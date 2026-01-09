"""Weather Agent client wrapper for LangGraph agent testing with trajectory trace extraction."""

import json
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Import the weather agent components
try:
    from app.weather_agent_langgraph import (
        create_agent,
        get_conversation_trajectory,
        extract_tool_calls,
        tools as agent_tools
    )
except ImportError:
    # Fallback for relative import if running from same directory
    from weather_agent_langgraph import (
        create_agent,
        get_conversation_trajectory,
        extract_tool_calls,
        tools as agent_tools
    )

load_dotenv()


class WeatherAgentClientWithTrace:
    """Client wrapper for Weather LangGraph Agent testing with trajectory trace extraction."""
    
    def __init__(self, thread_id: Optional[str] = None):
        """
        Initialize the Weather Agent Client with trace support.
        
        Args:
            thread_id: Optional thread ID for conversation continuity.
                      If None, uses a default thread ID.
        """
        self.agent = create_agent()
        self.thread_id = thread_id or "weather-agent-session-1"
        self.config = {"configurable": {"thread_id": self.thread_id}}
        
        # Cache tool descriptions for ArgumentCorrectnessMetric
        self._tool_descriptions = {}
        try:
            import json
            import inspect
            for tool in agent_tools:
                # Get description from tool.description or docstring
                description = getattr(tool, 'description', None)
                if not description and hasattr(tool, 'func'):
                    description = tool.func.__doc__
                
                # Keep description simple and natural for ArgumentCorrectnessMetric
                # Based on DeepEval examples, the description should be a simple statement
                # about what the tool does. ArgumentCorrectnessMetric will validate
                # that the input dictionary contains appropriate parameters based on the description.
                # Make description explicitly mention the parameter name to match the input dict key
                if description:
                    # Clean up the description - keep it simple and natural
                    description = description.strip()
                    # Normalize whitespace but keep the natural flow
                    description = " ".join(description.split())
                    
                    # Enhance description to explicitly mention parameter names
                    # This is critical for ArgumentCorrectnessMetric to validate correctly
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
                                # Format it to match what the input dictionary actually contains
                                if required_params:
                                    # Use required params first
                                    if len(required_params) == 1:
                                        # Make it very explicit: "The input must have a 'location' key"
                                        param_desc = f" The input dictionary must contain a '{required_params[0]}' key."
                                    else:
                                        param_list = ", ".join([f"'{p}'" for p in required_params[:-1]])
                                        param_desc = f" The input dictionary must contain keys: {param_list} and '{required_params[-1]}'."
                                    description = description + param_desc
                                elif param_names:
                                    # Fallback to all param names if no required specified
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
    
    def ask(self, query: str, capture_trace: bool = True, format_as_messages: bool = False):
        """
        Send a query to the weather agent and return response with trajectory trace.
        
        Args:
            query: The user's query/question
            capture_trace: If True, extract LangGraph trajectory using extract_langgraph_trajectory_from_thread
            format_as_messages: If True, format trajectory as OpenAI-style messages list
        
        Returns:
            Response object with text, citations, tools_called, trace_info (trajectory), 
            and optionally formatted_messages attributes
        """
        try:
            # Invoke the agent with the query
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=self.config
            )
            
            # Extract the final response text
            final_message = result["messages"][-1]
            response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # Extract tool calls information
            tool_calls = extract_tool_calls(result)
            tools_called = []
            
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_description = self._tool_descriptions.get(tool_name)
                
                tools_called.append({
                    "name": tool_name,
                    "args": tc.get("args", {}),
                    "id": tc.get("id", None),
                    "description": tool_description  # Include description for ArgumentCorrectnessMetric
                })
            
            # Extract trajectory if requested (using extract_langgraph_trajectory_from_thread)
            trace_info = None
            formatted_messages = None
            if capture_trace:
                trace_info = self._extract_trajectory()
                
                # Format as messages if requested
                if format_as_messages:
                    try:
                        from utils.trajectory_formatter import get_messages_from_agent_result, format_trajectory_from_messages
                        # Prefer formatting directly from agent result messages (most reliable)
                        # This extracts from the actual LangChain messages in the result
                        formatted_messages = get_messages_from_agent_result(result)
                        
                        # Fallback: if that doesn't work, try formatting from messages list directly
                        if not formatted_messages:
                            messages_list = result.get("messages", [])
                            if messages_list:
                                formatted_messages = format_trajectory_from_messages(messages_list)
                    except ImportError:
                        # Fallback: format directly from agent result messages
                        try:
                            from utils.trajectory_formatter import format_trajectory_from_messages
                            formatted_messages = format_trajectory_from_messages(result.get("messages", []))
                        except ImportError:
                            pass
                    except Exception as e:
                        # If formatting fails, log but don't fail
                        print(f"Warning: Failed to format messages: {e}")
                        formatted_messages = None
            
            # Create response object with required attributes
            class Response:
                def __init__(self, text: str, citations: Optional[List] = None, tools_called: Optional[List] = None, 
                           trace_info: Optional[Dict[str, Any]] = None, formatted_messages: Optional[List[Dict[str, Any]]] = None):
                    self.text = text
                    self.citations = citations or []
                    self.tools_called = tools_called or []
                    self.trace_info = trace_info
                    self.formatted_messages = formatted_messages
            
            return Response(
                text=response_text,
                citations=[],  # Weather agent doesn't use citations
                tools_called=tools_called,
                trace_info=trace_info,
                formatted_messages=formatted_messages
            )
            
        except Exception as e:
            # Return error response if agent call fails
            class Response:
                def __init__(self, text: str):
                    self.text = f"Error: {str(e)}"
                    self.citations = []
                    self.tools_called = []
                    self.trace_info = None
            
            return Response(text=f"Error: {str(e)}")
    
    def _extract_trajectory(self) -> Optional[Dict[str, Any]]:
        """
        Extract LangGraph trajectory from the current thread using extract_langgraph_trajectory_from_thread.
        Similar to test_graph_trajectory_strict.py implementation.
        
        Returns:
            Dictionary with trajectory information including outputs and steps, or None if extraction fails
        """
        try:
            from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread
        except ImportError:
            # If agentevals is not available, return None
            return None
        
        try:
            # Extract trajectory from the current thread using the agent and config
            extracted_trajectory = extract_langgraph_trajectory_from_thread(
                self.agent,
                self.config
            )
            
            return extracted_trajectory
        except Exception as e:
            # Return None if trajectory extraction fails
            return None
    
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
            import uuid
            self.thread_id = f"weather-agent-session-{uuid.uuid4().hex[:8]}"
        
        self.config = {"configurable": {"thread_id": self.thread_id}}


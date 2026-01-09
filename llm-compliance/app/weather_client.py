"""Weather Agent client wrapper for LangGraph agent testing."""

import json
import os
import time
from datetime import datetime, timezone
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


class WeatherAgentClient:
    """Client wrapper for Weather LangGraph Agent testing."""
    
    def __init__(self, thread_id: Optional[str] = None):
        """
        Initialize the Weather Agent Client.
        
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
    
    def ask(self, query: str):
        """
        Send a query to the weather agent and return response.
        
        Args:
            query: The user's query/question
        
        Returns:
            Response object with text, citations, and tools_called attributes
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
            
            # Create response object with required attributes
            class Response:
                def __init__(self, text: str, citations: Optional[List] = None, tools_called: Optional[List] = None):
                    self.text = text
                    self.citations = citations or []
                    self.tools_called = tools_called or []
            
            return Response(
                text=response_text,
                citations=[],  # Weather agent doesn't use citations
                tools_called=tools_called
            )
            
        except Exception as e:
            # Return error response if agent call fails
            class Response:
                def __init__(self, text: str):
                    self.text = f"Error: {str(e)}"
                    self.citations = []
                    self.tools_called = []
            
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
            import uuid
            self.thread_id = f"weather-agent-session-{uuid.uuid4().hex[:8]}"
        
        self.config = {"configurable": {"thread_id": self.thread_id}}


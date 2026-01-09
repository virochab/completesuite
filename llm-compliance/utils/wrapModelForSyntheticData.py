from deepeval.test_case import Turn
from typing import List
import sys
from pathlib import Path

# Add parent directory to Python path to enable imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.weather_client import WeatherAgentClient

# Initialize weather client (module-level for reuse across calls)
# Thread ID will be managed per conversation
_weather_clients = {}  # Dictionary to store clients by thread_id


async def model_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    """
    Model callback function for DeepEval ConversationSimulator.
    Calls the weather app with the user input and conversation history.
    
    Args:
        input: The current user input/query
        turns: List of previous conversation turns (for context)
        thread_id: Unique thread ID for conversation continuity
    
    Returns:
        Turn object with assistant response
    """
    try:
        # Get or create weather client for this thread_id
        if thread_id not in _weather_clients:
            _weather_clients[thread_id] = WeatherAgentClient(thread_id=thread_id)
        
        weather_client = _weather_clients[thread_id]
        
        # The weather client handles conversation history internally via thread_id
        # LangGraph agent maintains state through the checkpoint system
        # So we can just call ask() with the current input
        response = weather_client.ask(input)
        
        # Extract response text
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Log for debugging
        print(f"[Thread {thread_id}] Input: {input}")
        print(f"[Thread {thread_id}] Response: {response_text[:100]}...")
        
        return Turn(role="assistant", content=response_text)
        
    except Exception as e:
        # Return error response
        error_msg = f"Error calling weather app: {str(e)}"
        print(f"[Thread {thread_id}] Error: {error_msg}")
        return Turn(role="assistant", content=error_msg)
from deepeval.test_case import Turn
from typing import List
import sys
from pathlib import Path

# Add parent directory to Python path to enable imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.clientrag import RAGClient

# Initialize RAG clients (module-level for reuse across calls)
# Thread ID will be managed per conversation
_rag_clients = {}  # Dictionary to store clients by thread_id


async def model_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    """
    Model callback function for DeepEval ConversationSimulator.
    Calls the RAG PDF agent with the user input and conversation history.
    
    Args:
        input: The current user input/query
        turns: List of previous conversation turns (for context)
        thread_id: Unique thread ID for conversation continuity
    
    Returns:
        Turn object with assistant response
    """
    try:
        # Get or create RAG client for this thread_id
        if thread_id not in _rag_clients:
            # Initialize RAG client (direct mode, not API)
            # You can customize pdf_directory and vector_db_path if needed
            _rag_clients[thread_id] = RAGClient(use_api=False)
        
        rag_client = _rag_clients[thread_id]
        
        # Call the RAG client with the current input
        # The RAG client handles retrieval and LLM response
        response = rag_client.ask(input)
        
        # Extract response text
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Optionally include source file citations in the response
        if response.source_files:
            source_info = f" [Sources: {', '.join(response.source_files)}]"
            response_text = response_text + source_info
        
        # Extract retrieval context for TurnFaithfulnessMetric
        # RAGResponse has a context attribute that contains the retrieved context chunks
        retrieval_context = []
        if hasattr(response, 'context') and response.context:
            retrieval_context = response.context if isinstance(response.context, list) else [response.context]
        
        # Log for debugging
        print(f"[Thread {thread_id}] Input: {input}")
        print(f"[Thread {thread_id}] Response: {response_text[:100]}...")
        if response.source_files:
            print(f"[Thread {thread_id}] Source Files: {response.source_files}")
        if retrieval_context:
            print(f"[Thread {thread_id}] Retrieval Context: {len(retrieval_context)} chunks")
        
        # Create Turn with retrieval_context for TurnFaithfulnessMetric
        turn_kwargs = {
            "role": "assistant",
            "content": response_text
        }
        if retrieval_context:
            turn_kwargs["retrieval_context"] = retrieval_context
        
        return Turn(**turn_kwargs)
        
    except Exception as e:
        # Return error response
        error_msg = f"Error calling RAG agent: {str(e)}"
        print(f"[Thread {thread_id}] Error: {error_msg}")
        return Turn(role="assistant", content=error_msg)


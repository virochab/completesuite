"""
FastAPI wrapper for Weather Agent with trajectory trace extraction support.

This demonstrates how to use extract_langgraph_trajectory_from_thread
when the LangGraph agent is wrapped in a FastAPI application.
"""
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.weather_agent_langgraph import create_agent
from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread

app = FastAPI(title="Weather Agent API with Trace Support")

# Initialize the agent once at startup
agent = create_agent()


class QueryRequest(BaseModel):
    """Request model for weather queries"""
    query: str
    thread_id: Optional[str] = None  # Optional: if not provided, generates new one
    capture_trace: bool = True  # Whether to capture trajectory trace


class QueryResponse(BaseModel):
    """Response model with trace information"""
    response: str
    thread_id: str
    tools_called: List[Dict[str, Any]]
    trace_info: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    print("Weather Agent API started with trace support")


@app.post("/ask", response_model=QueryResponse)
async def ask_weather(
    request: QueryRequest,
    x_thread_id: Optional[str] = Header(None, alias="X-Thread-ID")
):
    """
    Ask the weather agent a question and optionally capture trajectory trace.
    
    The thread_id can be provided in:
    1. Request body (request.thread_id)
    2. Header (X-Thread-ID)
    3. Auto-generated if neither is provided
    
    Args:
        request: QueryRequest with query and optional thread_id
        x_thread_id: Optional thread ID from header
    
    Returns:
        QueryResponse with response text, thread_id, tools_called, and trace_info
    """
    try:
        # Determine thread_id: priority order is body > header > generate new
        thread_id = request.thread_id or x_thread_id or str(uuid.uuid4())
        
        # Create config with thread_id for checkpointing
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the agent
        from langchain_core.messages import HumanMessage
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config
        )
        
        # Extract response text
        final_message = result["messages"][-1]
        response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        # Extract tool calls
        from app.weather_agent_langgraph import extract_tool_calls
        tool_calls = extract_tool_calls(result)
        tools_called = []
        
        for tc in tool_calls:
            tools_called.append({
                "name": tc.get("name", "unknown"),
                "args": tc.get("args", {}),
                "id": tc.get("id", None),
            })
        
        # Extract trajectory trace if requested
        trace_info = None
        if request.capture_trace:
            try:
                trace_info = extract_langgraph_trajectory_from_thread(
                    agent,
                    config
                )
            except Exception as e:
                # Log error but don't fail the request
                print(f"Warning: Failed to extract trajectory: {e}")
                trace_info = {"error": str(e)}
        
        return QueryResponse(
            response=response_text,
            thread_id=thread_id,
            tools_called=tools_called,
            trace_info=trace_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/trace/{thread_id}")
async def get_trace(thread_id: str):
    """
    Get trajectory trace for a specific thread_id.
    
    This endpoint allows you to retrieve the trajectory trace for a previous
    conversation thread without making a new query.
    
    Args:
        thread_id: The thread ID to get trace for
    
    Returns:
        Dictionary with trajectory trace information
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Extract trajectory from the thread
        trace_info = extract_langgraph_trajectory_from_thread(
            agent,
            config
        )
        
        return {
            "thread_id": thread_id,
            "trace": trace_info
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Trace not found for thread_id {thread_id}: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "weather-agent"}


# Alternative approach: Using dependency injection for thread management
from fastapi import Depends

# In-memory thread store (in production, use Redis or database)
thread_store: Dict[str, Dict[str, Any]] = {}


def get_or_create_thread_id(
    thread_id: Optional[str] = Query(None),
    x_thread_id: Optional[str] = Header(None, alias="X-Thread-ID")
) -> str:
    """
    Dependency to get or create thread_id for request.
    
    This can be used as a FastAPI dependency to manage thread IDs.
    """
    if thread_id:
        return thread_id
    if x_thread_id:
        return x_thread_id
    # Generate new thread_id
    new_thread_id = str(uuid.uuid4())
    thread_store[new_thread_id] = {"created_at": "now"}
    return new_thread_id


@app.post("/ask-v2", response_model=QueryResponse)
async def ask_weather_v2(
    request: QueryRequest,
    thread_id: str = Depends(get_or_create_thread_id)
):
    """
    Alternative endpoint using dependency injection for thread management.
    
    This version uses FastAPI's dependency injection to handle thread_id.
    """
    try:
        # Create config with thread_id
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the agent
        from langchain_core.messages import HumanMessage
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config
        )
        
        # Extract response text
        final_message = result["messages"][-1]
        response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        # Extract tool calls
        from app.weather_agent_langgraph import extract_tool_calls
        tool_calls = extract_tool_calls(result)
        tools_called = []
        
        for tc in tool_calls:
            tools_called.append({
                "name": tc.get("name", "unknown"),
                "args": tc.get("args", {}),
                "id": tc.get("id", None),
            })
        
        # Extract trajectory trace if requested
        trace_info = None
        if request.capture_trace:
            try:
                trace_info = extract_langgraph_trajectory_from_thread(
                    agent,
                    config
                )
            except Exception as e:
                print(f"Warning: Failed to extract trajectory: {e}")
                trace_info = {"error": str(e)}
        
        return QueryResponse(
            response=response_text,
            thread_id=thread_id,
            tools_called=tools_called,
            trace_info=trace_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


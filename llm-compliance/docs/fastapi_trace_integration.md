# FastAPI Integration with LangGraph Trajectory Traces

This document explains how to use `extract_langgraph_trajectory_from_thread` when wrapping a LangGraph agent in FastAPI.

## Key Concepts

1. **Thread ID Management**: Each conversation needs a unique `thread_id` for checkpointing
2. **Config Object**: The `config` dict with `thread_id` must be consistent between invocation and trace extraction
3. **Agent Instance**: The same agent instance used for invocation must be used for trace extraction

## Implementation Approaches

### Approach 1: Extract Trace After Invocation (Recommended)

```python
from fastapi import FastAPI
from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread

app = FastAPI()
agent = create_agent()  # Initialize once

@app.post("/ask")
async def ask(query: str, thread_id: str = None):
    # Generate or use provided thread_id
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Invoke agent
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    
    # Extract trace immediately after invocation
    trace_info = extract_langgraph_trajectory_from_thread(agent, config)
    
    return {"response": result, "trace": trace_info}
```

### Approach 2: Separate Trace Endpoint

```python
@app.get("/trace/{thread_id}")
async def get_trace(thread_id: str):
    """Retrieve trace for a previous conversation"""
    config = {"configurable": {"thread_id": thread_id}}
    trace_info = extract_langgraph_trajectory_from_thread(agent, config)
    return {"trace": trace_info}
```

### Approach 3: Using Request Headers for Thread ID

```python
from fastapi import Header

@app.post("/ask")
async def ask(
    query: str,
    x_thread_id: Optional[str] = Header(None, alias="X-Thread-ID")
):
    thread_id = x_thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    trace_info = extract_langgraph_trajectory_from_thread(agent, config)
    
    return {
        "response": result,
        "thread_id": thread_id,  # Return so client can use for follow-up
        "trace": trace_info
    }
```

## Important Considerations

### 1. Thread ID Persistence

- **In-memory**: Works for single-server deployments
- **Redis/Database**: Required for multi-server deployments
- **Client-side**: Client can manage thread_id and send in headers

### 2. Agent Instance Lifecycle

```python
# ✅ GOOD: Initialize agent once at startup
agent = create_agent()

@app.on_event("startup")
async def startup():
    global agent
    agent = create_agent()

# ❌ BAD: Creating new agent per request
@app.post("/ask")
async def ask(query: str):
    agent = create_agent()  # Don't do this!
```

### 3. Config Consistency

The `config` dict must be identical between:
- Agent invocation: `agent.invoke(..., config=config)`
- Trace extraction: `extract_langgraph_trajectory_from_thread(agent, config)`

```python
# ✅ GOOD: Same config object
config = {"configurable": {"thread_id": thread_id}}
result = agent.invoke(..., config=config)
trace = extract_langgraph_trajectory_from_thread(agent, config)

# ❌ BAD: Different config objects
result = agent.invoke(..., config={"configurable": {"thread_id": thread_id}})
trace = extract_langgraph_trajectory_from_thread(agent, {"configurable": {"thread_id": thread_id}})
```

### 4. Error Handling

```python
try:
    trace_info = extract_langgraph_trajectory_from_thread(agent, config)
except Exception as e:
    # Log error but don't fail the request
    logger.warning(f"Failed to extract trace: {e}")
    trace_info = None
```

## Example Usage

### Client Code (Python)

```python
import requests

# First request - new conversation
response = requests.post(
    "http://localhost:8000/ask",
    json={"query": "What's the weather in New York?", "capture_trace": True},
    headers={"X-Thread-ID": "my-thread-123"}
)

data = response.json()
print(f"Response: {data['response']}")
print(f"Thread ID: {data['thread_id']}")
print(f"Trace steps: {data['trace_info']['outputs']['steps']}")

# Follow-up request - same conversation
response2 = requests.post(
    "http://localhost:8000/ask",
    json={"query": "What about Tokyo?"},
    headers={"X-Thread-ID": data['thread_id']}
)
```

### Client Code (JavaScript)

```javascript
// First request
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-Thread-ID': 'my-thread-123'
  },
  body: JSON.stringify({
    query: "What's the weather in New York?",
    capture_trace: true
  })
});

const data = await response.json();
console.log('Response:', data.response);
console.log('Trace steps:', data.trace_info?.outputs?.steps);

// Get trace separately
const traceResponse = await fetch(`http://localhost:8000/trace/${data.thread_id}`);
const traceData = await traceResponse.json();
```

## Best Practices

1. **Always return thread_id** to client so they can maintain conversation context
2. **Make trace extraction optional** via `capture_trace` parameter (it has overhead)
3. **Use consistent thread_id** across requests for multi-turn conversations
4. **Handle trace extraction errors gracefully** - don't fail the request if trace fails
5. **Consider caching** trace results if they're expensive to compute

## Performance Considerations

- Trace extraction reads from checkpoint storage, which may have latency
- Consider async trace extraction if not needed immediately
- For high-throughput scenarios, extract traces asynchronously or on-demand


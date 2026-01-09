# DeepEval MCP Testing - Setup and Run Guide

This guide explains how to set up and run DeepEval MCP tests.

## Prerequisites

1. Python 3.11+ with virtual environment activated
2. Required packages installed
3. API keys configured

## Step 1: Install Dependencies

Install all required packages:

```bash
pip install mcp anthropic openai deepeval fastmcp uvicorn python-dotenv pytest pytest-asyncio
```

Or if you have a requirements file, add these dependencies.

## Step 2: Set Up Environment Variables

Create a `.env` file in the `llm-compliance` directory (or set environment variables):

```bash
# Required: Choose one LLM provider
OPENAI_API_KEY=your-openai-api-key-here
# OR
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: LLM Provider Selection (default: openai)
LLM_PROVIDER=openai  # or "anthropic"

# Optional: Model Selection
OPENAI_MODEL=gpt-4o-mini  # default
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # default

# Optional: MCP Server URL (default: http://localhost:8000/github/mcp)
MCP_SERVER_URL=http://localhost:8000/github/mcp
```

## Step 3: Start the MCP Server

In a **separate terminal**, start the demo MCP server:

```bash
cd llm-compliance/tests/mcp
python deepeval_mcp_demo.py
```

You should see:
```
============================================================
Starting Demo MCP Server for DeepEval Testing
============================================================
Server URL: http://localhost:8000/github/mcp

Available tools:
  - search_repositories: Search GitHub repos
  - get_user_info: Get user information
  - get_trending_repos: Get trending repositories

Press Ctrl+C to stop
============================================================
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal running** - the server must be running for tests to work.

## Step 4: Run the Tests

In a **new terminal** (with the virtual environment activated), run the tests:

```bash
cd llm-compliance
pytest tests/mcp/test_deepeval_mcp.py -v
```

### Run Specific Tests

Run a specific test:
```bash
pytest tests/mcp/test_deepeval_mcp.py::test_mcp_basic_usage -v
pytest tests/mcp/test_deepeval_mcp.py::test_mcp_use_metric -v
pytest tests/mcp/test_deepeval_mcp.py::test_mcp_tool_calling -v
```

### Run with Verbose Output

```bash
pytest tests/mcp/test_deepeval_mcp.py -v -s
```

The `-s` flag shows print statements and detailed output.

## Step 5: Verify Results

The tests will:
1. Connect to the MCP server
2. List available tools
3. Process queries using the LLM (OpenAI or Anthropic)
4. Call MCP tools as needed
5. Create DeepEval test cases
6. Evaluate with MCPUseMetric

Expected output:
```
✓ Connected and initialized
✓ Found 3 tools
→ Calling OpenAI API (gpt-4o-mini)...
✓ OpenAI response: ...
→ Executing search_repositories with args: ...
✓ Tool result: ...
✓ Test case created with 1 tool calls
```

## Troubleshooting

### Error: "FastAPI service not running"
- Make sure the MCP server is running in a separate terminal
- Check that it's accessible at `http://localhost:8000/github/mcp`

### Error: "API key not found"
- Verify your `.env` file has the correct API key
- Check that `python-dotenv` is installed and loading the file

### Error: "Connection refused"
- Ensure the MCP server is running on port 8000
- Check firewall settings if using a different host

### Error: "No tools available"
- Verify the MCP server started successfully
- Check server logs for errors

## Test Files

- **`test_deepeval_mcp.py`**: Main test file with DeepEval integration
- **`deepeval_mcp_demo.py`**: Demo MCP server (GitHub tools)
- **`demo_mcp_server.py`**: Simple demo server (weather/calculator tools)

## Customization

### Use a Different MCP Server

Set the `MCP_SERVER_URL` environment variable:
```bash
export MCP_SERVER_URL=http://localhost:8000/custom/mcp
pytest tests/mcp/test_deepeval_mcp.py -v
```

### Use Anthropic Instead of OpenAI

Set in `.env`:
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key-here
```

### Use a Different Model

Set in `.env`:
```bash
OPENAI_MODEL=gpt-4-turbo
# or
ANTHROPIC_MODEL=claude-3-opus-20240229
```

## Next Steps

- Modify test queries in `test_deepeval_mcp.py`
- Add custom MCP tools to `deepeval_mcp_demo.py`
- Create additional test cases for your specific use cases


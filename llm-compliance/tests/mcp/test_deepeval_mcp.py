"""
DeepEval MCP Testing
Tests MCP server integration using DeepEval's MCPUseMetric.

Install: pip install mcp anthropic openai deepeval fastmcp uvicorn python-dotenv
Run: pytest tests/mcp/test_deepeval_mcp.py -v
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Optional, List, Literal

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client  # type: ignore[import-untyped]
from dotenv import load_dotenv
from deepeval.test_case import MCPServer, MCPToolCall, LLMTestCase
from deepeval.metrics import MCPUseMetric
from deepeval import evaluate

load_dotenv()

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Conditional imports
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class MCPDeepEvalTester:
    """Test MCP server integration with DeepEval."""
    
    def __init__(
        self, 
        mcp_url: str,
        llm_provider: Literal["openai", "anthropic"] = "openai"
    ):
        """
        Initialize MCP tester.
        
        Args:
            mcp_url: HTTP URL for MCP server (for HTTP/streamable-http transport)
            llm_provider: LLM provider to use ("openai" or "anthropic")
        """
        self.mcp_url = mcp_url
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm_provider = llm_provider
        self.mcp_servers: List[MCPServer] = []
        self.tools_called: List[MCPToolCall] = []
        
        # Initialize LLM client based on provider
        if llm_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif llm_provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
            self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Use 'openai' or 'anthropic'")
    
    async def connect(self):
        """Connect to MCP server and initialize."""
        # HTTP/streamable-http transport
        print(f"Connecting to MCP server at {self.mcp_url}...")
        transport = await self.exit_stack.enter_async_context(
            streamablehttp_client(self.mcp_url)
        )
        read, write, _ = transport
        
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        
        await self.session.initialize()
        print("✓ Connected and initialized")
        
        # List available tools
        tool_list = await self.session.list_tools()
        print(f"✓ Found {len(tool_list.tools)} tools")
        
        # Create MCPServer for DeepEval
        self.mcp_servers.append(
            MCPServer(
                server_name=self.mcp_url,
                transport="streamable-http",
                available_tools=tool_list.tools,
            )
        )
        
        return tool_list.tools
    
    async def process_query(self, query: str) -> tuple[str, List[MCPToolCall]]:
        """Process a query using LLM (OpenAI or Anthropic) and MCP tools."""
        print(f"\nProcessing query: {query}")
        
        # Clear previous tools called
        self.tools_called.clear()
        
        # Get available tools
        tool_response = await self.session.list_tools()
        
        if self.llm_provider == "openai":
            return await self._process_query_openai(query, tool_response.tools)
        else:
            return await self._process_query_anthropic(query, tool_response.tools)
    
    async def _process_query_openai(self, query: str, tools) -> tuple[str, List[MCPToolCall]]:
        """Process query using OpenAI."""
        messages = [{"role": "user", "content": query}]
        response_text = []
        
        # Convert MCP tools to OpenAI function format
        functions = []
        for tool in tools:
            functions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {},
                }
            })
        
        # Call OpenAI with tools
        print(f"→ Calling OpenAI API ({self.model})...")
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=functions if functions else None,
            tool_choice="auto",
            max_tokens=1000,
        )
        
        # Add assistant response to messages
        assistant_message = response.choices[0].message
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in (assistant_message.tool_calls or [])
            ]
        })
        
        # Extract text response
        if assistant_message.content:
            response_text.append(assistant_message.content)
            print(f"✓ OpenAI response: {assistant_message.content[:100]}...")
        
        # Process tool calls
        tool_calls = assistant_message.tool_calls or []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            
            print(f"→ Executing {tool_name} with args: {tool_args}")
            result = await self.session.call_tool(tool_name, tool_args)
            
            # Extract result content
            result_content = ""
            if hasattr(result, 'content'):
                if isinstance(result.content, list):
                    result_content = " ".join([str(item) for item in result.content])
                else:
                    result_content = str(result.content)
            else:
                result_content = str(result)
            
            print(f"✓ Tool result: {result_content[:150]}...")
            
            # Add tool result to messages for potential follow-up
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_content,
            })
            
            tool_called = MCPToolCall(
                name=tool_name,
                args=tool_args,
                result=result
            )
            self.tools_called.append(tool_called)
        
        # If there were tool calls, get final response
        if tool_calls:
            print("→ Getting final response after tool execution...")
            final_response_obj = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
            )
            final_content = final_response_obj.choices[0].message.content
            if final_content:
                response_text.append(final_content)
                print(f"✓ Final response: {final_content[:100]}...")
        
        final_response = "\n".join(response_text) if response_text else "No text response"
        return final_response, self.tools_called.copy()
    
    async def _process_query_anthropic(self, query: str, tools) -> tuple[str, List[MCPToolCall]]:
        """Process query using Anthropic Claude."""
        messages = [{"role": "user", "content": query}]
        response_text = []
        
        # Convert MCP tools to Anthropic format
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]
        
        # Call Claude with tools
        print(f"→ Calling Anthropic API ({self.model})...")
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )
        
        # Process response and execute tool calls
        tool_uses = []
        for content in response.content:
            if content.type == "text":
                response_text.append(content.text)
                print(f"✓ Claude response: {content.text[:100]}...")
            elif content.type == "tool_use":
                tool_uses.append(content)
                print(f"→ Claude wants to use tool: {content.name}")
        
        # Execute tool calls
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_args = tool_use.input
            tool_id = tool_use.id
            
            print(f"→ Executing {tool_name} with args: {tool_args}")
            result = await self.session.call_tool(tool_name, tool_args)
            print(f"✓ Tool result: {str(result.content)[:150]}...")
            
            tool_called = MCPToolCall(
                name=tool_name,
                args=tool_args,
                result=result
            )
            self.tools_called.append(tool_called)
        
        final_response = "\n".join(response_text) if response_text else "No text response"
        return final_response, self.tools_called.copy()
    
    async def create_test_case(self, query: str, actual_output: str) -> LLMTestCase:
        """Create a DeepEval test case."""
        return LLMTestCase(
            input=query,
            actual_output=actual_output,
            mcp_servers=self.mcp_servers,
            mcp_tools_called=self.tools_called.copy(),
        )
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.exit_stack.aclose()


@pytest.fixture(scope="module")
def mcp_url():
    """MCP server URL for testing (HTTP transport)."""
    # Default to GitHub demo server
    return os.getenv("MCP_SERVER_URL", "http://localhost:8000/github/mcp")


@pytest.fixture(scope="module")
def llm_provider():
    """LLM provider to use for testing."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider not in ["openai", "anthropic"]:
        provider = "openai"
    return provider


@pytest.mark.asyncio
async def test_mcp_basic_usage(mcp_url, llm_provider):
    """Test basic MCP usage with DeepEval."""
    tester = MCPDeepEvalTester(mcp_url=mcp_url, llm_provider=llm_provider)
    
    try:
        # Connect to server
        tools = await tester.connect()
        assert len(tools) > 0, "No tools available from MCP server"
        
        # Process a query
        query = "What's the weather in London?"
        response, tools_called = await tester.process_query(query)
        
        assert response, "No response received"
        print(f"\n✓ Response: {response[:200]}...")
        
        # Create test case
        test_case = await tester.create_test_case(query, response)
        
        assert test_case.input == query
        assert test_case.actual_output == response
        assert len(test_case.mcp_servers) > 0
        print(f"✓ Test case created with {len(test_case.mcp_tools_called)} tool calls")
        
    finally:
        await tester.cleanup()


@pytest.mark.asyncio
async def test_mcp_use_metric(mcp_url, llm_provider):
    """Test MCP usage with DeepEval's MCPUseMetric."""
    tester = MCPDeepEvalTester(mcp_url=mcp_url, llm_provider=llm_provider)
    
    try:
        # Connect to server
        await tester.connect()
        
        # Process queries that should trigger tool usage
        test_queries = [
            "What's the weather in Paris?",
            "Calculate 15 + 27",
            "Get a greeting in Spanish for Alice",
        ]
        
        test_cases = []
        for query in test_queries:
            response, _ = await tester.process_query(query)
            test_case = await tester.create_test_case(query, response)
            test_cases.append(test_case)
            print(f"\n✓ Created test case for: {query}")
        
        # Evaluate with MCPUseMetric (use appropriate model based on provider)
        eval_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if llm_provider == "openai" else os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        mcp_use_metric = MCPUseMetric(model=eval_model)
        
        print("\n" + "="*60)
        print("Running DeepEval evaluation...")
        print("="*60)
        
        results = evaluate(test_cases, [mcp_use_metric])
        
        print(f"\n✓ Evaluation complete")
        print(f"  Test cases: {len(test_cases)}")
        print(f"  Results: {results}")
        
        # Assert that evaluation completed
        assert results is not None
        
    finally:
        await tester.cleanup()


@pytest.mark.asyncio
async def test_mcp_tool_calling(mcp_url, llm_provider):
    """Test that MCP tools are actually called."""
    tester = MCPDeepEvalTester(mcp_url=mcp_url, llm_provider=llm_provider)
    
    try:
        # Connect to server
        await tester.connect()
        
        # Process a query that should trigger tool usage
        query = "Search for Python repositories on GitHub"
        response, tools_called = await tester.process_query(query)
        
        # Verify tools were called
        assert len(tools_called) > 0, "No tools were called"
        print(f"\n✓ Tools called: {[tc.name for tc in tools_called]}")
        
        # Create test case
        test_case = await tester.create_test_case(query, response)
        
        # Verify test case has tool calls
        assert len(test_case.mcp_tools_called) > 0
        assert test_case.mcp_tools_called[0].name in [tc.name for tc in tools_called]
        
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])


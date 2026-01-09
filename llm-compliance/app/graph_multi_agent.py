"""
Multi-Agent System using LangGraph
Each agent is autonomous with its own LLM instance and can communicate with other agents.
"""
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from .state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, OpenWeatherMapAPIWrapper
from dotenv import load_dotenv
from typing import Dict, Any, List
import os
import json
import re

# Load environment variables
load_dotenv()

# Initialize different LLM instances for different agents
def get_planner_llm():
    """LLM for Planner Agent"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))  # Slightly higher for planning
    return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)

def get_researcher_llm():
    """LLM for Researcher Agent"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)

def get_weather_llm():
    """LLM for Weather Agent"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)

def get_coordinator_llm():
    """LLM for Coordinator Agent"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)

# Initialize LLM instances for each agent
planner_llm = get_planner_llm()
researcher_llm = get_researcher_llm()
weather_llm = get_weather_llm()
coordinator_llm = get_coordinator_llm()

# Initialize Wikipedia API wrapper
wikipedia_api_wrapper = WikipediaAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=4000
)

# Initialize Weather API wrapper
try:
    weather_wrapper = OpenWeatherMapAPIWrapper()
except Exception as e:
    print(f"Warning: OpenWeatherMapAPIWrapper initialization failed: {e}")
    print("Weather API will use simulated data instead.")
    weather_wrapper = None

# Tools available to agents
@tool
def wikipedia_query(query: str) -> str:
    """Searches Wikipedia for information about a given topic."""
    try:
        wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)
        result = wikipedia_tool.invoke(query)
        return result if result else f"No Wikipedia results found for query: '{query}'"
    except Exception as e:
        return f"Error retrieving Wikipedia data for '{query}': {str(e)}"

@tool
def get_current_weather(location: str) -> str:
    """Fetches the current weather data for a given city."""
    if weather_wrapper is None:
        return f"Weather API is not configured. Using simulated data for {location}."
    try:
        return weather_wrapper.run(location)
    except Exception as e:
        return f"Error fetching weather for {location}: {e}"

# Helper function to extract location from query
def extract_location_from_query(query: str) -> str:
    """Extract location from a weather-related query using regex patterns."""
    patterns = [
        r'(?:weather|temperature|forecast|temp).*?\b(in|for|at)\s+([A-Z][a-zA-Z\s,]+?)(?:\?|$|\.|,|\s+$)',
        r'\b(in|for|at)\s+([A-Z][a-zA-Z\s,]+?)(?:\s+weather|\s+temperature|\s+forecast|\?|$|\.|,)',
        r'([A-Z][a-zA-Z\s,]+?)(?:\s+weather|\s+temperature|\s+forecast)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            location = match.group(2).strip() if len(match.groups()) >= 2 else match.group(1).strip()
            location = re.sub(r'\b(what|what\'s|the|is|are|how|tell|me|about|show|s|re|ll|ve|d)\b', '', location, flags=re.IGNORECASE).strip()
            location = re.sub(r'[?.,!;:]+$', '', location).strip()
            location = re.sub(r'\s+', ' ', location).strip()
            if location and len(location) > 1:
                return location
    
    capitalized_words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
    if capitalized_words:
        common_words = {'What', 'The', 'Weather', 'Temperature', 'Forecast', 'In', 'For', 'At', 'Is', 'Are'}
        locations = [w for w in capitalized_words if w not in common_words]
        if locations:
            return ' '.join(locations)
    
    return ""

# Helper function to extract topic from query
def extract_topic_from_query(query: str) -> str:
    """Extract the main topic from a question/query for Wikipedia search."""
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that extracts the main topic or subject from a question or query. "
                       "Return only the topic/subject name that would be best for searching Wikipedia. "
                       "Examples: 'What is the capital of France?' -> 'France', "
                       "'Tell me about Python programming' -> 'Python programming', "
                       "'Who discovered America?' -> 'Discovery of America' or 'America'"),
            ("human", "Query: {query}\nExtract the main topic for Wikipedia search (return only the topic, no explanation):")
        ])
        chain = prompt | researcher_llm
        response = chain.invoke({"query": query})
        topic = response.content.strip()
        topic = re.sub(r'^["\']|["\']$', '', topic).strip()
        if topic and len(topic) > 1:
            return topic
    except Exception as e:
        print(f"⚠️ LLM topic extraction failed: {e}, using regex fallback")
    
    # Regex fallback
    question_starters = [
        r'^(what|who|where|when|why|how|tell me about|explain|describe|what\'s|what is|what are|who is|who are|where is|where are)\s+',
        r'^(can you|could you|please|do you know)\s+',
    ]
    
    cleaned_query = query
    for pattern in question_starters:
        cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
    
    cleaned_query = re.sub(r'[?.,!;:]+$', '', cleaned_query).strip()
    capitalized_words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', cleaned_query)
    
    if capitalized_words:
        common_words = {'What', 'The', 'Who', 'Where', 'When', 'Why', 'How', 'Tell', 'Me', 'About', 'Explain', 'Describe', 'Is', 'Are', 'Can', 'Could', 'Please', 'Do', 'You', 'Know'}
        topics = [w for w in capitalized_words if w not in common_words]
        if topics:
            return topics[-1]
    
    return cleaned_query if cleaned_query else query


# ============================================================================
# MULTI-AGENT SYSTEM: Each agent is autonomous with its own LLM
# ============================================================================

def planner_agent(state: AgentState) -> AgentState:
    """
    Planner Agent: Autonomous agent responsible for analyzing queries and deciding the workflow.
    This agent has its own LLM instance and decision-making capability.
    """
    query = state.get("query", "")
    trajectory = state.get("trajectory", [])
    
    # Initialize trajectory with user query
    if not trajectory:
        trajectory.append({
            "role": "user",
            "content": query
        })
    
    if not query.strip():
        trajectory.append({
            "role": "assistant",
            "content": "I need to ask for clarification.",
            "agent": "planner"
        })
        return {"action": "ask_clarify", "trajectory": trajectory}
    
    # Use LLM to generate comprehensive plan
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Planner Agent. Your role is to analyze user queries and create a comprehensive plan. "
                   "You can delegate tasks to other agents: Researcher Agent (for Wikipedia queries) or Weather Agent (for weather queries). "
                   "\n"
                   "IMPORTANT: If a query asks for BOTH weather AND history/information/about (e.g., 'weather in Paris and tell me about its history'), "
                   "you MUST set action='retrieve_weather' AND needs_research=true. "
                   "The Weather Agent will handle both weather and research for mixed queries. "
                   "\n"
                   "For each query, generate a detailed step-by-step plan that includes: "
                   "1. Analysis of the user's query to identify information needs "
                   "2. Delegation to appropriate agents (Weather Agent, Researcher Agent, or both) "
                   "3. Tool calls that need to be executed (get_current_weather, wikipedia_query, or both) "
                   "4. Synthesis of retrieved information "
                   "5. Providing a comprehensive explanation to the user "
                   "\n"
                   "Respond with a JSON object containing: "
                   "- 'action': one of 'retrieve_weather' (for weather queries, including mixed queries), 'retrieve' (for research-only), or 'ask_clarify' (only if truly unclear) "
                   "- 'plan': a detailed step-by-step plan as a single string "
                   "- 'location': extracted location if weather-related (null otherwise) "
                   "- 'tool_calls': array of tool call objects with 'name' and 'arguments' (JSON string) "
                   "- 'needs_research': boolean - set to true if query asks for history, information, or background about a location/topic"),
        ("human", "User Query: {query}\n\nGenerate a comprehensive plan. Respond ONLY with valid JSON:")
    ])
    
    chain = prompt | planner_llm
    response = chain.invoke({"query": query})
    
    # First, check query content directly for mixed queries (more reliable than LLM response)
    query_lower = query.lower()
    has_weather = any(kw in query_lower for kw in ["weather", "temperature", "forecast", "climate"])
    has_research = any(kw in query_lower for kw in ["history", "tell me about", "information about", "about its", "about the", "and tell me"])
    
    try:
        # Try to parse JSON response - handle code blocks if present
        response_text = response.content.strip() if hasattr(response, 'content') else str(response)
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            # Extract JSON from code block
            lines = response_text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_json = not in_json
                    continue
                if in_json:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)
        
        plan_data = json.loads(response_text)
        action = plan_data.get("action", "ask_clarify")
        plan = plan_data.get("plan", "No plan generated.")
        location = plan_data.get("location")
        tool_calls_data = plan_data.get("tool_calls", [])
        needs_research = plan_data.get("needs_research", False)
        
        # Override with query-based detection if LLM missed mixed query
        if has_weather and has_research:
            needs_research = True
            if action == "ask_clarify":
                action = "retrieve_weather"
    except (json.JSONDecodeError, AttributeError, KeyError):
        # Fallback: detect from query content directly (more reliable)
        action = "ask_clarify"
        plan = f"Step 1: Analyze the user's query '{query}'. Step 2: Determine appropriate action. Step 3: Execute necessary tool calls. Step 4: Synthesize information. Step 5: Provide explanation to user."
        location = extract_location_from_query(query) if has_weather else None
        tool_calls_data = []
        needs_research = has_research
        
        # Infer action from query content
        if has_weather and has_research:
            # Mixed query - weather + research
            action = "retrieve_weather"
            needs_research = True
            if not location:
                location = extract_location_from_query(query) or query.split()[0] if query.split() else query
            tool_calls_data = [
                {"name": "get_current_weather", "arguments": json.dumps({"location": location})},
                {"name": "wikipedia_query", "arguments": json.dumps({"query": location})}
            ]
        elif has_weather:
            # Weather-only query
            action = "retrieve_weather"
            if not location:
                location = extract_location_from_query(query) or query.split()[0] if query.split() else query
            tool_calls_data = [{"name": "get_current_weather", "arguments": json.dumps({"location": location})}]
        elif has_research:
            # Research-only query
            action = "retrieve"
            tool_calls_data = [{"name": "wikipedia_query", "arguments": json.dumps({"query": query})}]
    
    # Convert tool_calls_data to proper format
    tool_calls = []
    for tc in tool_calls_data:
        if isinstance(tc, dict):
            tool_calls.append({
                "function": {
                    "name": tc.get("name", "unknown"),
                    "arguments": tc.get("arguments", "{}")
                }
            })
    
    # Append plan to trajectory
    trajectory.append({
        "role": "assistant",
        "content": plan,
        "agent": "planner",
        "tool_calls": tool_calls if tool_calls else []
    })
    
    # Prepare return state
    return_state = {"action": action, "trajectory": trajectory}
    if location:
        return_state["location"] = location
    if needs_research:
        return_state["needs_research"] = True
    
    return return_state


def researcher_agent(state: AgentState) -> AgentState:
    """
    Researcher Agent: Autonomous agent specialized in information retrieval from Wikipedia.
    This agent has its own LLM instance and can make independent decisions about research.
    """
    trajectory = state.get("trajectory", [])
    
    if state.get("action") != "retrieve":
        return {"docs": [], "trajectory": trajectory}
    
    query = state.get("query", "")
    
    # Researcher agent uses its own LLM to refine the search query
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Researcher Agent specialized in finding accurate information. "
                   "Your task is to extract the best search query for Wikipedia based on the user's question."),
        ("human", "User Question: {query}\n\nWhat should I search for in Wikipedia? (Return only the search query):")
    ])
    
    chain = prompt | researcher_llm
    response = chain.invoke({"query": query})
    search_query = response.content.strip()
    
    # Fallback to topic extraction if LLM doesn't provide good result
    if not search_query or len(search_query) < 2:
        search_query = extract_topic_from_query(query)
    
    print(f"🔬 Researcher Agent: Searching Wikipedia for '{search_query}'")
    
    # Researcher agent uses Wikipedia tool
    wikipedia_result = wikipedia_query.invoke(search_query)
    
    if wikipedia_result:
        docs = [wikipedia_result] if isinstance(wikipedia_result, str) else wikipedia_result
    else:
        docs = [f"No Wikipedia results found for query: '{query}'"]
    
    # Researcher agent reports its findings
    trajectory.append({
        "role": "tool",
        "content": "\n\n".join([str(doc)[:1000] for doc in docs[:3]]),
        "agent": "researcher"
    })
    
    return {"docs": docs, "trajectory": trajectory}


def weather_agent(state: AgentState) -> AgentState:
    """
    Weather Agent: Autonomous agent specialized in weather information.
    This agent has its own LLM instance and can handle weather-related queries independently.
    For mixed queries, it also triggers research.
    """
    trajectory = state.get("trajectory", [])
    
    if state.get("action") != "retrieve_weather":
        return {"docs": [], "trajectory": trajectory}
    
    location = state.get("location", "")
    query = state.get("query", "")
    needs_research = state.get("needs_research", False)
    
    if not location:
        location = extract_location_from_query(query)
    
    # Weather agent uses its own LLM to refine location if needed
    if not location or len(location) < 2:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Weather Agent. Extract the location from the user's query."),
            ("human", "Query: {query}\n\nWhat location should I get weather for? (Return only the location name):")
        ])
        chain = prompt | weather_llm
        response = chain.invoke({"query": query})
        location = response.content.strip()
    
    print(f"🌤️ Weather Agent: Getting weather for '{location}'")
    
    # Weather agent uses weather tool
    weather_result = get_current_weather.invoke(location)
    
    if weather_result:
        docs = [weather_result] if isinstance(weather_result, str) else [str(weather_result)]
    else:
        docs = [f"No weather results found for location: '{location}'"]
    
    # Weather agent reports its findings
    trajectory.append({
        "role": "tool",
        "content": "\n\n".join([str(doc) for doc in docs]),
        "agent": "weather"
    })
    
    # For mixed queries, also trigger research
    if needs_research:
        print(f"🔬 Weather Agent: Also triggering research for '{location}'")
        # Also perform research for the location
        research_query = location  # Use location as research topic
        wikipedia_result = wikipedia_query.invoke(research_query)
        
        if wikipedia_result:
            research_docs = [wikipedia_result] if isinstance(wikipedia_result, str) else wikipedia_result
        else:
            research_docs = [f"No Wikipedia results found for query: '{research_query}'"]
        
        # Add research findings to docs
        docs.extend(research_docs)
        
        # Add research to trajectory
        trajectory.append({
            "role": "tool",
            "content": "\n\n".join([str(doc)[:1000] for doc in research_docs[:3]]),
            "agent": "weather"  # Weather agent triggered research for mixed query
        })
    
    return {"docs": docs, "trajectory": trajectory}


def coordinator_agent(state: AgentState) -> AgentState:
    """
    Coordinator Agent: Autonomous agent that synthesizes information from other agents
    and generates the final response. This agent coordinates the work of other agents.
    """
    query = state.get("query", "")
    docs = state.get("docs", [])
    trajectory = state.get("trajectory", [])
    
    if not docs:
        # Coordinator agent asks for clarification
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Coordinator Agent. Your role is to synthesize information from other agents "
                       "and provide helpful responses to users. If you don't have enough information, ask for clarification."),
            ("human", "User Query: {query}\n\nGenerate a polite request for clarification:")
        ])
        chain = prompt | coordinator_llm
        response = chain.invoke({"query": query})
        answer = response.content
        
        trajectory.append({
            "role": "assistant",
            "content": answer,
            "agent": "coordinator"
        })
        
        return {"answer": answer, "trajectory": trajectory}
    
    # Coordinator agent synthesizes information from other agents
    context = "\n\n".join(docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Coordinator Agent. Your role is to synthesize information gathered by other agents "
                   "(Researcher Agent, Weather Agent) and provide a comprehensive answer to the user. "
                   "Use the information provided by other agents to answer the question accurately."),
        ("human", "Information gathered by other agents:\n{context}\n\nUser Question: {query}\n\n"
                  "Provide a comprehensive answer based on the information from other agents:")
    ])
    
    chain = prompt | coordinator_llm
    response = chain.invoke({"query": query, "context": context})
    answer = response.content
    
    # Coordinator agent provides final answer
    trajectory.append({
        "role": "assistant",
        "content": answer,
        "agent": "coordinator"
    })
    
    return {"answer": answer, "trajectory": trajectory}


def build_multi_agent_graph(checkpointer=None):
    """
    Build a multi-agent graph where each agent is autonomous and can communicate.
    
    Args:
        checkpointer: Optional checkpointer for thread-based trajectory extraction.
                     If provided, the graph will be compiled with checkpointing enabled.
    
    Returns:
        Compiled graph (with or without checkpointer)
    """
    g = StateGraph(AgentState)
    
    # Add agent nodes
    g.add_node("Planner Agent", planner_agent)
    g.add_node("Researcher Agent", researcher_agent)
    g.add_node("Weather Agent", weather_agent)
    g.add_node("Coordinator Agent", coordinator_agent)
    
    # Set entry point
    g.set_entry_point("Planner Agent")
    
    def route(state: AgentState):
        """Route based on Planner Agent's decision"""
        action = state.get("action", "")
        if action == "retrieve":
            return "Researcher Agent"
        elif action == "retrieve_weather":
            return "Weather Agent"
        else:
            return "Coordinator Agent"
    
    # Conditional routing from Planner Agent
    g.add_conditional_edges("Planner Agent", route, {
        "Researcher Agent": "Researcher Agent",
        "Weather Agent": "Weather Agent",
        "Coordinator Agent": "Coordinator Agent"
    })
    
    # Both Researcher and Weather agents report to Coordinator
    g.add_edge("Researcher Agent", "Coordinator Agent")
    g.add_edge("Weather Agent", "Coordinator Agent")
    
    # Coordinator provides final answer
    g.add_edge("Coordinator Agent", END)
    
    # Compile with or without checkpointer
    if checkpointer:
        checkpointer = MemorySaver()
        return g.compile(checkpointer=checkpointer)
    else:
        return g.compile()


# ==================== Conversation and Tool Call Extraction ====================

def create_multi_agent():
    """
    Create and compile the multi-agent graph with checkpointing enabled.
    
    Returns:
        Compiled graph with memory checkpointer
    """
    memory = MemorySaver()
    return build_multi_agent_graph(checkpointer=memory)


def extract_conversation_messages(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all conversation messages from the multi-agent result trajectory.
    
    Args:
        result: The result dictionary from agent.invoke() containing trajectory
    
    Returns:
        List of message dictionaries with role, content, agent, and tool_calls if applicable
    """
    trajectory = result.get("trajectory", [])
    conversation = []
    
    for msg in trajectory:
        msg_dict = {
            "role": msg.get("role"),
            "content": msg.get("content"),
            "agent": msg.get("agent"),  # Which agent generated this message
            "tool_calls": msg.get("tool_calls", [])
        }
        conversation.append(msg_dict)
    
    return conversation


def extract_tool_calls(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all tool calls from the multi-agent result trajectory.
    
    Args:
        result: The result dictionary from agent.invoke() containing trajectory
    
    Returns:
        List of tool call dictionaries with name, arguments, and agent
    """
    trajectory = result.get("trajectory", [])
    tool_calls = []
    
    for msg in trajectory:
        if msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                tool_call_dict = {
                    "name": None,
                    "args": {},
                    "agent": msg.get("agent"),  # Which agent made this tool call
                    "message_content": msg.get("content")
                }
                
                # Extract tool call info from different formats
                if isinstance(tool_call, dict):
                    if "function" in tool_call:
                        func_info = tool_call["function"]
                        tool_call_dict["name"] = func_info.get("name") if isinstance(func_info, dict) else getattr(func_info, "name", None)
                        args_str = func_info.get("arguments", "{}") if isinstance(func_info, dict) else getattr(func_info, "arguments", "{}")
                        if isinstance(args_str, str):
                            try:
                                tool_call_dict["args"] = json.loads(args_str)
                            except:
                                tool_call_dict["args"] = {}
                        else:
                            tool_call_dict["args"] = args_str
                    else:
                        tool_call_dict["name"] = tool_call.get("name")
                        tool_call_dict["args"] = tool_call.get("args", tool_call.get("arguments", {}))
                
                tool_calls.append(tool_call_dict)
    
    return tool_calls


def get_conversation_trajectory(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the complete conversation trajectory including messages and tool calls.
    
    Args:
        result: The result dictionary from agent.invoke()
    
    Returns:
        Dictionary with messages, tool_calls, and summary statistics
    """
    messages = extract_conversation_messages(result)
    tool_calls = extract_tool_calls(result)
    
    return {
        "messages": messages,
        "tool_calls": tool_calls,
        "total_messages": len(messages),
        "total_tool_calls": len(tool_calls),
        "answer": result.get("answer", ""),
        "query": result.get("query", ""),
        "action": result.get("action", "")
    }


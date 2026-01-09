from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict, total=False):
    query: str
    action: str
    location: str  # For weather queries
    docs: List[str]
    answer: str
    meta: Dict[str, Any]
    trajectory: List[Dict[str, Any]]  # Trajectory for agentevals evaluation

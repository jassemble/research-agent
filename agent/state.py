from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    question: str
    search_results: List[dict]   # accumulated results across iterations
    iterations: int
    max_iterations: int
    enough_info: bool
    report: str
    messages: Annotated[list, add_messages]

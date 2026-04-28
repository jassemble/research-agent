from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import search_node, evaluate_node, write_report_node


def route_after_evaluate(state: AgentState) -> str:
    """Conditional edge: keep searching or write the report."""
    if state["enough_info"]:
        return "write_report"
    return "search"


def build_graph():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("search", search_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("write_report", write_report_node)

    # Wire edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "search": "search",
            "write_report": "write_report",
        },
    )
    workflow.add_edge("write_report", END)

    return workflow.compile()

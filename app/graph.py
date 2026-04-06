"""LangGraph state machine for the analytics agent.

Graph topology:
  schema_extractor → query_guardrail → [conditional]
    ├─ approved → text_to_sql → execute_sql → [conditional]
    │   ├─ success → generate_insights → chart_selector → generate_chart → END
    │   └─ error (retry < 3) → text_to_sql (loop)
    │   └─ error (retry >= 3) → END (with error)
    └─ rejected → END (with error)
"""

from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes.schema_extractor import schema_extractor
from .nodes.guardrail import query_guardrail
from .nodes.text_to_sql import text_to_sql
from .nodes.execute_sql import execute_sql
from .nodes.generate_insights import generate_insights
from .nodes.chart_selector import chart_selector
from .nodes.generate_chart import generate_chart

MAX_RETRIES = 3


def should_retry_or_continue(state: AgentState) -> str:
    """Conditional edge: check if SQL execution succeeded or needs retry."""
    # If sql_executed is True, execution succeeded (even if 0 rows returned)
    if state.get("sql_executed"):
        return "success"

    # Check retry count
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return "max_retries"

    return "retry"


def check_guardrail(state: AgentState) -> str:
    """Conditional edge: check if the prompt was approved or rejected."""
    if state.get("status") == "rejected":
        return "rejected"
    return "approved"


def mark_success(state: AgentState) -> dict:
    """Terminal node marking the run as successful."""
    return {"status": "success"}


def mark_error(state: AgentState) -> dict:
    """Terminal node marking the run as failed after max retries."""
    return {"status": "error"}


def build_graph() -> StateGraph:
    """Build and compile the analytics agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("schema_extractor", schema_extractor)
    graph.add_node("query_guardrail", query_guardrail)
    graph.add_node("text_to_sql", text_to_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("generate_insights", generate_insights)
    graph.add_node("chart_selector", chart_selector)
    graph.add_node("generate_chart", generate_chart)
    graph.add_node("mark_success", mark_success)
    graph.add_node("mark_error", mark_error)

    # Set entry point
    graph.set_entry_point("schema_extractor")

    # Linear edges
    graph.add_edge("schema_extractor", "query_guardrail")
    
    # Conditional edge: Guardrail checking
    graph.add_conditional_edges(
        "query_guardrail",
        check_guardrail,
        {
            "approved": "text_to_sql",
            "rejected": END,
        },
    )

    # Sequence
    graph.add_edge("text_to_sql", "execute_sql")

    # Conditional edge: self-correction loop
    graph.add_conditional_edges(
        "execute_sql",
        should_retry_or_continue,
        {
            "success": "generate_insights",
            "retry": "text_to_sql",
            "max_retries": "mark_error",
        },
    )

    # After insights, select chart type, then generate chart
    graph.add_edge("generate_insights", "chart_selector")
    graph.add_edge("chart_selector", "generate_chart")

    # Terminal edges
    graph.add_edge("generate_chart", "mark_success")
    graph.add_edge("mark_success", END)
    graph.add_edge("mark_error", END)

    return graph.compile()


# Singleton compiled graph
analytics_graph = build_graph()

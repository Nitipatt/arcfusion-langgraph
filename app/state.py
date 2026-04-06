from typing import TypedDict, Any


class AgentState(TypedDict):
    """State definition for the LangGraph analytics agent."""
    session_id: str
    user_query: str
    history: list[dict[str, str]]
    db_schema: str
    generated_sql: str
    sql_errors: list[str]
    raw_data: list[dict[str, Any]]
    insight_title: str
    narrative_summary: str
    recommended_actions: list[str]
    echarts_config: dict[str, Any]
    selected_chart_type: str  # Chart type chosen by chart_selector node
    retry_count: int
    status: str  # "running", "success", "error"
    db_url: str  # Dynamic database URL from user's connection
    sql_executed: bool

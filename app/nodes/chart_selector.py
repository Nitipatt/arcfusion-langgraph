"""Chart selector node — LLM picks the best chart type based on data shape and intent."""

import json
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from ..llm import get_llm
from ..state import AgentState
from ..prompts import CHART_SELECTOR_PROMPT
import logging

logger = logging.getLogger(__name__)

VALID_CHART_TYPES = {"bar", "line", "area", "pie", "donut", "bubble", "stacked_bar", "none"}


class ChartSelectorOutput(BaseModel):
    """Structured output for chart type selection."""
    chart_type: str = Field(
        ...,
        description="The chosen chart type: 'bar', 'line', 'area', 'pie', 'donut', 'bubble', 'stacked_bar', or 'none'. Select 'none' if the data is a single scalar value, metadata, or does not benefit from visualization.",
    )
    reason: str = Field(
        ..., description="Brief explanation of why this chart type was selected"
    )


def _analyze_data_shape(raw_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract data shape metadata to help the LLM decide on chart type."""
    if not raw_data:
        return {
            "row_count": 0,
            "columns": "[]",
            "unique_first_col": 0,
            "numeric_columns": "[]",
        }

    keys = list(raw_data[0].keys())
    numeric_cols = [
        k for k in keys
        if any(isinstance(row.get(k), (int, float)) for row in raw_data[:10])
    ]
    first_col_values = {str(row.get(keys[0], "")) for row in raw_data} if keys else set()

    return {
        "row_count": len(raw_data),
        "columns": json.dumps(keys),
        "unique_first_col": len(first_col_values),
        "numeric_columns": json.dumps(numeric_cols),
    }


def _heuristic_fallback(raw_data: list[dict[str, Any]], user_query: str) -> str:
    """Rule-based fallback when LLM fails to pick a chart type."""
    query_lower = user_query.lower()

    # Check for composition keywords
    if any(kw in query_lower for kw in ["breakdown", "distribution", "share", "proportion", "composition"]):
        if raw_data and len(raw_data) <= 7:
            return "donut"
        return "pie"

    # Check for time-series keywords
    if any(kw in query_lower for kw in ["trend", "over time", "monthly", "daily", "weekly", "yearly", "timeline"]):
        return "line"

    # Check for comparison keywords
    if any(kw in query_lower for kw in ["compare", "vs", "versus", "comparison"]):
        if raw_data:
            keys = list(raw_data[0].keys())
            numeric_cols = [k for k in keys if any(isinstance(row.get(k), (int, float)) for row in raw_data[:5])]
            if len(numeric_cols) >= 3:
                return "bubble"
        return "stacked_bar"

    # Check data shape
    if raw_data:
        keys = list(raw_data[0].keys())
        # If there's a date-like column name
        if any(kw in keys[0].lower() for kw in ["date", "month", "year", "day", "week", "time", "period"]):
            return "line"

    return "bar"


def chart_selector(state: AgentState) -> dict[str, Any]:
    """Select the best chart type for the data visualization."""
    logger.info("Executing node: chart_selector")
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(ChartSelectorOutput)

    raw_data = state.get("raw_data", [])
    if not raw_data:
        logger.info("No data available, defaulting to bar chart")
        return {"selected_chart_type": "bar"}

    data_preview = raw_data[:50] if len(raw_data) > 50 else raw_data
    shape_info = _analyze_data_shape(raw_data)

    prompt = CHART_SELECTOR_PROMPT.format(
        user_query=state["user_query"],
        raw_data=json.dumps(data_preview, indent=2, default=str),
        **shape_info,
    )

    try:
        result: ChartSelectorOutput = structured_llm.invoke([HumanMessage(content=prompt)])
        chart_type = result.chart_type.lower().strip()

        # Validate the selected type
        if chart_type not in VALID_CHART_TYPES:
            logger.warning(f"LLM selected invalid chart type '{chart_type}', falling back to heuristic")
            chart_type = _heuristic_fallback(raw_data, state["user_query"])

        logger.info(f"Selected chart type: {chart_type} (reason: {result.reason})")
        return {"selected_chart_type": chart_type}
    except Exception as e:
        logger.error(f"Chart selector LLM failed: {str(e)}", exc_info=True)
        fallback_type = _heuristic_fallback(raw_data, state["user_query"])
        logger.info(f"Using heuristic fallback chart type: {fallback_type}")
        return {"selected_chart_type": fallback_type}

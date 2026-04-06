"""Generate insights node — uses LLM structured output for analysis."""

import json
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from ..config import settings
from ..llm import get_llm
from ..state import AgentState
from ..prompts import INSIGHTS_PROMPT
import logging

logger = logging.getLogger(__name__)


class InsightOutput(BaseModel):
    """Structured output from the insight generation LLM."""
    insight_title: str = Field(..., description="A compelling, concise title for the insight")
    narrative_summary: str = Field(
        ..., description="A 2-3 sentence paragraph summarizing the key findings"
    )
    recommended_actions: list[str] = Field(
        ..., description="Exactly 3 actionable recommendations",
        min_length=3,
        max_length=3,
    )





def generate_insights(state: AgentState) -> dict:
    """Analyze raw data and produce structured business insights."""
    logger.info("Executing node: generate_insights")
    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(InsightOutput)

    # Limit data size for context window
    raw_data = state.get("raw_data", [])
    if len(raw_data) > 100:
        data_preview = raw_data[:100]
        data_note = f"\n(Showing first 100 of {len(raw_data)} rows)"
    else:
        data_preview = raw_data
        data_note = ""

    # Build history context
    history_context = "No previous context."
    history = state.get("history", [])
    if history:
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_context = "\n".join(lines)

    prompt = INSIGHTS_PROMPT.format(
        history_context=history_context,
        user_query=state["user_query"],
        generated_sql=state.get("generated_sql", ""),
        raw_data=json.dumps(data_preview, indent=2, default=str) + data_note,
    )

    try:
        result: InsightOutput = structured_llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"Successfully generated insights: '{result.insight_title}'")
        return {
            "insight_title": result.insight_title,
            "narrative_summary": result.narrative_summary,
            "recommended_actions": result.recommended_actions,
        }
    except Exception as e:
        logger.error(f"Failed to generate structured insights: {str(e)}", exc_info=True)
        # Fallback if structured output fails
        return {
            "insight_title": "Data Analysis Results",
            "narrative_summary": f"Analysis of {len(raw_data)} records for: {state['user_query']}",
            "recommended_actions": [
                "Review the data for patterns",
                "Compare with historical benchmarks",
                "Share findings with stakeholders",
            ],
        }

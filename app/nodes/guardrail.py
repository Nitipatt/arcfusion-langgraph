"""Guardrail node — prevents off-topic LLM usage."""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from ..llm import get_llm
from ..state import AgentState
from ..prompts import GUARDRAIL_PROMPT
import logging

logger = logging.getLogger(__name__)


class GuardrailOutput(BaseModel):
    """Structured output from the guardrail check."""
    is_related: bool = Field(
        ..., description="True if related to analytics or schema, False if completely off-topic"
    )
    reason: str = Field(
        ..., description="Brief 1-sentence explanation of why it is related or not"
    )


def query_guardrail(state: AgentState) -> dict:
    """Check if the user query is relevant to the database or analytics."""
    logger.info(f"Executing node: query_guardrail for query '{state.get('user_query')}'")
    # Fast model with very low temperature
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(GuardrailOutput)

    # Note: DB schema might be extremely long, but the LLM just needs a glimpse
    # We can truncate the schema if it's too large, but usually it's manageable.
    schema = state.get("db_schema", "No schema available.")
    if len(schema) > 2000:
        schema = schema[:2000] + "\n...(truncated)"

    prompt = GUARDRAIL_PROMPT.format(
        user_query=state.get("user_query", ""),
        db_schema=schema,
    )

    try:
        result: GuardrailOutput = structured_llm.invoke([HumanMessage(content=prompt)])
        
        if not result.is_related:
            logger.warning(f"Guardrail rejected query: {result.reason}")
            # Gracefully handle off-topic queries by treating them as conversational responses rather than systemic errors
            return {
                "status": "rejected",
                "insight_title": "Off-topic Request",
                "narrative_summary": f"I am a data analytics assistant focused on executing robust queries against your database schema. Unfortunately, I cannot help with that specific request.\n\n**Reason:** {result.reason}\n\nPlease ask me a question related to generating insights from your tables!",
            }
        
        logger.info(f"Guardrail approved query. Reason: {result.reason}")
        return {"status": "approved"}
    
    except Exception as e:
        logger.error(f"Guardrail check failed: {str(e)}", exc_info=True)
        # If the LLM call fails for some reason (e.g., structured output parsing fails),
        # Default to fail-open (allow the query through) to not block legitimate traffic because of a brittle guardrail.
        return {"status": "approved"}

"""Text-to-SQL node — LLM translates natural language to read-only SQL."""

import re
from langchain_core.messages import HumanMessage
from ..llm import get_llm
from ..state import AgentState
from ..prompts import TEXT_TO_SQL_PROMPT, TEXT_TO_SQL_RETRY_CONTEXT
import logging

logger = logging.getLogger(__name__)





def _validate_sql(sql: str) -> str | None:
    """Validate that SQL is read-only. Returns error message or None."""
    dangerous = re.findall(
        r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b',
        sql,
        re.IGNORECASE,
    )
    if dangerous:
        return f"Dangerous SQL operation detected: {', '.join(dangerous)}. Only SELECT is allowed."
    return None


def text_to_sql(state: AgentState) -> dict:
    """Generate a read-only SQL query from the user's natural language question."""
    logger.info("Executing node: text_to_sql")
    llm = get_llm(temperature=0.0)

    # Build error context for retries
    error_context = ""
    if state.get("sql_errors"):
        last_error = state["sql_errors"][-1]
        error_context = TEXT_TO_SQL_RETRY_CONTEXT.format(
            error=last_error,
            previous_sql=state.get("generated_sql", ""),
        )

    # Build history context
    history_context = "No previous context."
    history = state.get("history", [])
    if history:
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_context = "\n".join(lines)

    prompt = TEXT_TO_SQL_PROMPT.format(
        db_schema=state["db_schema"],
        history_context=history_context,
        user_query=state["user_query"],
        error_context=error_context,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    sql = response.content.strip()

    # Clean markdown fencing if present
    sql = re.sub(r'^```(?:sql)?\s*', '', sql)
    sql = re.sub(r'\s*```$', '', sql)
    sql = sql.strip()

    # Validate read-only
    validation_error = _validate_sql(sql)
    if validation_error:
        logger.warning(f"SQL validation failed: {validation_error}")
        errors = state.get("sql_errors", [])
        errors.append(validation_error)
        return {
            "generated_sql": sql,
            "sql_errors": errors,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    logger.info(f"Generated valid SQL query:\n{sql}")
    return {"generated_sql": sql}

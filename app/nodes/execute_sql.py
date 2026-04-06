"""Execute SQL node — runs the generated query against the user's database."""

from datetime import date, datetime
from decimal import Decimal
from sqlalchemy import create_engine, text
import logging
from ..state import AgentState

logger = logging.getLogger(__name__)


def _serialize_value(val):
    """Convert non-JSON-serializable values."""
    if isinstance(val, (date, datetime)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    return val


def execute_sql(state: AgentState) -> dict:
    """Execute the generated SQL query and return results as list of dicts."""
    logger.info("Executing node: execute_sql")
    sql = state.get("generated_sql", "")
    if not sql:
        logger.warning("No SQL query generated to execute.")
        return {
            "sql_errors": state.get("sql_errors", []) + ["No SQL query generated"],
            "retry_count": state.get("retry_count", 0) + 1,
            "sql_executed": False,
        }

    db_url = state.get("db_url", "")
    if not db_url:
        return {
            "sql_errors": state.get("sql_errors", []) + ["No database connection configured"],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    engine = create_engine(db_url, pool_pre_ping=True)

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchall()

            raw_data = [
                {col: _serialize_value(val) for col, val in zip(columns, row)}
                for row in rows
            ]

            logger.info(f"SQL execution successful. Retrieved {len(raw_data)} rows.")
            return {"raw_data": raw_data, "sql_executed": True}

    except Exception as e:
        error_msg = f"SQL execution error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors = state.get("sql_errors", [])
        errors.append(error_msg)
        return {
            "sql_errors": errors,
            "retry_count": state.get("retry_count", 0) + 1,
            "sql_executed": False,
        }
    finally:
        engine.dispose()

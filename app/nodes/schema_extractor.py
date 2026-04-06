"""Schema extractor node — dumps DDL and sample rows from the user's database."""

from sqlalchemy import create_engine, text
import logging

from ..state import AgentState
from ..config import settings
from ..schema_cache import SchemaCache

logger = logging.getLogger(__name__)

# Module-level singleton
_schema_cache = SchemaCache(
    db_url=settings.database_url,
    ttl_seconds=settings.SCHEMA_CACHE_TTL_SECONDS,
)


def get_schema_cache() -> SchemaCache:
    """Return the schema cache singleton (for stats/invalidation)."""
    return _schema_cache


def schema_extractor(state: AgentState) -> dict:
    logger.info("Executing node: schema_extractor")
    db_url = state.get("db_url", "")
    if not db_url:
        logger.warning("No database URL provided in state.")
        return {"db_schema": "-- No database connection configured"}

    # Extract schema from URL search_path parameter (default to 'public')
    import re
    schema_match = re.search(r'search_path%3D(\w+)', db_url)
    target_schema = schema_match.group(1) if schema_match else 'public'

    # Check schema cache first
    if settings.CACHE_ENABLED:
        cached_schema = _schema_cache.get(db_url)
        if cached_schema is not None:
            return {"db_schema": cached_schema}

    engine = create_engine(db_url, pool_pre_ping=True)
    schema_parts: list[str] = []

    try:
        with engine.connect() as conn:
            tables = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :schema AND table_type = 'BASE TABLE' "
                    "AND table_name NOT IN ('query_cache', 'schema_cache', "
                    "'databasechangelog', 'databasechangeloglock')"
                ),
                {"schema": target_schema},
            ).fetchall()

            for (table_name,) in tables:
                columns = conn.execute(
                    text(
                        "SELECT column_name, data_type, is_nullable "
                        "FROM information_schema.columns "
                        "WHERE table_schema = :schema AND table_name = :table "
                        "ORDER BY ordinal_position"
                    ),
                    {"schema": target_schema, "table": table_name},
                ).fetchall()

                schema_parts.append(f"-- Table: {table_name}")
                col_defs = []
                col_names = []
                for col_name, data_type, nullable in columns:
                    null_str = "" if nullable == "YES" else " NOT NULL"
                    col_defs.append(f"  {col_name} {data_type}{null_str}")
                    col_names.append(col_name)
                schema_parts.append(f"CREATE TABLE {table_name} (")
                schema_parts.append(",\n".join(col_defs))
                schema_parts.append(");")

                # Sample rows
                try:
                    rows = conn.execute(
                        text(f'SELECT * FROM "{table_name}" LIMIT 3')
                    ).fetchall()
                    schema_parts.append(f"-- Sample rows ({', '.join(col_names)}):")
                    for row in rows:
                        schema_parts.append(f"--   {tuple(row)}")
                except Exception:
                    schema_parts.append("-- (no sample rows available)")
                schema_parts.append("")
    finally:
        engine.dispose()

    schema_str = "\n".join(schema_parts)
    logger.info(f"Schema extracted successfully with {len(schema_parts)} parts.")

    # Store in schema cache
    if settings.CACHE_ENABLED:
        _schema_cache.put(db_url, schema_str)

    return {"db_schema": schema_str}


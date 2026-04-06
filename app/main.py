"""LangGraph AI Service — FastAPI entrypoint.

This service handles AI orchestration separately from the main API.
It accepts dynamic database connection info from the API service.
"""

import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from .config import settings
from .graph import analytics_graph
from .state import AgentState
from .cache import SemanticCache
from .nodes.schema_extractor import get_schema_cache

app = FastAPI(
    title="LangGraph AI Service",
    description="AI orchestration service for data analytics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize semantic cache singleton
_query_cache = SemanticCache(
    db_url=settings.database_url,
    similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
    ttl_seconds=settings.CACHE_TTL_SECONDS,
    max_size=settings.CACHE_MAX_SIZE,
)


class DbConnectionInfo(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    user: str = "postgres"
    password: str = ""
    schema: str = "public"
    sslmode: str = "disable"

    @property
    def connection_url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?options=-csearch_path%3D{self.schema}&sslmode={self.sslmode}"
        )


class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    session_id: str = Field(default="", description="Session ID for follow-ups")
    history: list[dict[str, str]] = Field(default=[], description="Conversational history")
    db_connection: DbConnectionInfo | None = None


class AnalyzeResponse(BaseModel):
    session_id: str
    user_query: str
    generated_sql: str = ""
    insight_title: str = ""
    narrative_summary: str = ""
    recommended_actions: list[str] = []
    echarts_config: dict[str, Any] = {}
    raw_data: list[dict[str, Any]] = []
    sql_errors: list[str] = []
    status: str = "success"
    error: str | None = None
    cache_hit: bool = False


@app.get("/health")
async def health():
    return {"status": "ok", "service": "langgraph"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Run the full LangGraph analytics pipeline."""
    session_id = request.session_id or str(uuid.uuid4())

    # Build database URL from connection info or fall back to config
    if request.db_connection:
        db_url = request.db_connection.connection_url
    else:
        db_url = settings.database_url

    # Check semantic cache first
    if settings.CACHE_ENABLED:
        cached_result = _query_cache.get(request.query, user_db_url=db_url)
        if cached_result is not None:
            logger.info(f"Returning cached result for session {session_id}")
            return AnalyzeResponse(
                session_id=session_id,
                user_query=request.query,
                cache_hit=True,
                **cached_result,
            )

    initial_state: AgentState = {
        "session_id": session_id,
        "user_query": request.query,
        "history": request.history,
        "db_schema": "",
        "generated_sql": "",
        "sql_errors": [],
        "raw_data": [],
        "insight_title": "",
        "narrative_summary": "",
        "recommended_actions": [],
        "echarts_config": {},
        "selected_chart_type": "",
        "retry_count": 0,
        "status": "running",
        "db_url": db_url,
        "sql_executed": False,
    }

    try:
        logger.info(f"Starting analysis for session {session_id} with query: '{request.query}'")
        result = analytics_graph.invoke(initial_state)
        logger.info(f"Graph execution completed for session {session_id} with status: {result.get('status')}")

        status = result.get("status", "success")
        error_msg = None
        if status == "rejected" and result.get("sql_errors"):
            error_msg = result["sql_errors"][-1]

        response_data = {
            "generated_sql": result.get("generated_sql", ""),
            "insight_title": result.get("insight_title", ""),
            "narrative_summary": result.get("narrative_summary", ""),
            "recommended_actions": result.get("recommended_actions", []),
            "echarts_config": result.get("echarts_config", {}),
            "raw_data": result.get("raw_data", []),
            "sql_errors": result.get("sql_errors", []),
            "status": status,
            "error": error_msg,
        }

        # Store in semantic cache on success
        if settings.CACHE_ENABLED and status == "success":
            _query_cache.put(request.query, user_db_url=db_url, result=response_data)

        return AnalyzeResponse(
            session_id=result.get("session_id", session_id),
            user_query=result.get("user_query", request.query),
            cache_hit=False,
            **response_data,
        )
    except Exception as e:
        return AnalyzeResponse(
            session_id=session_id,
            user_query=request.query,
            status="error",
            error=str(e),
            sql_errors=[str(e)],
        )


# ── Cache management endpoints ──────────────────────────────────────


@app.get("/cache/stats")
async def cache_stats():
    """Return cache statistics for monitoring."""
    return {
        "query_cache": _query_cache.stats(),
        "schema_cache": get_schema_cache().stats(),
        "cache_enabled": settings.CACHE_ENABLED,
    }


@app.delete("/cache")
async def cache_invalidate():
    """Invalidate all caches."""
    _query_cache.invalidate_all()
    get_schema_cache().invalidate()
    return {"status": "ok", "message": "All caches invalidated"}


@app.delete("/cache/query")
async def cache_invalidate_query():
    """Invalidate only the query cache."""
    _query_cache.invalidate_all()
    return {"status": "ok", "message": "Query cache invalidated"}


@app.delete("/cache/schema")
async def cache_invalidate_schema():
    """Invalidate only the schema cache."""
    get_schema_cache().invalidate()
    return {"status": "ok", "message": "Schema cache invalidated"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.SERVICE_PORT)

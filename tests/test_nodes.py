"""Tests for the LangGraph service nodes and graph structure."""

import pytest
from unittest.mock import patch, MagicMock
from app.state import AgentState
from app.nodes.text_to_sql import text_to_sql
from app.nodes.execute_sql import execute_sql, _serialize_value
from app.nodes.schema_extractor import schema_extractor
from app.nodes.guardrail import query_guardrail
from app.nodes.chart_selector import chart_selector, _heuristic_fallback, _analyze_data_shape
from app.graph import analytics_graph, should_retry_or_continue


def _make_state(**overrides) -> AgentState:
    """Helper to create a valid AgentState with sensible defaults."""
    defaults: AgentState = {
        "session_id": "test",
        "user_query": "test query",
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
        "db_url": "postgresql://test:test@localhost/test",
        "sql_executed": False,
    }
    defaults.update(overrides)
    return defaults


class TestAgentState:
    """Tests for the AgentState type."""

    def test_state_fields(self):
        state = _make_state()
        assert state["session_id"] == "test"
        assert state["db_url"] == "postgresql://test:test@localhost/test"
        assert state["retry_count"] == 0
        assert state["sql_errors"] == []
        assert state["selected_chart_type"] == ""


class TestSerializeValue:
    """Tests for the SQL result serializer."""

    def test_serialize_date(self):
        from datetime import date
        result = _serialize_value(date(2024, 1, 15))
        assert result == "2024-01-15"

    def test_serialize_datetime(self):
        from datetime import datetime
        result = _serialize_value(datetime(2024, 1, 15, 10, 30))
        assert result == "2024-01-15T10:30:00"

    def test_serialize_decimal(self):
        from decimal import Decimal
        result = _serialize_value(Decimal("19.99"))
        assert result == 19.99
        assert isinstance(result, float)

    def test_serialize_string(self):
        result = _serialize_value("hello")
        assert result == "hello"

    def test_serialize_int(self):
        result = _serialize_value(42)
        assert result == 42

    def test_serialize_none(self):
        result = _serialize_value(None)
        assert result is None


class TestShouldRetryOrContinue:
    """Tests for the retry conditional edge."""

    def test_retry_on_error(self):
        state = {
            "sql_errors": ["Some error"],
            "retry_count": 1,
        }
        assert should_retry_or_continue(state) == "retry"

    def test_no_retry_after_max(self):
        state = {
            "sql_errors": ["err1", "err2", "err3"],
            "retry_count": 3,
        }
        assert should_retry_or_continue(state) == "max_retries"

    def test_proceed_on_success(self):
        state = {
            "sql_errors": [],
            "retry_count": 0,
            "raw_data": [{"col": 1}],
            "sql_executed": True,
        }
        assert should_retry_or_continue(state) == "success"


class TestTextToSql:
    """Tests for the text_to_sql node."""

    @patch("app.nodes.text_to_sql.get_llm")
    def test_blocks_destructive_sql(self, mock_get_llm):
        """Should reject DELETE/DROP/INSERT/UPDATE queries."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="DELETE FROM users;")
        mock_get_llm.return_value = mock_llm

        state = _make_state(user_query="delete all users", db_schema="CREATE TABLE users (id INT);")
        result = text_to_sql(state)
        assert len(result.get("sql_errors", [])) > 0
        assert "read-only" in result["sql_errors"][-1].lower() or "dangerous" in result["sql_errors"][-1].lower()


class TestExecuteSql:
    """Tests for the execute_sql node."""

    def test_no_sql_provided(self):
        state = _make_state()
        result = execute_sql(state)
        assert "No SQL query generated" in result["sql_errors"]
        assert result["retry_count"] == 1

    def test_no_db_url(self):
        state = _make_state(generated_sql="SELECT 1", db_url="")
        result = execute_sql(state)
        assert "No database connection configured" in result["sql_errors"]


class TestSchemaExtractor:
    """Tests for the schema_extractor node."""

    def test_no_db_url(self):
        state = _make_state(db_url="")
        result = schema_extractor(state)
        assert "No database connection" in result["db_schema"]


class TestGraphStructure:
    """Tests for the compiled LangGraph structure."""

    def test_graph_is_compiled(self):
        assert analytics_graph is not None

    def test_graph_has_nodes(self):
        """Verify all expected nodes are in the graph."""
        node_names = list(analytics_graph.nodes.keys())
        assert "schema_extractor" in node_names
        assert "query_guardrail" in node_names
        assert "text_to_sql" in node_names
        assert "execute_sql" in node_names
        assert "generate_insights" in node_names
        assert "chart_selector" in node_names
        assert "generate_chart" in node_names


class TestGuardrail:
    """Tests for the query_guardrail node."""

    @patch("app.nodes.guardrail.get_llm")
    def test_guardrail_approves_related(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(is_related=True, reason="Related to sales.")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _make_state(user_query="how are sales doing?", db_schema="table sales ()")
        result = query_guardrail(state)
        assert result["status"] == "approved"

    @patch("app.nodes.guardrail.get_llm")
    def test_guardrail_rejects_unrelated(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(is_related=False, reason="Off-topic joke request.")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _make_state(user_query="tell me a joke about dogs", db_schema="table users ()")
        result = query_guardrail(state)
        assert result["status"] == "rejected"
        assert len(result["sql_errors"]) == 1
        assert "Off-topic" in result["sql_errors"][0]


class TestChartSelector:
    """Tests for the chart_selector node."""

    @patch("app.nodes.chart_selector.get_llm")
    def test_selects_chart_type_via_llm(self, mock_get_llm):
        """Should return a valid chart type from the LLM."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(chart_type="pie", reason="Composition data.")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _make_state(
            user_query="What is the revenue breakdown by product?",
            raw_data=[{"product": "A", "revenue": 100}, {"product": "B", "revenue": 200}],
        )
        result = chart_selector(state)
        assert result["selected_chart_type"] == "pie"

    @patch("app.nodes.chart_selector.get_llm")
    def test_falls_back_on_invalid_type(self, mock_get_llm):
        """Should use heuristic fallback when LLM returns an invalid chart type."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(chart_type="invalid_type", reason="Test.")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _make_state(
            user_query="Show sales by product",
            raw_data=[{"product": "A", "sales": 100}],
        )
        result = chart_selector(state)
        # Should fall back to heuristic (bar for category comparison)
        assert result["selected_chart_type"] in {"bar", "line", "area", "pie", "donut", "bubble", "stacked_bar"}

    @patch("app.nodes.chart_selector.get_llm")
    def test_falls_back_on_llm_error(self, mock_get_llm):
        """Should use heuristic fallback when LLM raises an exception."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("LLM API error")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = _make_state(
            user_query="Show monthly trend",
            raw_data=[{"month": "Jan", "value": 10}, {"month": "Feb", "value": 20}],
        )
        result = chart_selector(state)
        assert result["selected_chart_type"] in {"bar", "line", "area", "pie", "donut", "bubble", "stacked_bar"}

    def test_defaults_to_bar_on_empty_data(self):
        """Should default to bar chart when there is no data."""
        state = _make_state(user_query="Show something", raw_data=[])
        result = chart_selector(state)
        assert result["selected_chart_type"] == "bar"


class TestHeuristicFallback:
    """Tests for the heuristic chart type fallback logic."""

    def test_trend_query_returns_line(self):
        data = [{"date": "2024-01", "revenue": 100}, {"date": "2024-02", "revenue": 200}]
        assert _heuristic_fallback(data, "Show revenue trend over time") == "line"

    def test_breakdown_query_returns_donut(self):
        data = [{"product": "A", "share": 30}, {"product": "B", "share": 70}]
        assert _heuristic_fallback(data, "Show revenue breakdown by product") == "donut"

    def test_breakdown_many_categories_returns_pie(self):
        data = [{"cat": f"C{i}", "val": i * 10} for i in range(10)]
        assert _heuristic_fallback(data, "Show the distribution of categories") == "pie"

    def test_compare_with_3_numeric_returns_bubble(self):
        data = [{"x": 1, "y": 2, "z": 3}]
        assert _heuristic_fallback(data, "Compare x vs y vs z") == "bubble"

    def test_compare_without_numeric_returns_stacked_bar(self):
        data = [{"category": "A", "value": 10}]
        assert _heuristic_fallback(data, "Compare sales versus targets") == "stacked_bar"

    def test_date_column_returns_line(self):
        data = [{"date": "2024-01", "val": 100}]
        assert _heuristic_fallback(data, "Show me the data") == "line"

    def test_generic_query_returns_bar(self):
        data = [{"product": "A", "sales": 100}]
        assert _heuristic_fallback(data, "Show me product sales") == "bar"


class TestAnalyzeDataShape:
    """Tests for the data shape analysis helper."""

    def test_empty_data(self):
        result = _analyze_data_shape([])
        assert result["row_count"] == 0

    def test_basic_data(self):
        data = [
            {"product": "A", "revenue": 100, "count": 5},
            {"product": "B", "revenue": 200, "count": 10},
        ]
        result = _analyze_data_shape(data)
        assert result["row_count"] == 2
        assert "product" in result["columns"]
        assert result["unique_first_col"] == 2
        assert "revenue" in result["numeric_columns"]
        assert "count" in result["numeric_columns"]

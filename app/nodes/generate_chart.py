"""Generate chart config node — LLM produces ECharts option JSON for the selected chart type."""

import json
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from ..llm import get_llm
from ..state import AgentState
from ..prompts import CHART_PROMPT
import logging

logger = logging.getLogger(__name__)

BRAND_COLORS = ["#2dd4bf", "#1e3a8a", "#38bdf8", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]


class ChartOutput(BaseModel):
    """Structured output for ECharts configuration."""
    chart_type: str = Field(
        ..., description="The chart type: 'bar', 'line', 'area', 'pie', 'donut', 'bubble', or 'stacked_bar'"
    )
    echarts_option: dict[str, Any] = Field(
        ..., description="A complete, valid Apache ECharts option JSON object"
    )


def generate_chart(state: AgentState) -> dict[str, Any]:
    """Generate an ECharts configuration based on the query results and selected chart type."""
    logger.info("Executing node: generate_chart")
    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(ChartOutput)

    raw_data = state.get("raw_data", [])
    selected_chart_type = state.get("selected_chart_type", "bar")

    if selected_chart_type == "none":
        logger.info("Chart type is 'none', skipping chart generation")
        return {"echarts_config": {}}

    if len(raw_data) > 100:
        data_preview = raw_data[:100]
    else:
        data_preview = raw_data

    prompt = CHART_PROMPT.format(
        user_query=state["user_query"],
        raw_data=json.dumps(data_preview, indent=2, default=str),
        selected_chart_type=selected_chart_type,
    )

    try:
        result: ChartOutput = structured_llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"Successfully generated chart config of type: {result.chart_type}")
        return {"echarts_config": result.echarts_option}
    except Exception as e:
        logger.error(f"Failed to generate structured chart: {str(e)}", exc_info=True)
        logger.info(f"Using fallback chart configuration for type: {selected_chart_type}")
        return {"echarts_config": _fallback_chart(raw_data, state["user_query"], selected_chart_type)}


def _fallback_chart(raw_data: list[dict], query: str, chart_type: str = "bar") -> dict[str, Any]:
    """Generate a fallback chart config when LLM fails, appropriate for the selected type."""
    if not raw_data:
        return {}

    keys = list(raw_data[0].keys())
    x_key = keys[0]
    y_key = next((k for k in keys[1:] if isinstance(raw_data[0].get(k), (int, float))), keys[-1])

    title_config = {"text": query[:60], "left": "center", "textStyle": {"fontSize": 14}}
    tooltip_config = {"trigger": "axis"}
    grid_config = {"bottom": "20%", "left": "10%", "right": "10%"}

    x_data = [str(row.get(x_key, "")) for row in raw_data[:20]]
    y_data = [row.get(y_key, 0) for row in raw_data[:20]]

    if chart_type in ("pie", "donut"):
        pie_data = [{"name": str(row.get(x_key, "")), "value": row.get(y_key, 0)} for row in raw_data[:10]]
        radius = ["45%", "70%"] if chart_type == "donut" else ["0%", "70%"]
        return {
            "title": title_config,
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"bottom": "5%", "left": "center"},
            "series": [
                {
                    "type": "pie",
                    "radius": radius,
                    "data": pie_data,
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.2)",
                        }
                    },
                    "label": {"formatter": "{b}: {d}%"},
                    "itemStyle": {"borderRadius": 6, "borderColor": "#fff", "borderWidth": 2},
                }
            ],
            "color": BRAND_COLORS,
        }

    if chart_type == "line":
        return {
            "title": title_config,
            "tooltip": tooltip_config,
            "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30}},
            "yAxis": {"type": "value"},
            "series": [
                {
                    "type": "line",
                    "data": y_data,
                    "smooth": True,
                    "areaStyle": {"opacity": 0.05},
                    "itemStyle": {"color": BRAND_COLORS[0]},
                    "lineStyle": {"width": 2},
                }
            ],
            "grid": grid_config,
        }

    if chart_type == "area":
        return {
            "title": title_config,
            "tooltip": tooltip_config,
            "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30}},
            "yAxis": {"type": "value"},
            "series": [
                {
                    "type": "line",
                    "data": y_data,
                    "smooth": True,
                    "areaStyle": {"opacity": 0.4},
                    "itemStyle": {"color": BRAND_COLORS[0]},
                    "lineStyle": {"width": 2},
                }
            ],
            "grid": grid_config,
        }

    if chart_type == "bubble":
        # For bubble fallback, use first 3 numeric columns or repeat y_key
        numeric_keys = [k for k in keys if any(isinstance(row.get(k), (int, float)) for row in raw_data[:5])]
        if len(numeric_keys) >= 3:
            scatter_data = [
                [row.get(numeric_keys[0], 0), row.get(numeric_keys[1], 0), row.get(numeric_keys[2], 0)]
                for row in raw_data[:30]
            ]
        else:
            scatter_data = [[i, row.get(y_key, 0), max(1, abs(row.get(y_key, 0)))] for i, row in enumerate(raw_data[:30])]

        return {
            "title": title_config,
            "tooltip": {"trigger": "item"},
            "xAxis": {"type": "value", "name": numeric_keys[0] if len(numeric_keys) >= 1 else "X"},
            "yAxis": {"type": "value", "name": numeric_keys[1] if len(numeric_keys) >= 2 else "Y"},
            "series": [
                {
                    "type": "scatter",
                    "data": scatter_data,
                    "symbolSize": 20,
                    "itemStyle": {"color": BRAND_COLORS[0], "opacity": 0.7},
                }
            ],
            "grid": grid_config,
        }

    if chart_type == "stacked_bar":
        # Try to group by a second dimension
        if len(keys) >= 3:
            group_key = keys[1] if keys[1] != y_key else keys[0]
            groups = {}
            for row in raw_data[:50]:
                g = str(row.get(group_key, ""))
                if g not in groups:
                    groups[g] = {}
                cat = str(row.get(x_key, ""))
                groups[g][cat] = row.get(y_key, 0)

            categories = list(dict.fromkeys(str(row.get(x_key, "")) for row in raw_data[:50]))
            series = []
            for i, (group_name, values) in enumerate(list(groups.items())[:6]):
                series.append({
                    "name": group_name,
                    "type": "bar",
                    "stack": "total",
                    "data": [values.get(c, 0) for c in categories],
                    "itemStyle": {"color": BRAND_COLORS[i % len(BRAND_COLORS)]},
                })

            return {
                "title": title_config,
                "tooltip": {"trigger": "axis"},
                "legend": {"bottom": "0%"},
                "xAxis": {"type": "category", "data": categories, "axisLabel": {"rotate": 30}},
                "yAxis": {"type": "value"},
                "series": series,
                "grid": {"bottom": "15%"},
            }

    # Default: bar chart
    return {
        "title": title_config,
        "tooltip": tooltip_config,
        "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30}},
        "yAxis": {"type": "value"},
        "series": [
            {
                "type": "bar",
                "data": y_data,
                "itemStyle": {"color": BRAND_COLORS[0], "borderRadius": [4, 4, 0, 0]},
            }
        ],
        "grid": grid_config,
    }

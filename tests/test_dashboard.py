"""Tests for the Grafana dashboard JSON."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def dashboard() -> dict:
    path = Path(__file__).parent.parent / "dashboards" / "semantic-cache.json"
    assert path.exists(), f"Dashboard file not found: {path}"
    return json.loads(path.read_text())


def _get_all_panels(dashboard: dict) -> list[dict]:
    """Flatten panels including those nested in row panels."""
    panels = dashboard.get("panels", [])
    all_panels: list[dict] = []
    for p in panels:
        if p.get("type") == "row":
            all_panels.extend(p.get("panels", []))
        else:
            all_panels.append(p)
    return all_panels


def test_dashboard_has_eight_panels(dashboard: dict) -> None:
    assert len(_get_all_panels(dashboard)) == 8


def test_dashboard_has_namespace_template_variable(dashboard: dict) -> None:
    variables = dashboard.get("templating", {}).get("list", [])
    names = [v["name"] for v in variables]
    assert "namespace" in names


def test_stream_bypass_panel_filters_by_namespace(dashboard: dict) -> None:
    """Verify the stream bypass panel PromQL filters by namespace variable."""
    panels = _get_all_panels(dashboard)
    stream_panel = next(p for p in panels if p["title"] == "Stream Bypass Rate")
    expr = stream_panel["targets"][0]["expr"]
    assert "namespace" in expr, f"Stream bypass panel should filter by namespace, got: {expr}"


def test_similarity_histogram_uses_increase(dashboard: dict) -> None:
    """Verify similarity histogram uses increase() not raw cumulative buckets."""
    panels = _get_all_panels(dashboard)
    sim_panel = next(p for p in panels if p["title"] == "Similarity Score Distribution")
    expr = sim_panel["targets"][0]["expr"]
    assert expr.startswith("increase("), f"Similarity histogram should use increase(), got: {expr}"

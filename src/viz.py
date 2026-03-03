"""Graphviz SVG rendering of the knowledge graph with mastery color coding
and hollow concept indicators."""

from __future__ import annotations

from datetime import datetime

import graphviz

from src.coverage import (
    CoverageCalculator,
    STATUS_DECAYING,
    STATUS_FAILING,
    STATUS_FLAKY,
    STATUS_STABLE,
    STATUS_UNTESTED,
)
from src.graph_store import KnowledgeGraph

# Status → hex color mapping
STATUS_COLORS: dict[str, str] = {
    STATUS_STABLE: "#22c55e",
    STATUS_DECAYING: "#eab308",
    STATUS_FLAKY: "#f97316",
    STATUS_FAILING: "#ef4444",
    STATUS_UNTESTED: "#6b7280",
}

HOLLOW_BORDER_COLOR = "#ef4444"


def render_graph(
    graph: KnowledgeGraph,
    coverage: CoverageCalculator,
    now: datetime | None = None,
) -> str:
    """Render the knowledge graph as an SVG string.

    Each node is color-coded by retrieval status (fill at 25% opacity)
    and hollow concepts get a red border with penwidth 3.

    Args:
        graph: The knowledge graph to render.
        coverage: Coverage calculator for status and hollow detection.
        now: Optional datetime for retrieval strength computation.

    Returns:
        SVG string of the rendered graph.
    """
    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="TB", bgcolor="white")

    concepts = graph.concepts

    for cid, concept in concepts.items():
        status = coverage.retrieval_status(cid, now=now)
        hollow = coverage.is_hollow(cid, now=now)
        strength = coverage.retrieval_strength(cid, now=now)

        color = STATUS_COLORS.get(status, STATUS_COLORS[STATUS_UNTESTED])
        fill_color = color + "40"  # 25% opacity

        # Use concept name if available, otherwise fall back to ID
        label = f"{concept.name or concept.id}\n{strength:.0%}"

        if hollow:
            border_color = HOLLOW_BORDER_COLOR
            penwidth = "3"
        else:
            border_color = color
            penwidth = "1"

        dot.node(
            cid,
            label=label,
            style="filled",
            fillcolor=fill_color,
            color=border_color,
            penwidth=penwidth,
        )

    # Edges: prerequisite → dependent
    for cid, concept in concepts.items():
        for prereq_id in concept.prerequisites:
            if prereq_id in concepts:
                dot.edge(prereq_id, cid)

    return dot.pipe(encoding="utf-8")

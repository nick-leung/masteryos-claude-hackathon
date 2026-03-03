"""Prerequisite inference engine — propagates mastery signals up and down
the knowledge graph and selects the next concept to assess."""

from __future__ import annotations

from src.graph_store import KnowledgeGraph


class PrerequisiteInference:
    """Propagates mastery signals through a knowledge graph and picks the
    next concept for assessment."""

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate_success(self, concept_id: str) -> None:
        """Boost direct prerequisites that have confidence < 0.7.

        For each such prereq: mastery += 0.15 (capped at 0.7),
        confidence set to 0.4 (inferred).
        """
        prereqs = self._graph.get_prerequisites(concept_id)
        for pid in prereqs:
            concept = self._graph.get(pid)
            if concept is None:
                continue
            if concept.confidence >= 0.7:
                continue
            concept.mastery_estimate = min(0.7, concept.mastery_estimate + 0.15)
            concept.confidence = 0.4

    def propagate_failure(self, concept_id: str) -> None:
        """Multiply confidence by 0.6 for all transitive dependents."""
        dependents = self._graph.get_all_dependents(concept_id)
        for did in dependents:
            concept = self._graph.get(did)
            if concept is None:
                continue
            concept.confidence *= 0.6

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_next_concept(self) -> str | None:
        """Pick the highest-priority concept for assessment.

        Score = (1 - confidence) × max(1, connectivity)
        Connectivity = len(all_prereqs) + len(all_dependents)
        Tie-break: topological order (foundational first).
        Skip mastery-confirmed: mastery >= 0.7 AND confidence >= 0.7.
        Returns None if all concepts are mastery-confirmed.
        """
        topo = self._graph.topological_sort()
        topo_index = {cid: i for i, cid in enumerate(topo)}

        best_id: str | None = None
        best_score: float = -1.0
        best_topo: int = len(topo)  # worst possible

        for cid, concept in self._graph.concepts.items():
            # Skip mastery-confirmed
            if concept.mastery_estimate >= 0.7 and concept.confidence >= 0.7:
                continue

            all_prereqs = self._graph.get_all_prerequisites(cid)
            all_deps = self._graph.get_all_dependents(cid)
            connectivity = len(all_prereqs) + len(all_deps)
            score = (1.0 - concept.confidence) * max(1, connectivity)

            tidx = topo_index.get(cid, len(topo))

            if (score > best_score) or (score == best_score and tidx < best_topo):
                best_score = score
                best_topo = tidx
                best_id = cid

        return best_id

"""KnowledgeGraph — NetworkX DAG wrapper for concept storage, traversal,
and JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import networkx as nx

from src.models import Concept, Misconception, RetrievalEvent, from_dict, to_dict


class KnowledgeGraph:
    """A directed acyclic graph of :class:`Concept` nodes backed by NetworkX."""

    def __init__(self, domain_name: str = ""):
        self.domain_name = domain_name
        self._g: nx.DiGraph = nx.DiGraph()
        self._concepts: dict[str, Concept] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_concept(self, concept: Concept) -> None:
        """Add a concept and its prerequisite edges. Rolls back on cycle."""
        cid = concept.id
        # Snapshot for rollback
        had_node = cid in self._concepts
        old_concept = self._concepts.get(cid)
        old_edges: list[tuple[str, str]] = []

        if had_node:
            old_edges = list(self._g.in_edges(cid))
            self._g.remove_edges_from(old_edges)

        self._concepts[cid] = concept
        self._g.add_node(cid)

        # Add prerequisite edges: prereq → concept
        for prereq_id in concept.prerequisites:
            if prereq_id in self._concepts:
                self._g.add_edge(prereq_id, cid)

        # Also add edges from this concept to dependents that list it as prereq
        for other_id, other in self._concepts.items():
            if other_id != cid and cid in other.prerequisites:
                if not self._g.has_edge(cid, other_id):
                    self._g.add_edge(cid, other_id)

        # Cycle check — rollback if needed
        if not nx.is_directed_acyclic_graph(self._g):
            # Rollback
            self._g.remove_node(cid)
            if had_node:
                self._concepts[cid] = old_concept
                self._g.add_node(cid)
                for u, v in old_edges:
                    self._g.add_edge(u, v)
                # Re-add outgoing edges
                for other_id, other in self._concepts.items():
                    if other_id != cid and cid in other.prerequisites:
                        self._g.add_edge(cid, other_id)
            else:
                del self._concepts[cid]
            raise ValueError(f"Adding concept {cid!r} would create a cycle")

    def remove_concept(self, concept_id: str) -> None:
        """Remove a concept and clean up prerequisite lists of dependents."""
        if concept_id not in self._concepts:
            raise KeyError(f"Concept {concept_id!r} not found")

        # Clean prerequisite lists of concepts that reference this one
        for other in self._concepts.values():
            if concept_id in other.prerequisites:
                other.prerequisites.remove(concept_id)

        del self._concepts[concept_id]
        self._g.remove_node(concept_id)

    def get(self, concept_id: str) -> Concept | None:
        """Return a concept by id, or None if not found."""
        return self._concepts.get(concept_id)

    @property
    def concepts(self) -> dict[str, Concept]:
        """Read-only access to concept dict."""
        return dict(self._concepts)

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._concepts

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_prerequisites(self, concept_id: str) -> list[str]:
        """Direct prerequisites (one level)."""
        if concept_id not in self._g:
            raise KeyError(f"Concept {concept_id!r} not found")
        return list(self._g.predecessors(concept_id))

    def get_all_prerequisites(self, concept_id: str) -> set[str]:
        """Transitive closure of prerequisites (ancestors)."""
        if concept_id not in self._g:
            raise KeyError(f"Concept {concept_id!r} not found")
        return nx.ancestors(self._g, concept_id)

    def get_dependents(self, concept_id: str) -> list[str]:
        """Direct dependents (one level)."""
        if concept_id not in self._g:
            raise KeyError(f"Concept {concept_id!r} not found")
        return list(self._g.successors(concept_id))

    def get_all_dependents(self, concept_id: str) -> set[str]:
        """Transitive closure of dependents (descendants)."""
        if concept_id not in self._g:
            raise KeyError(f"Concept {concept_id!r} not found")
        return nx.descendants(self._g, concept_id)

    def topological_sort(self) -> list[str]:
        """Return concepts in topological order."""
        return list(nx.topological_sort(self._g))

    def detect_cycles(self) -> list[tuple[str, str]]:
        """Return cycle edges if any exist (should be empty post-validation)."""
        try:
            return list(nx.find_cycle(self._g))
        except nx.NetworkXNoCycle:
            return []

    def get_roots(self) -> list[str]:
        """Concepts with no prerequisites (in_degree == 0)."""
        return [n for n in self._g.nodes() if self._g.in_degree(n) == 0]

    def get_leaves(self) -> list[str]:
        """Concepts with no dependents (out_degree == 0)."""
        return [n for n in self._g.nodes() if self._g.out_degree(n) == 0]

    def dependency_count(self, concept_id: str) -> int:
        """Number of transitive dependents."""
        return len(self.get_all_dependents(concept_id))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save graph to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "domain_name": self.domain_name,
            "concepts": [to_dict(c) for c in self._concepts.values()],
        }
        p.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        """Load graph from JSON. Two-pass: create concepts, then add edges."""
        p = Path(path)
        data = json.loads(p.read_text())
        kg = cls(domain_name=data.get("domain_name", ""))

        # Pass 1: create all concepts without prerequisites
        raw_concepts: list[dict] = data.get("concepts", [])
        prereq_map: dict[str, list[str]] = {}
        for cdata in raw_concepts:
            prereqs = cdata.pop("prerequisites", [])
            concept = from_dict(Concept, {**cdata, "prerequisites": []})
            kg._concepts[concept.id] = concept
            kg._g.add_node(concept.id)
            prereq_map[concept.id] = prereqs

        # Pass 2: add prerequisites and edges (skip dangling refs)
        for cid, prereqs in prereq_map.items():
            valid_prereqs = [p for p in prereqs if p in kg._concepts]
            kg._concepts[cid].prerequisites = valid_prereqs
            for prereq_id in valid_prereqs:
                kg._g.add_edge(prereq_id, cid)

        return kg

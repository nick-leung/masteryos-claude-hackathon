"""Multi-pass Claude extraction pipeline: text chunks → validated KnowledgeGraph DAG."""

from __future__ import annotations

from src.claude_calls import ClaudeCalls
from src.graph_store import KnowledgeGraph
from src.models import (
    Concept,
    ConceptDef,
    GraphBuilderOutput,
    Misconception,
    TextChunk,
    _sanitize_slug,
    to_dict,
)


def _merge_concepts(
    pass1: GraphBuilderOutput, pass2: GraphBuilderOutput
) -> list[ConceptDef]:
    """Merge concepts from two passes.

    Union by id; richer description wins; prerequisites and misconceptions merged.
    All IDs are sanitized on entry.
    """
    by_id: dict[str, ConceptDef] = {}

    for concept in [*pass1.concepts, *pass2.concepts]:
        cid = _sanitize_slug(concept.id)
        if cid in by_id:
            existing = by_id[cid]
            # Richer description wins (longer text)
            if len(concept.description) > len(existing.description):
                existing.description = concept.description
            if len(concept.name) > len(existing.name):
                existing.name = concept.name
            # Merge prerequisites (union, sanitized)
            existing_prereqs = set(existing.prerequisites)
            for p in concept.prerequisites:
                sp = _sanitize_slug(p)
                if sp not in existing_prereqs:
                    existing.prerequisites.append(sp)
                    existing_prereqs.add(sp)
            # Merge misconceptions (union)
            existing_miscon = set(existing.misconceptions)
            for m in concept.misconceptions:
                if m not in existing_miscon:
                    existing.misconceptions.append(m)
                    existing_miscon.add(m)
        else:
            # Sanitize IDs on the way in
            concept.id = cid
            concept.prerequisites = [_sanitize_slug(p) for p in concept.prerequisites]
            by_id[cid] = concept

    return list(by_id.values())


def _remove_dangling_prereqs(concepts: list[ConceptDef]) -> list[ConceptDef]:
    """Remove prerequisites that reference non-existent concept ids."""
    known_ids = {c.id for c in concepts}
    for c in concepts:
        c.prerequisites = [p for p in c.prerequisites if p in known_ids]
    return concepts


def _break_cycles(concepts: list[ConceptDef]) -> list[ConceptDef]:
    """Break cycles in the prerequisite graph using DFS back-edge removal.

    Builds a forward adjacency (prereq → dependent) and removes the last
    back-edge found during DFS traversal.
    """
    # Forward adjacency: edge from prereq to dependent
    forward: dict[str, list[str]] = {c.id: [] for c in concepts}
    for c in concepts:
        for p in c.prerequisites:
            if p in forward:
                forward[p].append(c.id)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {c.id: WHITE for c in concepts}
    back_edges: list[tuple[str, str]] = []

    def dfs(u: str) -> None:
        color[u] = GRAY
        for v in sorted(forward.get(u, [])):  # sorted for determinism
            if color[v] == GRAY:
                back_edges.append((u, v))
            elif color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    for c in sorted(concepts, key=lambda c: c.id):  # deterministic start order
        if color[c.id] == WHITE:
            dfs(c.id)

    # Remove back-edges: edge (u, v) means u→v in forward graph,
    # which means v lists u as a prerequisite
    edges_to_remove = set(back_edges)
    for c in concepts:
        c.prerequisites = [
            p for p in c.prerequisites if (p, c.id) not in edges_to_remove
        ]

    return concepts


def _build_dag(
    concepts: list[ConceptDef], domain_name: str
) -> KnowledgeGraph:
    """Insert concepts into KnowledgeGraph in topological order.

    Uses Kahn's algorithm for ordering. Falls back to force-add with
    progressively dropped prerequisites if insertion fails.
    """
    kg = KnowledgeGraph(domain_name)

    # Kahn's algorithm for topological order
    in_degree: dict[str, int] = {c.id: 0 for c in concepts}
    forward: dict[str, list[str]] = {c.id: [] for c in concepts}
    concept_map: dict[str, ConceptDef] = {c.id: c for c in concepts}

    for c in concepts:
        for p in c.prerequisites:
            if p in forward:
                forward[p].append(c.id)
                in_degree[c.id] += 1

    queue = sorted(cid for cid, deg in in_degree.items() if deg == 0)
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for dep in sorted(forward[node]):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
        queue.sort()

    # Safety: append any remaining nodes (shouldn't happen post cycle-breaking)
    remaining = [cid for cid in concept_map if cid not in set(order)]
    remaining.sort()
    order.extend(remaining)

    # Insert in topological order
    for cid in order:
        cdef = concept_map[cid]
        misconceptions = [
            Misconception(
                id=_sanitize_slug(m) if m.strip() else "unnamed",
                description=m,
                concept_id=cid,
            )
            for m in cdef.misconceptions
        ]
        concept = Concept(
            id=cid,
            name=cdef.name,
            description=cdef.description,
            prerequisites=list(cdef.prerequisites),
            known_misconceptions=misconceptions,
        )
        try:
            kg.add_concept(concept)
        except ValueError:
            # Force-add fallback: drop prereqs one at a time until it works
            prereqs = list(concept.prerequisites)
            added = False
            while prereqs and not added:
                prereqs.pop()
                retry = Concept(
                    id=cid,
                    name=cdef.name,
                    description=cdef.description,
                    prerequisites=list(prereqs),
                    known_misconceptions=misconceptions,
                )
                try:
                    kg.add_concept(retry)
                    added = True
                except ValueError:
                    continue

            if not added:
                # Last resort: no prereqs at all
                fallback = Concept(
                    id=cid,
                    name=cdef.name,
                    description=cdef.description,
                    prerequisites=[],
                    known_misconceptions=misconceptions,
                )
                try:
                    kg.add_concept(fallback)
                except ValueError:
                    pass  # Concept already exists — skip

    return kg


def build(
    chunks: list[TextChunk],
    domain_name: str,
    claude: ClaudeCalls | None = None,
) -> KnowledgeGraph:
    """Multi-pass extraction pipeline: extract → review → merge → validate → build DAG.

    Args:
        chunks: Ingested text chunks.
        domain_name: Human-readable domain name.
        claude: Optional ClaudeCalls instance (creates default if None).

    Returns:
        A validated KnowledgeGraph.

    Raises:
        ValueError: If fewer than 3 concepts are extracted after merge.
    """
    if claude is None:
        claude = ClaudeCalls()

    # Pass 1: Extract
    pass1 = claude.build_graph(chunks)

    # Pass 2: Review
    graph_data = [to_dict(c) for c in pass1.concepts]
    pass2 = claude.review_graph(graph_data, chunks)

    # Merge
    merged = _merge_concepts(pass1, pass2)

    # Belt-and-suspenders slug sanitization
    for c in merged:
        c.id = _sanitize_slug(c.id)
        c.prerequisites = [_sanitize_slug(p) for p in c.prerequisites]

    # Deduplicate by ID
    seen: dict[str, ConceptDef] = {}
    for c in merged:
        if c.id not in seen:
            seen[c.id] = c
    deduped = list(seen.values())

    # Validate: minimum 3 concepts
    if len(deduped) < 3:
        raise ValueError(
            f"Too few concepts extracted ({len(deduped)}); minimum is 3"
        )

    # Remove dangling prereqs
    deduped = _remove_dangling_prereqs(deduped)

    # Break cycles
    deduped = _break_cycles(deduped)

    # Build DAG
    return _build_dag(deduped, domain_name)

"""FSRS-based spaced repetition scheduler — grades retrieval events, updates
stability/difficulty, computes recall probability, and manages review queues."""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

from src.graph_store import KnowledgeGraph
from src.models import _clamp01


# Grade constants
AGAIN = 1
HARD = 2
GOOD = 3
EASY = 4

# FSRS parameters per grade
INITIAL_STABILITY = {AGAIN: 0.1, HARD: 0.5, GOOD: 1.0, EASY: 4.0}
GROWTH_FACTORS = {AGAIN: 0.2, HARD: 0.8, GOOD: 2.0, EASY: 3.5}
DIFFICULTY_ADJ = {AGAIN: 0.15, HARD: 0.05, GOOD: 0.0, EASY: -0.10}


class FSRSScheduler:
    """FSRS spaced repetition scheduler operating on a KnowledgeGraph.

    Grades retrieval events, updates stability/difficulty per concept,
    computes recall probability, and manages review queues.
    """

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph

    # ------------------------------------------------------------------
    # Grade mapping
    # ------------------------------------------------------------------

    @staticmethod
    def grade_response(correct: bool, reasoning_quality: str) -> int:
        """Map assessment outcome to FSRS grade (1-4).

        - Incorrect (any reasoning) -> AGAIN (1)
        - Correct + weak -> HARD (2)
        - Correct + misconception -> GOOD (3) (contradictory input edge case)
        - Correct + strong -> EASY (4)
        """
        if not correct:
            return AGAIN
        if reasoning_quality == "weak":
            return HARD
        if reasoning_quality == "misconception":
            return GOOD
        # strong -> EASY
        return EASY

    # ------------------------------------------------------------------
    # Stability / difficulty updates
    # ------------------------------------------------------------------

    def record_review(self, concept_id: str, grade: int, now: datetime | None = None) -> None:
        """Update stability and difficulty for a concept after a review.

        For first review (stability == 0), uses initial stability values.
        For subsequent reviews, applies growth factor and difficulty modifier.
        """
        concept = self._graph.get(concept_id)
        if concept is None:
            raise KeyError(f"Concept not found: {concept_id!r}")

        if now is None:
            now = datetime.now(timezone.utc)

        # Update difficulty (clamp 0-1)
        concept.difficulty_param = _clamp01(concept.difficulty_param + DIFFICULTY_ADJ[grade])

        # Update stability
        if concept.stability == 0.0:
            # Never reviewed — use initial stability for this grade
            concept.stability = INITIAL_STABILITY[grade]
        else:
            difficulty_modifier = 1.0 - 0.5 * concept.difficulty_param
            concept.stability = max(0.1, concept.stability * GROWTH_FACTORS[grade] * difficulty_modifier)

        concept.last_reviewed = now.isoformat()

    # ------------------------------------------------------------------
    # Recall probability
    # ------------------------------------------------------------------

    def recall_probability(self, concept_id: str, now: datetime | None = None) -> float:
        """Compute recall probability R(t) = e^(-t/S).

        Returns 0.0 if the concept has never been reviewed (stability == 0).
        Returns 1.0 if reviewed at or after `now`.
        """
        concept = self._graph.get(concept_id)
        if concept is None:
            raise KeyError(f"Concept not found: {concept_id!r}")

        if concept.stability == 0.0 or concept.last_reviewed is None:
            return 0.0

        if now is None:
            now = datetime.now(timezone.utc)

        last = datetime.fromisoformat(concept.last_reviewed)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        t = (now - last).total_seconds() / 86400.0  # days
        if t <= 0:
            return 1.0

        return math.exp(-t / concept.stability)

    # ------------------------------------------------------------------
    # Review queue
    # ------------------------------------------------------------------

    def get_review_queue(self, now: datetime | None = None) -> list[str]:
        """Return concept IDs sorted by review priority (highest first).

        Priority = overdue_days x (1 + downstream_count) x (1 / prereq_depth)

        prereq_depth = len(all_prerequisites) + 1 (to avoid division by zero).
        For never-reviewed concepts, overdue_days defaults to 1.0.
        Only includes concepts with overdue_days > 0.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        scored: list[tuple[float, str]] = []

        for cid, concept in self._graph.concepts.items():
            if concept.last_reviewed is None or concept.stability == 0.0:
                # Never reviewed — treat as overdue with 1 day
                overdue_days = 1.0
            else:
                last = datetime.fromisoformat(concept.last_reviewed)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                t = (now - last).total_seconds() / 86400.0
                overdue_days = max(0.0, t - concept.stability)

            if overdue_days <= 0:
                continue

            downstream_count = self._graph.dependency_count(cid)
            prereq_depth = len(self._graph.get_all_prerequisites(cid)) + 1

            priority = overdue_days * (1 + downstream_count) * (1.0 / prereq_depth)
            scored.append((priority, cid))

        # Sort by priority descending, then alphabetically for determinism
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [cid for _, cid in scored]

    # ------------------------------------------------------------------
    # Knowledge debt
    # ------------------------------------------------------------------

    def get_knowledge_debt(self, now: datetime | None = None) -> list[str]:
        """Return concepts previously mastered but now decayed below R < 0.5.

        Previously mastered: mastery_estimate >= 0.7 OR 3+ correct retrievals.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        debt: list[str] = []

        for cid, concept in self._graph.concepts.items():
            correct_count = sum(1 for e in concept.retrieval_history if e.correct)
            was_mastered = concept.mastery_estimate >= 0.7 or correct_count >= 3

            if not was_mastered:
                continue

            r = self.recall_probability(cid, now=now)
            if r < 0.5:
                debt.append(cid)

        return debt

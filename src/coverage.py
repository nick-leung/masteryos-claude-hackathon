"""Five-level coverage calculator with hollow detection, retrieval status
classification, and priority queue generation."""

from __future__ import annotations

from datetime import datetime, timezone

from src.graph_store import KnowledgeGraph
from src.models import CoverageReport, HollowWarning, _clamp01
from src.scheduler import FSRSScheduler


# Retrieval status constants
STATUS_STABLE = "🟢"
STATUS_DECAYING = "🟡"
STATUS_FLAKY = "🟠"
STATUS_FAILING = "🔴"
STATUS_UNTESTED = "⚫"

# Priority ordering (lower index = higher priority)
_STATUS_PRIORITY = {
    STATUS_FAILING: 0,
    STATUS_FLAKY: 2,
    STATUS_DECAYING: 3,
    STATUS_UNTESTED: 4,
    STATUS_STABLE: 5,
}
# Hollow is priority 1 but handled separately since it's not a status


class CoverageCalculator:
    """Computes five coverage levels, retrieval status, hollow warnings,
    and a priority study queue for all concepts in a KnowledgeGraph.

    Args:
        graph: The knowledge graph containing concepts.
        scheduler: The FSRS scheduler for recall probability computation.
    """

    def __init__(self, graph: KnowledgeGraph, scheduler: FSRSScheduler):
        self._graph = graph
        self._scheduler = scheduler

    # ------------------------------------------------------------------
    # Level 1: Concept Coverage
    # ------------------------------------------------------------------

    def concept_coverage(self) -> float:
        """Fraction of concepts that have been assessed at least once."""
        total = len(self._graph)
        if total == 0:
            return 0.0
        assessed = sum(
            1 for c in self._graph.concepts.values()
            if len(c.retrieval_history) > 0
        )
        return assessed / total

    # ------------------------------------------------------------------
    # Level 2: Retrieval Strength
    # ------------------------------------------------------------------

    def retrieval_strength(self, concept_id: str, now: datetime | None = None) -> float:
        """Recall probability R(t) = e^(-t/S) for a single concept.

        Returns 0.0 for untested concepts.
        """
        return self._scheduler.recall_probability(concept_id, now=now)

    def mean_retrieval_strength(self, now: datetime | None = None) -> float:
        """Mean retrieval strength across all concepts."""
        concepts = self._graph.concepts
        if not concepts:
            return 0.0
        total = sum(self.retrieval_strength(cid, now=now) for cid in concepts)
        return total / len(concepts)

    # ------------------------------------------------------------------
    # Level 3: Prerequisite Coverage
    # ------------------------------------------------------------------

    def prerequisite_coverage(self, concept_id: str, now: datetime | None = None) -> float:
        """Min retrieval strength among direct prerequisites.

        Returns 1.0 if the concept has no prerequisites.
        """
        prereqs = self._graph.get_prerequisites(concept_id)
        if not prereqs:
            return 1.0
        return min(self.retrieval_strength(pid, now=now) for pid in prereqs)

    def mean_prerequisite_coverage(self, now: datetime | None = None) -> float:
        """Mean prerequisite coverage across all concepts."""
        concepts = self._graph.concepts
        if not concepts:
            return 0.0
        total = sum(self.prerequisite_coverage(cid, now=now) for cid in concepts)
        return total / len(concepts)

    # ------------------------------------------------------------------
    # Level 4: Transfer Coverage
    # ------------------------------------------------------------------

    def transfer_coverage(self, concept_id: str) -> float:
        """Fraction of retrieval events at difficulty >= 4 (high-level).

        Returns 0.0 if the concept has no retrieval history.
        """
        concept = self._graph.get(concept_id)
        if concept is None:
            raise KeyError(f"Concept not found: {concept_id!r}")
        events = concept.retrieval_history
        if not events:
            return 0.0
        high_level = sum(1 for e in events if e.difficulty_level >= 4)
        return high_level / len(events)

    def mean_transfer_coverage(self) -> float:
        """Mean transfer coverage across all concepts."""
        concepts = self._graph.concepts
        if not concepts:
            return 0.0
        total = sum(self.transfer_coverage(cid) for cid in concepts)
        return total / len(concepts)

    # ------------------------------------------------------------------
    # Level 5: Misconception Coverage
    # ------------------------------------------------------------------

    def misconception_coverage(self, concept_id: str) -> float:
        """Fraction of known misconceptions ruled out.

        Returns 1.0 if the concept has no known misconceptions.
        """
        concept = self._graph.get(concept_id)
        if concept is None:
            raise KeyError(f"Concept not found: {concept_id!r}")
        misconceptions = concept.known_misconceptions
        if not misconceptions:
            return 1.0
        ruled_out = sum(1 for m in misconceptions if m.ruled_out)
        return ruled_out / len(misconceptions)

    def mean_misconception_coverage(self) -> float:
        """Mean misconception coverage across all concepts."""
        concepts = self._graph.concepts
        if not concepts:
            return 0.0
        total = sum(self.misconception_coverage(cid) for cid in concepts)
        return total / len(concepts)

    # ------------------------------------------------------------------
    # Retrieval Status
    # ------------------------------------------------------------------

    def retrieval_status(self, concept_id: str, now: datetime | None = None) -> str:
        """Classify a concept's retrieval status.

        Returns one of: 🟢 Stable, 🟡 Decaying, 🟠 Flaky, 🔴 Failing, ⚫ Untested.
        """
        concept = self._graph.get(concept_id)
        if concept is None:
            raise KeyError(f"Concept not found: {concept_id!r}")

        events = concept.retrieval_history
        if not events:
            return STATUS_UNTESTED

        r = self.retrieval_strength(concept_id, now=now)

        # 🔴 Failing: R < 0.5 OR last retrieval incorrect
        if r < 0.5 or not events[-1].correct:
            return STATUS_FAILING

        # 🟠 Flaky: R >= 0.5 but any of last 3 incorrect
        last_3 = events[-3:] if len(events) >= 3 else events
        if any(not e.correct for e in last_3):
            return STATUS_FLAKY

        # 🟢 Stable: R >= 0.8 AND last 2 correct
        if r >= 0.8 and len(events) >= 2 and all(e.correct for e in events[-2:]):
            return STATUS_STABLE

        # 🟡 Decaying: everything else (0.5 <= R < 0.8, or R >= 0.8 but insufficient history)
        return STATUS_DECAYING

    # ------------------------------------------------------------------
    # Hollow Detection
    # ------------------------------------------------------------------

    def is_hollow(self, concept_id: str, now: datetime | None = None) -> bool:
        """Check if a concept is hollow: strong own strength but weak prerequisites.

        Hollow = own_strength > 0.7 AND prerequisite_coverage < 0.4
        """
        own = self.retrieval_strength(concept_id, now=now)
        prereq = self.prerequisite_coverage(concept_id, now=now)
        return own > 0.7 and prereq < 0.4

    def get_hollow_warnings(self, now: datetime | None = None) -> list[HollowWarning]:
        """Return HollowWarning for each hollow concept."""
        warnings: list[HollowWarning] = []
        for cid in self._graph.concepts:
            if not self.is_hollow(cid, now=now):
                continue
            # Identify the weak prerequisites
            weak = [
                pid for pid in self._graph.get_prerequisites(cid)
                if self.retrieval_strength(pid, now=now) < 0.4
            ]
            warnings.append(HollowWarning(concept_id=cid, weak_prerequisites=weak))
        return warnings

    # ------------------------------------------------------------------
    # Overall Score
    # ------------------------------------------------------------------

    def overall_score(self, now: datetime | None = None) -> float:
        """Weighted overall coverage: 0.30×CC + 0.30×RS + 0.20×PC + 0.10×TC + 0.10×MC."""
        cc = self.concept_coverage()
        rs = self.mean_retrieval_strength(now=now)
        pc = self.mean_prerequisite_coverage(now=now)
        tc = self.mean_transfer_coverage()
        mc = self.mean_misconception_coverage()
        return _clamp01(0.30 * cc + 0.30 * rs + 0.20 * pc + 0.10 * tc + 0.10 * mc)

    # ------------------------------------------------------------------
    # Priority Queue
    # ------------------------------------------------------------------

    def priority_queue(self, now: datetime | None = None) -> list[str]:
        """Return concept IDs in study priority order.

        Order: Failing → Hollow → Flaky → Decaying → Untested → Stable.
        Within each group: more downstream dependents first.
        """
        # Compute status and hollow flag for each concept
        entries: list[tuple[int, int, str]] = []  # (priority, -dependents, cid)
        for cid in self._graph.concepts:
            status = self.retrieval_status(cid, now=now)
            hollow = self.is_hollow(cid, now=now)

            # Determine priority bucket
            if status == STATUS_FAILING:
                prio = 0
            elif hollow:
                prio = 1
            else:
                prio = _STATUS_PRIORITY[status]

            dep_count = self._graph.dependency_count(cid)
            entries.append((prio, -dep_count, cid))

        entries.sort(key=lambda x: (x[0], x[1], x[2]))
        return [cid for _, _, cid in entries]

    # ------------------------------------------------------------------
    # Full Report
    # ------------------------------------------------------------------

    def generate_report(self, now: datetime | None = None) -> CoverageReport:
        """Generate a full CoverageReport with all metrics."""
        concepts = self._graph.concepts
        total = len(concepts)
        assessed = sum(1 for c in concepts.values() if len(c.retrieval_history) > 0)
        mastered = sum(1 for c in concepts.values() if c.mastery_estimate >= 0.7)

        cc = self.concept_coverage()
        rs = self.mean_retrieval_strength(now=now)
        pc = self.mean_prerequisite_coverage(now=now)
        tc = self.mean_transfer_coverage()
        mc = self.mean_misconception_coverage()
        score = self.overall_score(now=now)

        return CoverageReport(
            total_concepts=total,
            assessed_count=assessed,
            mastered_count=mastered,
            coverage_percent=cc,
            breadth_percent=cc,
            depth_score=rs,
            recency_score=pc,
            confidence_score=tc,
            overall_score=score,
            hollow_warnings=self.get_hollow_warnings(now=now),
            priority_queue=self.priority_queue(now=now),
        )

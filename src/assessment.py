"""Adaptive assessment engine with exponential difficulty ramp-up and
session orchestrator that routes between concepts based on Claude's evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.claude_calls import ClaudeCalls
from src.coverage import CoverageCalculator
from src.graph_store import KnowledgeGraph
from src.inference import PrerequisiteInference
from src.models import (
    AssessmentResult,
    CoverageReport,
    GeneratedQuestion,
    RetrievalEvent,
    _clamp_difficulty,
)
from src.scheduler import FSRSScheduler


# ---------------------------------------------------------------------------
# Exponential ramp-up
# ---------------------------------------------------------------------------


def compute_next_level(
    current_level: int,
    correct: bool,
    reasoning_quality: str,
    streak: int = 1,
) -> int:
    """Compute the next difficulty level based on assessment outcome.

    Rules:
        - Correct + strong/misconception reasoning: jump min(2^(streak-1), 5-current)
        - Correct + weak reasoning: stay at current level
        - Incorrect: drop 1 level (minimum 1)

    Args:
        current_level: Current difficulty level (1-5).
        correct: Whether the answer was correct.
        reasoning_quality: One of 'strong', 'weak', 'misconception'.
        streak: Consecutive correct+strong/misconception streak count.

    Returns:
        Next difficulty level (1-5).
    """
    if not correct:
        return max(1, current_level - 1)
    if reasoning_quality == "weak":
        return current_level
    # strong or misconception — exponential jump
    jump = min(2 ** (streak - 1), 5 - current_level)
    return _clamp_difficulty(current_level + jump)


# ---------------------------------------------------------------------------
# Internal session state
# ---------------------------------------------------------------------------


@dataclass
class _SessionState:
    """Internal mutable session state."""
    mode: str = "adaptive"
    max_questions: int = 10
    coverage_threshold: float = 0.8
    current_concept_id: str | None = None
    current_level: int = 3
    streak: int = 0
    question_count: int = 0
    pending_question: GeneratedQuestion | None = None
    active: bool = False


# ---------------------------------------------------------------------------
# Session orchestrator
# ---------------------------------------------------------------------------


class SessionOrchestrator:
    """Manages assessment sessions: concept selection, question generation,
    answer evaluation, routing, and session termination.

    Args:
        graph: The knowledge graph.
        claude: Claude API wrapper for question generation / evaluation.
        scheduler: FSRS scheduler for spaced repetition updates.
        inference: Prerequisite inference engine for concept selection.
        coverage: Coverage calculator for threshold checks and reports.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        claude: ClaudeCalls,
        scheduler: FSRSScheduler,
        inference: PrerequisiteInference,
        coverage: CoverageCalculator,
    ):
        self._graph = graph
        self._claude = claude
        self._scheduler = scheduler
        self._inference = inference
        self._coverage = coverage
        self._state = _SessionState()

    # ------------------------------------------------------------------
    # Public state access
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict:
        """Public read-only snapshot of session state."""
        return {
            "mode": self._state.mode,
            "max_questions": self._state.max_questions,
            "coverage_threshold": self._state.coverage_threshold,
            "current_concept_id": self._state.current_concept_id,
            "current_level": self._state.current_level,
            "streak": self._state.streak,
            "question_count": self._state.question_count,
            "active": self._state.active,
            "has_pending_question": self._state.pending_question is not None,
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        mode: str = "adaptive",
        max_questions: int = 10,
        coverage_threshold: float = 0.8,
    ) -> GeneratedQuestion | None:
        """Start a new assessment session.

        Returns the first GeneratedQuestion, or None if all concepts are
        already mastered (session ends immediately).
        """
        self._state = _SessionState(
            mode=mode,
            max_questions=max_questions,
            coverage_threshold=coverage_threshold,
            active=True,
        )

        # Select first concept
        concept_id = self._inference.select_next_concept()
        if concept_id is None:
            self._state.active = False
            return None

        self._state.current_concept_id = concept_id
        self._state.current_level = 3

        # Generate first question
        question = self._generate_question(concept_id, self._state.current_level)
        self._state.pending_question = question
        self._state.question_count = 1
        return question

    def submit_answer(self, answer: str) -> tuple[AssessmentResult, GeneratedQuestion | None]:
        """Submit an answer to the pending question.

        Empty/whitespace answers are treated as incorrect with weak reasoning.

        Returns:
            Tuple of (AssessmentResult, next GeneratedQuestion or None if session ended).

        Raises:
            RuntimeError: If no active session or no pending question.
        """
        if not self._state.active:
            raise RuntimeError("No active session")
        if self._state.pending_question is None:
            raise RuntimeError("No pending question to answer")

        question = self._state.pending_question
        self._state.pending_question = None
        concept_id = question.concept_id
        concept = self._graph.get(concept_id)

        # --- Evaluate ---
        if not answer or not answer.strip():
            result = AssessmentResult(
                correct=False,
                difficulty_level=question.difficulty_level,
                reasoning_quality="weak",
                explanation="Empty answer submitted.",
                new_mastery_estimate=concept.mastery_estimate if concept else 0.0,
                question_type=question.question_type,
                next_action="drop_level",
            )
        else:
            result = self._claude.evaluate_answer(
                concept_name=concept.name if concept else "",
                concept_description=concept.description if concept else "",
                concept_id=concept_id,
                question_text=question.question_text,
                answer_text=answer,
                difficulty=question.difficulty_level,
                history=concept.retrieval_history if concept else [],
            )

        # --- Record retrieval event and update models ---
        if concept is not None:
            event = RetrievalEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                difficulty_level=result.difficulty_level,
                correct=result.correct,
                reasoning_quality=result.reasoning_quality,
                misconception_id=result.misconception_id,
                question_type=result.question_type,
                response_text=answer or "",
            )
            concept.retrieval_history.append(event)
            concept.mastery_estimate = result.new_mastery_estimate

            # Spaced repetition update
            grade = self._scheduler.grade_response(result.correct, result.reasoning_quality)
            self._scheduler.record_review(concept_id, grade)

            # Propagate through inference engine
            if result.correct:
                self._inference.propagate_success(concept_id)
            else:
                self._inference.propagate_failure(concept_id)

        # --- Update streak ---
        if result.correct and result.reasoning_quality in ("strong", "misconception"):
            self._state.streak += 1
        else:
            self._state.streak = 0

        # --- Check termination ---
        if self._should_terminate():
            self._state.active = False
            return result, None

        # --- Route to next concept + level ---
        next_concept_id, next_level = self._decide_next(result)
        self._state.current_concept_id = next_concept_id
        self._state.current_level = next_level

        # --- Generate next question ---
        next_question = self._generate_question(next_concept_id, next_level)
        self._state.pending_question = next_question
        self._state.question_count += 1

        return result, next_question

    def end_session(self) -> CoverageReport:
        """End the current session and return a coverage report."""
        self._state.active = False
        return self._coverage.generate_report()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_question(self, concept_id: str, level: int) -> GeneratedQuestion:
        """Generate a question for a concept at a given difficulty level."""
        concept = self._graph.get(concept_id)
        if concept is None:
            raise KeyError(f"Concept not found: {concept_id!r}")
        return self._claude.generate_question(
            concept_name=concept.name,
            concept_description=concept.description,
            concept_id=concept_id,
            difficulty=level,
            history=concept.retrieval_history,
        )

    def _should_terminate(self) -> bool:
        """Check whether the session should end."""
        if self._state.question_count >= self._state.max_questions:
            return True
        if self._coverage.overall_score() >= self._state.coverage_threshold:
            return True
        return False

    def _decide_next(self, result: AssessmentResult) -> tuple[str, int]:
        """Route to the next concept and difficulty level based on Claude's next_action.

        Routing table:
            mastery_confirmed  → next concept via inference engine; level 3
            test_prerequisite  → weakest prerequisite; level 3 (fallback: advance)
            drop_level         → same concept; current_level - 1
            probe_lateral      → related concept with lowest confidence; level 3 (fallback: advance)
            advance (default)  → same concept; compute_next_level()
        """
        action = result.next_action
        current_id = self._state.current_concept_id
        current_level = self._state.current_level

        if action == "mastery_confirmed":
            next_id = self._inference.select_next_concept()
            if next_id is None:
                return current_id, 3
            return next_id, 3

        if action == "test_prerequisite":
            prereqs = self._graph.get_prerequisites(current_id)
            if not prereqs:
                return self._advance(current_id, current_level, result)
            weakest = min(
                prereqs,
                key=lambda pid: self._graph.get(pid).confidence
                if self._graph.get(pid) else 1.0,
            )
            return weakest, 3

        if action == "drop_level":
            return current_id, max(1, current_level - 1)

        if action == "probe_lateral":
            related = self._find_related(current_id)
            if not related:
                return self._advance(current_id, current_level, result)
            target = min(
                related,
                key=lambda cid: self._graph.get(cid).confidence
                if self._graph.get(cid) else 1.0,
            )
            return target, 3

        # "advance" (default)
        return self._advance(current_id, current_level, result)

    def _advance(
        self, concept_id: str, current_level: int, result: AssessmentResult
    ) -> tuple[str, int]:
        """Advance on the same concept with computed next level."""
        new_level = compute_next_level(
            current_level, result.correct, result.reasoning_quality, self._state.streak
        )
        return concept_id, new_level

    def _find_related(self, concept_id: str) -> set[str]:
        """Find related concepts: siblings (sharing a prerequisite) + direct dependents."""
        related: set[str] = set()
        prereqs = self._graph.get_prerequisites(concept_id)
        for pid in prereqs:
            siblings = self._graph.get_dependents(pid)
            related.update(siblings)
        dependents = self._graph.get_dependents(concept_id)
        related.update(dependents)
        related.discard(concept_id)
        return related

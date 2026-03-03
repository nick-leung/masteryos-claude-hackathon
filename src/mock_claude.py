"""Offline MockClaudeCalls — returns canned data from captured API responses."""

from __future__ import annotations

import json
import logging
from itertools import cycle
from pathlib import Path

from src.models import (
    AssessmentResult, ConceptDef, GeneratedQuestion,
    GraphBuilderOutput, TextChunk,
)

logger = logging.getLogger(__name__)

_DEFAULT_CAPTURE_DIR = Path(__file__).parent.parent / "logs" / "api_responses"


def _load_captures(capture_dir: Path, prefix: str) -> list[dict]:
    """Load all captured response JSON files matching a prefix, sorted by time."""
    files = sorted(capture_dir.glob(f"{prefix}_*.json"))
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results


class MockClaudeCalls:
    """Drop-in replacement for ClaudeCalls that returns captured responses.

    No API calls are made. Responses cycle through the captured data.
    """

    def __init__(self, capture_dir: str | Path | None = None):
        self.capture_dir = Path(capture_dir) if capture_dir else _DEFAULT_CAPTURE_DIR
        self._questions = cycle(_load_captures(self.capture_dir, "generate_question"))
        self._evaluations = cycle(_load_captures(self.capture_dir, "evaluate_answer"))
        logger.info(f"[MockClaudeCalls] Loaded captures from {self.capture_dir}")

    def build_graph(self, chunks: list[TextChunk]) -> GraphBuilderOutput:
        """Return empty graph -- demo route builds from review data instead."""
        return GraphBuilderOutput(concepts=[])

    def review_graph(
        self, graph_data: list[dict], chunks: list[TextChunk],
    ) -> GraphBuilderOutput:
        """Return empty graph -- demo route doesn't need this."""
        return GraphBuilderOutput(concepts=[])

    def generate_question(
        self,
        concept_name: str,
        concept_description: str,
        concept_id: str,
        difficulty: int,
        history=None,
        domain_name: str = "",
    ) -> GeneratedQuestion:
        """Return next captured question, adapting concept_id and difficulty."""
        data = next(self._questions)
        r = data["response"]
        return GeneratedQuestion(
            concept_id=concept_id,
            difficulty_level=difficulty,
            question_type=r.get("question_type", "recall"),
            question_text=r["question_text"],
            rubric=r.get("rubric", ""),
        )

    def evaluate_answer(
        self,
        concept_name: str,
        concept_description: str,
        concept_id: str,
        question_text: str,
        answer_text: str,
        difficulty: int,
        history=None,
    ) -> AssessmentResult:
        """Return next captured evaluation."""
        data = next(self._evaluations)
        r = data["response"]
        return AssessmentResult(
            correct=r["correct"],
            difficulty_level=difficulty,
            reasoning_quality=r["reasoning_quality"],
            misconception_id=r.get("misconception_id"),
            explanation=r.get("explanation", ""),
            new_mastery_estimate=r.get("new_mastery_estimate", 0.0),
            question_type=r.get("question_type", ""),
            next_action=r.get("next_action", "advance"),
        )

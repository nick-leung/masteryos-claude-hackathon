"""All dataclasses for MasteryOS: core domain models, Claude structured output schemas,
report models, and JSON persistence utilities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Validation / clamping helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')


def _validate_slug(value: str, field_name: str = "slug") -> str:
    """Strict slug validation — raise ValueError on bad slugs."""
    if not value or not _SLUG_RE.match(value):
        raise ValueError(f"Invalid {field_name}: {value!r} — must be lowercase alphanumeric with hyphens")
    return value


def _sanitize_slug(value: str) -> str:
    """Lenient slug sanitization — normalize to lowercase, hyphens, digits."""
    s = value.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    s = s.strip('-')
    return s if s else "unnamed"


def _clamp01(v: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(v)))


def _clamp_difficulty(v: int) -> int:
    """Clamp an integer to [1, 5]."""
    return max(1, min(5, int(v)))


_VALID_REASONING = {"strong", "weak", "misconception"}
_VALID_QUESTION_TYPES = {"recognition", "recall", "application", "analysis", "transfer", ""}
_VALID_NEXT_ACTIONS = {"advance", "probe_lateral", "drop_level", "mastery_confirmed", "test_prerequisite"}

# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


@dataclass
class Misconception:
    """A known misconception related to a concept."""
    id: str
    description: str
    concept_id: str
    tested: bool = False
    detected: bool = False
    ruled_out: bool = False

    def __post_init__(self):
        _validate_slug(self.id, "misconception id")
        _validate_slug(self.concept_id, "concept_id")


@dataclass
class RetrievalEvent:
    """A single assessment retrieval event."""
    timestamp: str
    difficulty_level: int
    correct: bool
    reasoning_quality: str
    misconception_id: str | None = None
    question_type: str = ""
    response_text: str = ""

    def __post_init__(self):
        self.difficulty_level = _clamp_difficulty(self.difficulty_level)
        if self.reasoning_quality not in _VALID_REASONING:
            raise ValueError(f"Invalid reasoning_quality: {self.reasoning_quality!r}")
        if self.question_type not in _VALID_QUESTION_TYPES:
            raise ValueError(f"Invalid question_type: {self.question_type!r}")


@dataclass
class Concept:
    """A single concept node in the knowledge graph."""
    id: str
    name: str
    description: str = ""
    prerequisites: list[str] = field(default_factory=list)
    known_misconceptions: list[Misconception] = field(default_factory=list)
    mastery_estimate: float = 0.0
    confidence: float = 0.0
    retrieval_history: list[RetrievalEvent] = field(default_factory=list)
    last_reviewed: str | None = None
    next_review: str | None = None
    stability: float = 0.0
    difficulty_param: float = 0.0

    def __post_init__(self):
        _validate_slug(self.id, "concept id")
        self.mastery_estimate = _clamp01(self.mastery_estimate)
        self.confidence = _clamp01(self.confidence)
        self.stability = max(0.0, float(self.stability))

# ---------------------------------------------------------------------------
# Claude structured output schemas
# ---------------------------------------------------------------------------


@dataclass
class ConceptDef:
    """A concept definition as returned by Claude during graph building."""
    id: str
    name: str
    description: str = ""
    prerequisites: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    code_examples: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.id = _sanitize_slug(self.id)
        # Guard against string instead of list (toolslm may pass raw string)
        if isinstance(self.prerequisites, str):
            self.prerequisites = [s.strip() for s in self.prerequisites.split(",") if s.strip()]
        if isinstance(self.misconceptions, str):
            self.misconceptions = [s.strip() for s in self.misconceptions.split(",") if s.strip()]
        if isinstance(self.code_examples, str):
            self.code_examples = [self.code_examples] if self.code_examples.strip() else []
        self.prerequisites = [_sanitize_slug(p) for p in self.prerequisites if p and str(p).strip()]


@dataclass
class GraphBuilderOutput:
    """Claude's output: a list of concept definitions."""
    concepts: list[ConceptDef] = field(default_factory=list)


@dataclass
class GeneratedQuestion:
    """An assessment question generated by Claude."""
    concept_id: str
    difficulty_level: int
    question_type: str
    question_text: str
    rubric: str = ""

    def __init__(self, concept_id, difficulty_level, question_type, question_text, rubric="", **_extra):
        self.concept_id = _sanitize_slug(concept_id)
        self.difficulty_level = _clamp_difficulty(difficulty_level)
        self.question_type = question_type
        self.question_text = question_text
        self.rubric = rubric


@dataclass
class AssessmentResult:
    """Claude's evaluation of a student's answer."""
    correct: bool
    difficulty_level: int
    reasoning_quality: str
    misconception_id: str | None = None
    explanation: str = ""
    new_mastery_estimate: float = 0.0
    question_type: str = ""
    next_action: str = "advance"

    def __init__(self, correct, difficulty_level, reasoning_quality,
                 misconception_id=None, explanation="", new_mastery_estimate=0.0,
                 question_type="", next_action="advance", **_extra):
        self.correct = correct
        self.difficulty_level = _clamp_difficulty(difficulty_level)
        if reasoning_quality not in _VALID_REASONING:
            raise ValueError(f"Invalid reasoning_quality: {reasoning_quality!r}")
        self.reasoning_quality = reasoning_quality
        self.misconception_id = misconception_id
        self.explanation = explanation
        if question_type not in _VALID_QUESTION_TYPES:
            raise ValueError(f"Invalid question_type: {question_type!r}")
        self.question_type = question_type
        if next_action not in _VALID_NEXT_ACTIONS:
            raise ValueError(f"Invalid next_action: {next_action!r}")
        self.next_action = next_action
        self.new_mastery_estimate = _clamp01(new_mastery_estimate)

# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


@dataclass
class HollowWarning:
    """Warning about a strong concept with weak prerequisites."""
    concept_id: str
    weak_prerequisites: list[str] = field(default_factory=list)

    def __post_init__(self):
        _validate_slug(self.concept_id, "concept_id")


@dataclass
class CoverageReport:
    """Full coverage report with all 5 metrics, hollow warnings, priority queue."""
    total_concepts: int = 0
    assessed_count: int = 0
    mastered_count: int = 0
    coverage_percent: float = 0.0
    breadth_percent: float = 0.0
    depth_score: float = 0.0
    recency_score: float = 0.0
    confidence_score: float = 0.0
    overall_score: float = 0.0
    hollow_warnings: list[HollowWarning] = field(default_factory=list)
    priority_queue: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.coverage_percent = _clamp01(self.coverage_percent)
        self.breadth_percent = _clamp01(self.breadth_percent)
        self.depth_score = _clamp01(self.depth_score)
        self.recency_score = _clamp01(self.recency_score)
        self.confidence_score = _clamp01(self.confidence_score)
        self.overall_score = _clamp01(self.overall_score)


@dataclass
class TextChunk:
    """A chunk of ingested text with source metadata."""
    text: str
    source_type: str = ""
    source_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 0

# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

# Map of (parent_class, field_name) → child class for nested reconstruction
_NESTED_MAP: dict[tuple[type, str], type] = {
    (Concept, "known_misconceptions"): Misconception,
    (Concept, "retrieval_history"): RetrievalEvent,
    (GraphBuilderOutput, "concepts"): ConceptDef,
    (CoverageReport, "hollow_warnings"): HollowWarning,
}


def to_dict(obj: Any) -> Any:
    """Recursive asdict conversion for dataclasses."""
    return asdict(obj)


def from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Reconstruct a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}

    # Recursively reconstruct nested dataclasses in list fields
    for (parent, fname), child_cls in _NESTED_MAP.items():
        if cls is parent and fname in filtered and isinstance(filtered[fname], list):
            filtered[fname] = [
                from_dict(child_cls, item) if isinstance(item, dict) else item
                for item in filtered[fname]
            ]
    return cls(**filtered)


def save_json(obj: Any, path: str | Path) -> None:
    """Save a dataclass (or list/dict) to JSON, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = to_dict(obj) if hasattr(obj, '__dataclass_fields__') else obj
    p.write_text(json.dumps(data, indent=2, default=str))


def load_json(cls_or_path: type | str | Path, path: str | Path | None = None) -> Any:
    """Load JSON from file. If cls provided, reconstruct typed dataclass."""
    if path is None:
        # Called as load_json(path)
        p = Path(cls_or_path)
        return json.loads(p.read_text())
    else:
        # Called as load_json(cls, path)
        p = Path(path)
        data = json.loads(p.read_text())
        return from_dict(cls_or_path, data)

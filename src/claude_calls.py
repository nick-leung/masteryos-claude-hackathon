"""Structured Claude API integration via claudette.

Three call types: graph building/review, question generation, and answer evaluation.
Uses tool-based structured output with multi-turn extraction for graph operations.
"""

from __future__ import annotations

import json
import logging
import re as _re
import time
from dataclasses import asdict
from pathlib import Path

import tiktoken
from claudette import Client, Chat, mk_msg, models
from toolslm.funccall import mk_ns
import toolslm.funccall as _tslm

# Patch toolslm _coerce_inputs to handle generic types (list[str] etc.)
# that are callable in Python 3.12+ but shouldn't be called as constructors.
# TODO: remove once toolslm handles generic aliases natively.
try:
    _orig_coerce = _tslm._coerce_inputs
    def _patched_coerce(func, inputs):
        "Coerce inputs, skipping generic aliases like list[str]"
        from typing import get_type_hints
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        res = {}
        for k, v in inputs.items():
            ann = hints.get(k)
            origin = getattr(ann, '__origin__', None)
            if origin is not None:
                # Generic type like list[str], dict[str,int] — don't try to call it
                res[k] = v
            elif ann in _tslm.custom_types:
                res[k] = ann(v)
            elif isinstance(v, dict) and callable(ann):
                res[k] = ann(**v)
            else:
                res[k] = v
        return res
    _tslm._coerce_inputs = _patched_coerce
except AttributeError:
    pass  # toolslm API changed; patch no longer needed

from src.models import (
    AssessmentResult,
    ConceptDef,
    GeneratedQuestion,
    GraphBuilderOutput,
    RetrievalEvent,
    TextChunk,
)

logger = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")
_MAX_ANSWER_TOKENS = 2000

# ---------------------------------------------------------------------------
# System prompts (ported from v4 — source-grounded, code-aware)
# ---------------------------------------------------------------------------

_GRAPH_BUILD_SP = """\
You are a knowledge graph builder for educational content.

## Task
Extract specific, source-grounded concepts from the provided text chunks.

## Rules
- Extract concepts that are SPECIFIC to the source material — \
reference specific ideas, terms, methods, and frameworks from the source
- Concept names MUST reflect actual terms or ideas from the source material
- Each concept description MUST be detailed and grounded in the source, \
including concrete examples where relevant
- Identify prerequisite relationships between concepts \
(which concepts must be understood first)
- Note common misconceptions students might have
- Use slug IDs derived from the concept names (e.g. "spaced-repetition", "leitner-system")
- Do NOT extract overly generic concepts — every concept must be \
grounded in the specific material provided

## Good vs Bad Concepts
✅ GOOD (source-specific): concepts tied to specific ideas, methods, \
frameworks, or findings discussed in the source material
❌ BAD (generic topics): vague umbrella terms like "learning", "memory", \
"best practices" that could apply to any topic

## Extraction Strategy
1. First scan all chunks for key terms, methods, frameworks, and named ideas
2. Group related ideas into coherent concepts
3. For each concept, include specific details from the source — \
definitions, examples, key properties
4. In descriptions, reference specific details and explain their significance

## Output
Call the `add_concept` tool once for EACH concept you extract. \
Extract all relevant concepts."""

_GRAPH_REVIEW_SP = """\
You are reviewing a set of extracted concepts against \
the original source material.

## Task
Review the concepts below and correct any issues. Then output the complete \
corrected concept list.

## Review Checklist
- Are any prerequisite relationships missing or incorrect?
- Are any descriptions inaccurate or missing key details from the source?
- Are there redundant concepts that should be merged?
- Are there important concepts from the source that were missed?
- Does every concept have concrete examples from the source material?
- **Source specificity**: Are any concepts too generic? Every concept should be \
grounded in specific ideas, methods, or frameworks from the source. Replace any \
overly generic concept with more specific ones from the source material

## Existing Concepts
{concepts_section}

## Rules
- Call `add_concept` once for EACH concept in the final reviewed set
- Include corrected versions of existing concepts AND any new concepts found
- Remove truly redundant concepts by not including them
- Keep concepts specific to the source material — avoid overly generic terms"""

_QUESTION_SP = """\
You are generating an assessment question for a student.

## Domain
{domain_name}

## Target Concept
**{concept_name}** (difficulty: {difficulty}/5)

### Description
{concept_description}

## Rules
- ONLY test knowledge present in the concept description above — NEVER test facts \
not in the description
- Ground the question in the specific concept, not general knowledge
- Adjust complexity to match difficulty level (1=recognition, 5=transfer/novel application)
- Provide a correct_answer_summary for grading reference
{history_section}

## Output
Call the `generate_question` tool with your question."""

_EVALUATE_SP = """\
You are evaluating a student's answer to a knowledge question.

## Grading Rules
- **Generous partial credit**: If the answer shows partial understanding, mark it correct.
- **Brief correct answers are strong**: If the answer is factually correct and addresses \
the question, set reasoning_quality to "strong" — even if the answer is very short or terse.
- **Source-grounded only**: ONLY evaluate based on facts present in the CONCEPT DESCRIPTION \
below. Do NOT penalize the student for omitting information beyond the source material, \
and do NOT penalize for rejecting inferences that go beyond the source.
- **Flag only clear misconceptions**: Only set misconception_id if the student demonstrates \
a clear, unambiguous misunderstanding that matches a known misconception. Minor imprecisions \
or omissions are NOT misconceptions.

## Concept Description (source of truth)
{concept_description}
{misconceptions_section}"""


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------

def _make_collecting_concept_tool(collector: list):
    """Create a concept tool that appends results to `collector` list."""
    def add_concept(
        id: str,  # Slug ID, e.g. "itertools-chain"
        name: str,  # Display name, e.g. "itertools.chain"
        description: str,  # What it does, specific to source material. Include concrete examples where relevant
        prerequisites: list[str] = None,  # Slug IDs of prerequisite concepts
        misconceptions: list[str] = None,  # Common mistakes students make
        code_examples: list[str] = None,  # Optional examples from the source
    ) -> ConceptDef:
        """Add a concept to the knowledge graph. Call once per concept."""
        c = ConceptDef(
            id=id, name=name, description=description,
            prerequisites=prerequisites or [],
            misconceptions=misconceptions or [],
            code_examples=code_examples or [],
        )
        collector.append(c)
        return c
    return add_concept


def _make_question_tool():
    """Create structured output tool for question generation."""
    def generate_question(
        concept_id: str,  # Slug ID of the target concept
        question_text: str,  # The question text
        question_type: str = "recall",  # One of: recognition, recall, application, analysis, transfer
        difficulty_level: float = 3.0,  # 1.0 (easy) to 5.0 (hard)
        hints: list[str] = None,  # Optional progressive hints
        correct_answer_summary: str = "",  # Brief summary of the correct answer for grading
    ) -> GeneratedQuestion:
        """Generate an assessment question about a code concept."""
        return GeneratedQuestion(
            concept_id=concept_id, question_text=question_text,
            question_type=question_type, difficulty_level=difficulty_level,
            hints=hints or [], correct_answer_summary=correct_answer_summary,
        )
    return generate_question


def _make_eval_tool(difficulty: int = 3):
    """Create the structured output tool function for answer evaluation."""
    def assess_answer(
        correct: bool,  # Whether the answer is correct (be generous with partial credit)
        reasoning_quality: str,  # "strong" if answer shows good understanding, "weak" otherwise
        feedback: str,  # Brief, encouraging feedback for the student
        misconception_id: str = "",  # Only set for clear, unambiguous misunderstandings
    ) -> AssessmentResult:
        """Evaluate the student's answer and provide an assessment result."""
        return AssessmentResult(
            correct=correct,
            difficulty_level=difficulty,
            reasoning_quality=reasoning_quality,
            explanation=feedback,
            misconception_id=misconception_id or None,
        )
    return assess_answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_answer(text: str) -> str:
    """Truncate answer text to _MAX_ANSWER_TOKENS tokens."""
    tokens = _ENC.encode(text)
    if len(tokens) <= _MAX_ANSWER_TOKENS:
        return text
    return _ENC.decode(tokens[:_MAX_ANSWER_TOKENS])


def _format_history(history: list[RetrievalEvent], max_events: int = 5) -> str:
    """Format last N retrieval events for prompt context."""
    if not history:
        return "No prior assessment history."
    recent = history[-max_events:]
    lines = []
    for e in recent:
        status = "correct" if e.correct else "incorrect"
        line = f"- Level {e.difficulty_level}: {status} ({e.reasoning_quality})"
        if e.misconception_id:
            line += f" [misconception: {e.misconception_id}]"
        if e.question_type:
            line += f" [{e.question_type}]"
        lines.append(line)
    return "Recent history:\n" + "\n".join(lines)


def _format_misconceptions(concept) -> str:
    """Build the misconceptions section for the evaluation system prompt."""
    misconceptions = getattr(concept, 'known_misconceptions', None) or getattr(concept, 'misconceptions', None) or []
    if not misconceptions:
        return ""
    lines = "\n".join(f"- `{m}`" for m in misconceptions)
    return f"\n## Known Misconceptions (only flag if clearly demonstrated)\n{lines}\n"


def _extract_symbols(text: str) -> list[str]:
    """Extract likely code symbols (function/class/method names, parameters) from text."""
    symbols = []
    # Function/method definitions: def foo(bar, baz=None)
    for m in _re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)', text):
        sig = f"{m.group(1)}({m.group(2).strip()})"
        symbols.append(sig)
    # Class definitions: class Foo(Base)
    for m in _re.finditer(r'class\s+(\w+)\s*(?:\(([^)]*)\))?', text):
        cls = m.group(1)
        base = m.group(2)
        symbols.append(f"class {cls}({base.strip()})" if base else f"class {cls}")
    # Decorators: @decorator or @decorator(args)
    for m in _re.finditer(r'@(\w[\w.]*(?:\([^)]*\))?)', text):
        symbols.append(f"@{m.group(1)}")
    # Module-level assignments: FOO = ... or foo_bar = ...
    for m in _re.finditer(r'^([A-Z][A-Z_0-9]+)\s*=', text, _re.MULTILINE):
        symbols.append(m.group(1))
    return symbols


def _format_chunks(chunks: list[TextChunk]) -> str:
    """Format text chunks into a single user message, with extracted symbol summary."""
    parts = []
    all_symbols = []
    for i, chunk in enumerate(chunks):
        source_label = f" (source: {chunk.source_id})" if chunk.source_id else ""
        parts.append(f"## Chunk {i + 1}{source_label}\n{chunk.text}")
        all_symbols.extend(_extract_symbols(chunk.text))

    body = "\n\n".join(parts)

    # Add symbol summary if we found any, to help Claude focus on specifics
    if all_symbols:
        # Deduplicate preserving order
        seen = set()
        unique = []
        for s in all_symbols:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        symbol_list = "\n".join(f"- `{s}`" for s in unique[:50])
        body = (
            f"## Key Terms Found in Source\n"
            f"The following terms and symbols were detected in the source. "
            f"Use these to help identify concepts:\n{symbol_list}\n\n{body}"
        )

    return body


def _format_concepts_for_review(concepts: list[ConceptDef]) -> str:
    """Format existing concepts as text for the review prompt."""
    parts = []
    for c in concepts:
        prereqs = ", ".join(c.prerequisites) if c.prerequisites else "(none)"
        parts.append(
            f"### {c.name} (`{c.id}`)\n"
            f"- Prerequisites: {prereqs}\n"
            f"- Description: {c.description}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Multi-turn extraction helper
# ---------------------------------------------------------------------------

def _multi_turn_extract(client, user_content, system_prompt, max_steps=20):
    """Extract concepts using multi-turn Chat.toolloop.

    Falls back to single-shot client.structured if Chat creation fails
    (e.g. in tests with mock clients).
    """
    concepts = []
    collecting_fn = _make_collecting_concept_tool(concepts)

    try:
        # Production path: multi-turn extraction via Chat.toolloop
        chat = Chat(
            model=getattr(client, 'model', models[0]),
            sp=system_prompt,
            tools=[collecting_fn],
            cli=client.c,  # pass raw Anthropic SDK client, not claudette wrapper
        )
        responses = list(chat.toolloop(user_content, max_steps=max_steps))
        logger.info(f"[_multi_turn_extract] toolloop returned {len(responses)} responses, collected {len(concepts)} concepts")
        for i, r in enumerate(responses):
            logger.debug(f"[_multi_turn_extract] response[{i}]: type={type(r).__name__}, stop_reason={getattr(chat, 'c', None) and getattr(chat.c, 'stop_reason', '?')}")
    except Exception as e:
        # Fallback: single-shot structured (for tests with mock clients)
        import traceback as _tb
        err_detail = ''.join(_tb.format_exc())
        logger.warning(f"[_multi_turn_extract] Chat.toolloop failed ({e}), falling back to client.structured\n{err_detail}")
        # Write error to file for debugging
        try:
            Path("logs/toolloop_errors.log").parent.mkdir(parents=True, exist_ok=True)
            with open("logs/toolloop_errors.log", "a") as f:
                f.write(f"\n{'='*60}\n{time.strftime('%Y-%m-%dT%H:%M:%S')}\n{err_detail}\n")
        except Exception:
            pass
        msgs = [mk_msg(user_content)]
        structured_result = client.structured(
            msgs, tools=[collecting_fn], ns=mk_ns(collecting_fn), sp=system_prompt,
        )
        if not concepts and structured_result:
            concepts = [r for r in structured_result if isinstance(r, ConceptDef)]

    return concepts


# ---------------------------------------------------------------------------
# ClaudeCalls class — public API
# ---------------------------------------------------------------------------

class ClaudeCalls:
    """Structured Claude API wrapper for MasteryOS.

    Uses heavy model (Opus) for graph operations and fast model (Sonnet)
    for assessment operations.
    """

    def __init__(
        self,
        heavy_model: str | None = None,
        fast_model: str | None = None,
        capture_dir: str | None = None,
    ):
        self.heavy_model = heavy_model or models[0]
        self.fast_model = fast_model or models[1]
        self._heavy_client = Client(self.heavy_model, cache=True)
        self._fast_client = Client(self.fast_model)
        self._capture_dir = Path(capture_dir) if capture_dir else None
        if self._capture_dir:
            self._capture_dir.mkdir(parents=True, exist_ok=True)

    def _capture(self, method: str, inputs: dict, response):
        """Write an API response to a timestamped JSON file for mock validation."""
        if not self._capture_dir:
            return
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        path = self._capture_dir / f"{method}_{ts}.json"
        # Avoid collisions if multiple calls in same second
        n = 1
        while path.exists():
            path = self._capture_dir / f"{method}_{ts}_{n}.json"
            n += 1
        try:
            # Convert response to serializable form
            if hasattr(response, '__dataclass_fields__'):
                resp_data = asdict(response)
            elif isinstance(response, list):
                resp_data = [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in response]
            else:
                resp_data = str(response)
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "method": method,
                "inputs": inputs,
                "response": resp_data,
            }
            path.write_text(json.dumps(payload, indent=2, default=str))
            logger.debug(f"[capture] Wrote {path}")
        except Exception as e:
            logger.warning(f"[capture] Failed to write {path}: {e}")

    def build_graph(self, chunks: list[TextChunk]) -> GraphBuilderOutput:
        """Extract a knowledge graph from source chunks.

        Uses multi-turn Chat.toolloop for thorough extraction, falling back
        to single-shot structured() if toolloop is unavailable.

        Args:
            chunks: List of TextChunk objects from ingestion.

        Returns:
            GraphBuilderOutput with extracted concepts.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("Cannot build graph from empty chunks list")

        t0 = time.perf_counter()
        word_count = sum(len(c.text.split()) for c in chunks)
        logger.info(f"[build_graph] Extracting from {len(chunks)} chunks (~{word_count} words)")

        user_content = _format_chunks(chunks)
        concepts = _multi_turn_extract(self._heavy_client, user_content, _GRAPH_BUILD_SP)

        elapsed = time.perf_counter() - t0
        concept_names = ", ".join(c.name for c in concepts[:5])
        if len(concepts) > 5:
            concept_names += f", ... (+{len(concepts) - 5} more)"
        logger.info(
            f"[build_graph] Extracted {len(concepts)} concepts: {concept_names} "
            f"time={elapsed:.2f}s"
        )

        result = GraphBuilderOutput(concepts=concepts)
        self._capture("build_graph", {"chunk_count": len(chunks), "word_count": word_count}, result)
        return result

    def review_graph(
        self, graph_data: list[dict], chunks: list[TextChunk]
    ) -> GraphBuilderOutput:
        """Review and improve an existing graph against source material.

        Uses multi-turn Chat.toolloop for thorough review, falling back
        to single-shot structured() if toolloop is unavailable.

        Args:
            graph_data: List of concept dicts from the current graph.
            chunks: Original source chunks for reference.

        Returns:
            GraphBuilderOutput with improved concepts.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("Cannot review graph with empty chunks list")

        t0 = time.perf_counter()
        logger.info(f"[review_graph] Reviewing {len(graph_data)} concepts against {len(chunks)} chunks")

        # Convert graph_data dicts back to ConceptDef for formatting
        existing_concepts = []
        for d in graph_data:
            try:
                existing_concepts.append(ConceptDef(
                    id=d.get("id", ""), name=d.get("name", ""),
                    description=d.get("description", ""),
                    prerequisites=d.get("prerequisites", []),
                    misconceptions=d.get("misconceptions", []),
                    code_examples=d.get("code_examples", []),
                ))
            except Exception:
                pass

        sp = _GRAPH_REVIEW_SP.format(
            concepts_section=_format_concepts_for_review(existing_concepts),
        )

        user_content = (
            "Review the concepts listed in your instructions against these source chunks. "
            "Output the complete corrected concept list.\n\n"
            + _format_chunks(chunks)
        )

        concepts = _multi_turn_extract(self._heavy_client, user_content, sp)

        elapsed = time.perf_counter() - t0
        concept_names = ", ".join(c.name for c in concepts[:5])
        if len(concepts) > 5:
            concept_names += f", ... (+{len(concepts) - 5} more)"
        logger.info(
            f"[review_graph] Reviewed → {len(concepts)} concepts: {concept_names} "
            f"time={elapsed:.2f}s"
        )

        result = GraphBuilderOutput(concepts=concepts)
        self._capture("review_graph", {
            "existing_concept_count": len(graph_data),
            "chunk_count": len(chunks),
        }, result)
        return result

    def generate_question(
        self,
        concept_name: str,
        concept_description: str,
        concept_id: str,
        difficulty: int,
        history: list[RetrievalEvent] | None = None,
        domain_name: str = "",
    ) -> GeneratedQuestion:
        """Generate an assessment question for a concept.

        Args:
            concept_name: Human-readable concept name.
            concept_description: Concept description.
            concept_id: Concept slug id.
            difficulty: Target difficulty level (1-5).
            history: Optional retrieval history for context.
            domain_name: Name of the domain for context.

        Returns:
            GeneratedQuestion with question text and rubric.
        """
        t0 = time.perf_counter()

        # Build history section for prompt
        history_section = ""
        if history:
            history_section = (
                "\n## Previous Questions (avoid repeating)\n"
                + "\n".join(f"- {getattr(e, 'question_text', str(e))[:100]}" for e in (history or [])[-5:])
            )

        sp = _QUESTION_SP.format(
            domain_name=domain_name or "General",
            concept_name=concept_name,
            difficulty=difficulty,
            concept_description=concept_description,
            history_section=history_section,
        )

        logger.info(
            f"[generate_question] concept={concept_id} difficulty={difficulty} "
            f"domain={domain_name!r}"
        )

        msgs = [mk_msg("Generate a question for this concept.")]
        tool_fn = _make_question_tool()

        results = self._fast_client.structured(
            msgs, tools=[tool_fn], ns=mk_ns(tool_fn), sp=sp, temp=0.7,
        )

        elapsed = time.perf_counter() - t0

        capture_inputs = {
            "concept_id": concept_id, "concept_name": concept_name,
            "difficulty": difficulty, "domain_name": domain_name,
        }

        if results:
            question = results[0]
            logger.info(
                f"[generate_question] concept={concept_id} difficulty={difficulty} "
                f"type={question.question_type} "
                f"q={question.question_text[:80]!r} time={elapsed:.2f}s"
            )
            self._capture("generate_question", capture_inputs, question)
            return question

        # Fallback if Claude returned no tool call
        logger.warning(f"[generate_question] No tool call returned, defaulting. time={elapsed:.2f}s")
        fallback = GeneratedQuestion(
            concept_id=concept_id, difficulty_level=difficulty,
            question_type="recall", question_text="Explain this concept.", rubric=""
        )
        self._capture("generate_question", capture_inputs, fallback)
        return fallback

    def evaluate_answer(
        self,
        concept_name: str,
        concept_description: str,
        concept_id: str,
        question_text: str,
        answer_text: str,
        difficulty: int,
        history: list[RetrievalEvent] | None = None,
    ) -> AssessmentResult:
        """Evaluate a student's answer to an assessment question.

        Args:
            concept_name: Human-readable concept name.
            concept_description: Concept description.
            concept_id: Concept slug id.
            question_text: The question that was asked.
            answer_text: The student's answer (will be truncated if too long).
            difficulty: The difficulty level of the question.
            history: Optional retrieval history for context.

        Returns:
            AssessmentResult with evaluation details.
        """
        t0 = time.perf_counter()
        answer_text = _truncate_answer(answer_text)
        history_str = _format_history(history or [])

        # Build system prompt with concept description as grading source
        # Create a simple namespace to pass to _format_misconceptions
        class _Concept:
            pass
        _c = _Concept()
        _c.known_misconceptions = []
        _c.misconceptions = []

        sp = _EVALUATE_SP.format(
            concept_description=concept_description,
            misconceptions_section="",
        )

        # Build user message: question + student answer + history
        user_content = (
            f"## Question (Difficulty {difficulty}/5)\n{question_text}\n\n"
            f"## Student's Answer\n{answer_text}\n\n"
            f"## Student History\n{history_str}"
        )

        logger.info(
            f"[evaluate_answer] concept={concept_id} "
            f"question={question_text[:60]!r} answer_len={len(answer_text)}"
        )

        msgs = [mk_msg(user_content)]
        tool_fn = _make_eval_tool(difficulty=difficulty)

        results = self._fast_client.structured(
            msgs, tools=[tool_fn], ns=mk_ns(tool_fn), sp=sp,
        )

        elapsed = time.perf_counter() - t0

        capture_inputs = {
            "concept_id": concept_id, "concept_name": concept_name,
            "difficulty": difficulty, "question_text": question_text[:200],
            "answer_text": answer_text[:200],
        }

        if results:
            result = results[0]
            logger.info(
                f"[evaluate_answer] correct={result.correct} "
                f"reasoning={result.reasoning_quality} "
                f"feedback={getattr(result, 'feedback', '')[:60]!r} "
                f"misconception={result.misconception_id!r} time={elapsed:.2f}s"
            )
            self._capture("evaluate_answer", capture_inputs, result)
            return result

        # Fallback if Claude returned no tool call
        logger.warning(f"[evaluate_answer] No tool call returned, defaulting. time={elapsed:.2f}s")
        fallback = AssessmentResult(
            correct=False, difficulty_level=difficulty,
            reasoning_quality="weak", explanation="Evaluation failed."
        )
        self._capture("evaluate_answer", capture_inputs, fallback)
        return fallback

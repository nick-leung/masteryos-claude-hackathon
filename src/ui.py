"""FastHTML + MonsterUI web interface for MasteryOS.

Single-user application with four pages: Home (ingest), Graph view,
Assessment flow, and Coverage dashboard. Uses HTMX for real-time updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from fasthtml.common import *
from monsterui.all import *
from starlette.datastructures import UploadFile

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from fastcore.parallel import threaded

from src.assessment import SessionOrchestrator
from src.claude_calls import ClaudeCalls
from src.mock_claude import MockClaudeCalls
from src.coverage import CoverageCalculator
from src.graph_builder import build as build_graph, _merge_concepts
from src.graph_store import KnowledgeGraph
from src.inference import PrerequisiteInference
from src.ingestion import ingest
from src.models import (
    AssessmentResult, Concept, ConceptDef, CoverageReport,
    GeneratedQuestion, GraphBuilderOutput, Misconception, _sanitize_slug,
)
from src.scheduler import FSRSScheduler
from src.viz import render_graph

# ---------------------------------------------------------------------------
# Demo data path
# ---------------------------------------------------------------------------
_DEMO_PATH = Path(__file__).parent / "demo_data" / "demo_graph.json"


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    """Single-user application state."""
    graph: KnowledgeGraph | None = None
    scheduler: FSRSScheduler | None = None
    coverage: CoverageCalculator | None = None
    inference: PrerequisiteInference | None = None
    orchestrator: SessionOrchestrator | None = None
    claude: ClaudeCalls | None = None
    domain_name: str = ""
    building: bool = False
    build_error: str = ""
    pending_question: GeneratedQuestion | None = None


STATE = AppState()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app, rt = fast_app(
    hdrs=Theme.blue.headers(highlightjs=True, daisy=True),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page(*content, title: str = "MasteryOS"):
    """Wrap content in a standard page layout with nav."""
    nav = NavBar(
        A("Home", href=index),
        A("Graph", href=graph),
        A("Assess", href=assess),
        A("Coverage", href=coverage),
        brand=A(H3("MasteryOS"), href=index),
        sticky=True,
    )
    return Title(title), Container(nav, *content, cls=('mt-5', ContainerT.xl, 'space-y-4'))


def _no_graph_redirect():
    """Return a redirect response if no graph is built."""
    if STATE.graph is None:
        return RedirectResponse("/", status_code=303)
    return None


def _metric_card(label: str, value: float, icon_name: str):
    """Single metric as a Card with icon, matching dashboard InfoCard pattern."""
    pct = int(value * 100)
    return Card(
        DivFullySpaced(
            Span(f"{pct}%", cls=TextPresets.bold_lg),
            UkIcon(icon_name, height=20, width=20),
        ),
        Progress(value=str(pct), max="100"),
        header=H4(label),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@rt
def index():
    """Home page — source input form."""
    tagline = P("Turn any learning material into an adaptive quiz.", cls=TextT.lead)

    # --- Post-build: shift to next steps ---
    if STATE.graph is not None and not STATE.building:
        n = len(STATE.graph)
        return _page(
            tagline,
            Alert(f"✅ {n} concepts built for '{STATE.domain_name}'", cls=AlertT.success),
            DivLAligned(
                A("View Graph", href=graph, cls=('uk-btn', ButtonT.primary)),
                A("Start Assessment", href=assess, cls=('uk-btn', ButtonT.primary)),
                A("Coverage", href=coverage, cls=('uk-btn', ButtonT.secondary)),
            ),
            DividerSplit(),
            _build_form(collapsed=True),
            title="MasteryOS",
        )

    # --- Building state ---
    status = ""
    if STATE.building:
        status = _building_status()
    elif STATE.build_error:
        status = Alert(f"❌ {STATE.build_error}", cls=AlertT.error)

    # --- Demo link ---
    demo_form = ""
    if _DEMO_PATH.exists() and STATE.graph is None:
        demo_form = Form(
            Button("Try with sample data", type="submit",
                   cls=ButtonT.secondary),
            method="post", action=demo,
        )

    return _page(tagline, status, _build_form(), demo_form, title="MasteryOS")


def _building_status():
    """Shared building-in-progress fragment with spinner and HTMX polling."""
    return Div(
        DivLAligned(
            Loading(cls=(LoadingT.spinner, LoadingT.md)),
            P("Building… this may take 30–60 seconds.", cls=TextT.warning),
        ),
        id="build-status",
        hx_get=build_status, hx_trigger="every 5s", hx_swap="outerHTML",
    )



@rt
def build_status():
    """HTMX polling endpoint — returns updated status fragment."""
    if STATE.building: return _building_status()
    # Build finished — redirect to graph view on success, home on error
    dest = "/graph" if STATE.graph is not None and not STATE.build_error else "/"
    return HtmxResponseHeaders(redirect=dest)



def _build_form(collapsed=False):
    """Render the source input form."""
    if collapsed:
        return Div(
            Form(
                LabelInput("Topic", id="domain_name", name="domain_name",
                           placeholder="Topic (auto-detected if blank)"),
                LabelInput("URL", id="source_url", name="source_url",
                           placeholder="Paste a URL…"),
                Button("Build", type="submit", cls=(ButtonT.secondary, "mt-2")),
                method="post", action=build,
                cls="space-y-2",
            ),
        )

    return Form(
        LabelInput("Source URL", id="source_url", name="source_url",
                   placeholder="Paste a URL (web page, Wikipedia, YouTube…)",
                   ),
        Details(
            Summary("Or paste text / upload PDF", cls=TextPresets.muted_sm,
                    style="cursor:pointer"),
            Div(
                LabelTextArea("Text", id="source_text", name="source_text", rows="4",
                              placeholder="Paste text here…"),
                Upload("Choose PDF", id="pdf_file", name="pdf_file", accept=".pdf"),
                cls="space-y-2 mt-2",
            ),
        ),
        LabelInput("Topic name", id="domain_name", name="domain_name",
                   placeholder="Topic name (auto-detected if blank)",
                   ),
        LoaderButton("Build Knowledge Graph", type="submit", cls=ButtonT.primary),
        method="post", action=build,
        cls="space-y-3",
    )


@threaded
def _run_build(source: str, domain: str):
    """Run the blocking ingest + graph-build pipeline in a background thread."""
    try:
        chunks = ingest(source)
        if not chunks:
            STATE.build_error = "No text could be extracted from the source."
            STATE.building = False
            return

        claude = ClaudeCalls(heavy_model="claude-haiku-4-5", fast_model="claude-haiku-4-5", capture_dir="logs/api_responses")
        graph = build_graph(chunks, domain, claude)

        # Initialize all components
        STATE.graph = graph
        STATE.claude = claude
        STATE.scheduler = FSRSScheduler(graph)
        STATE.coverage = CoverageCalculator(graph, STATE.scheduler)
        STATE.inference = PrerequisiteInference(graph)

    except Exception as e:
        STATE.build_error = str(e)
    finally:
        STATE.building = False


@rt(methods=['post'])
async def build(domain_name: str = "", source_url: str = "", source_text: str = "", pdf_file: UploadFile | None = None):
    """Ingest source and build the knowledge graph."""
    if STATE.building:
        return RedirectResponse("/", status_code=303)

    STATE.building = True
    STATE.build_error = ""

    # Determine source (PDF read must happen here — it's async)
    source = ""
    if pdf_file is not None and pdf_file.filename:
        content = await pdf_file.read()
        if content:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(content)
            tmp.close()
            source = tmp.name

    if not source and source_url.strip():
        source = source_url.strip()

    if not source and source_text.strip():
        source = source_text.strip()

    if not source:
        STATE.build_error = "No source provided — enter a URL, paste text, or upload a PDF."
        STATE.building = False
        return RedirectResponse("/", status_code=303)

    domain = domain_name.strip()
    if not domain:
        # Auto-derive from source
        if source_url.strip():
            domain = urlparse(source_url.strip()).netloc.removeprefix("www.")
        elif pdf_file and pdf_file.filename:
            domain = Path(pdf_file.filename).stem.replace("_", " ").replace("-", " ").title()
        elif source_text.strip():
            words = source_text.strip().split()[:5]
            domain = " ".join(words).rstrip(".,;:!?")
            if len(domain) > 40: domain = domain[:40] + "…"
        else:
            domain = "Untitled"
    STATE.domain_name = domain

    # Launch blocking work in background thread — returns immediately
    _run_build(source, domain)

    return RedirectResponse("/", status_code=303)


@rt(methods=['post'])
def demo():
    """Load a demo knowledge graph from captured API data — no API key needed."""
    if not _DEMO_PATH.exists():
        STATE.build_error = "Demo data not found."
        return RedirectResponse("/", status_code=303)

    try:
        with open(_DEMO_PATH) as f:
            data = json.load(f)

        raw_concepts = data["response"]["concepts"]
        concepts = []
        for c in raw_concepts:
            # Coerce misconceptions: may be dicts or strings
            miscon = c.get("misconceptions", [])
            if isinstance(miscon, list):
                miscon = [m if isinstance(m, str) else str(m) for m in miscon]
            else:
                miscon = [str(miscon)] if miscon else []
            concepts.append(ConceptDef(
                id=c["id"],
                name=c["name"],
                description=c.get("description", ""),
                prerequisites=c.get("prerequisites", []),
                misconceptions=miscon,
                code_examples=c.get("code_examples", []),
            ))

        graph_output = GraphBuilderOutput(concepts=concepts)
        graph = build_graph.__wrapped__(graph_output) if hasattr(build_graph, '__wrapped__') else None

        # Build graph using the same pipeline as /build
        merged = _merge_concepts(graph_output, GraphBuilderOutput(concepts=[]))

        domain = "Spaced Repetition (Demo)"
        kg = KnowledgeGraph(domain)
        for cdef in merged:
            misconceptions = [
                Misconception(
                    id=_sanitize_slug(m) if m.strip() else "unnamed",
                    description=m,
                    concept_id=cdef.id,
                )
                for m in cdef.misconceptions
            ]
            concept = Concept(
                id=cdef.id, name=cdef.name, description=cdef.description,
                prerequisites=list(cdef.prerequisites),
                known_misconceptions=misconceptions,
            )
            try:
                kg.add_concept(concept)
            except ValueError:
                # Drop prereqs until it works
                prereqs = list(concept.prerequisites)
                while prereqs:
                    prereqs.pop()
                    try:
                        kg.add_concept(Concept(
                            id=cdef.id, name=cdef.name, description=cdef.description,
                            prerequisites=prereqs, known_misconceptions=misconceptions,
                        ))
                        break
                    except ValueError:
                        continue
                else:
                    try:
                        kg.add_concept(Concept(
                            id=cdef.id, name=cdef.name, description=cdef.description,
                            prerequisites=[], known_misconceptions=misconceptions,
                        ))
                    except ValueError:
                        pass

        STATE.graph = kg
        STATE.claude = MockClaudeCalls()
        STATE.scheduler = FSRSScheduler(kg)
        STATE.coverage = CoverageCalculator(kg, STATE.scheduler)
        STATE.inference = PrerequisiteInference(kg)
        STATE.domain_name = domain

    except Exception as e:
        STATE.build_error = f"Failed to load demo: {e}"
        return RedirectResponse("/", status_code=303)

    return RedirectResponse("/graph", status_code=303)


@rt
def graph():
    """Graph view — Graphviz DAG as inline SVG with stats."""
    redir = _no_graph_redirect()
    if redir:
        return redir

    graph = STATE.graph
    svg = render_graph(graph, STATE.coverage)

    return _page(
        Card(
            NotStr(svg),
            header=(H3(f"Knowledge Graph — {STATE.domain_name}"), Subtitle(f"{len(graph)} concepts")),
            cls="overflow-auto",
        ),
        title="MasteryOS — Graph",
    )


@rt
def assess():
    """Start an assessment session — show first question."""
    redir = _no_graph_redirect()
    if redir:
        return redir

    # Ensure all required components exist (may be missing after server restart)
    if STATE.inference is None or STATE.claude is None:
        return RedirectResponse("/", status_code=303)

    STATE.orchestrator = SessionOrchestrator(
        graph=STATE.graph,
        claude=STATE.claude,
        scheduler=STATE.scheduler,
        inference=STATE.inference,
        coverage=STATE.coverage,
    )

    try:
        question = STATE.orchestrator.start_session()
    except Exception as e:
        STATE.orchestrator = None
        return _page(
            Alert(f"Could not start the assessment: {e}", cls=AlertT.error),
            A("Back to Home", href=index, cls=('uk-btn', ButtonT.primary)),
            title="MasteryOS — Error",
        )

    if question is None:
        return _page(
            Alert("All concepts mastered! No assessment needed — you've covered everything.", cls=AlertT.success),
            A("View Coverage", href=coverage, cls=('uk-btn', ButtonT.primary)),
            title="MasteryOS — Assessment",
        )

    STATE.pending_question = question
    return _page(_question_card(question), title="MasteryOS — Assessment")


def _step_progress():
    """Render a simple question counter with progress bar."""
    if not (STATE.orchestrator and STATE.orchestrator._state):
        return ""
    s = STATE.orchestrator._state
    n, total = s.question_count, s.max_questions
    return Div(
        DivFullySpaced(
            P(f"Question {n} of {total}", cls=TextPresets.bold_sm),
            P(f"{int(n / total * 100)}%", cls=TextPresets.muted_sm),
        ),
        Progress(value=str(n), max=str(total)),
    )


def _question_card(q: GeneratedQuestion):
    """Render a question card with answer form."""
    concept = STATE.graph.get(q.concept_id) if STATE.graph else None
    concept_name = concept.name if concept else q.concept_id

    return Card(
        _step_progress(),
        render_md(q.question_text),
        Form(
            LabelTextArea("Your answer", id="answer", placeholder="Type your answer…",
                          rows="4"),
            LoaderButton("Submit", type="submit", cls=ButtonT.primary),
            hx_post=answer, hx_target="#assess-content", cls="space-y-3",
        ),
        header=CardTitle(concept_name),
        footer=Form(
            Button("End Session", type="submit", cls=ButtonT.link),
            method="post", action=end_session,
        ),
        id="assess-content",
    )


def _result_card(result: AssessmentResult, next_q: GeneratedQuestion | None):
    """Render an assessment result card."""
    explanation = result.explanation.strip() if result.explanation else ""

    if result.correct:
        result_alert = Alert("Correct!", cls=AlertT.success)
    else:
        result_alert = Alert("Incorrect", cls=AlertT.error)

    content = [result_alert]

    if STATE.pending_question:
        content.append(Blockquote(STATE.pending_question.question_text[:300]))

    if explanation:
        content.append(P(explanation))

    if next_q is not None:
        footer = DivLAligned(
            Form(
                Button("Next Question →", type="submit", cls=ButtonT.primary),
                method="post", action=next_question,
            ),
            Form(
                Button("End Session", type="submit", cls=ButtonT.link),
                method="post", action=end_session,
            ),
            cls="space-x-2",
        )
    else:
        content.append(Alert("Session complete!", cls=AlertT.info))
        footer = A("View Coverage", href=coverage, cls=('uk-btn', ButtonT.primary))

    return Card(*content, header=CardTitle("Result"), footer=footer, id="assess-content")
@rt(methods=['post'])
def answer(answer: str = "", request=None):
    """Submit an answer and show the result."""
    if STATE.orchestrator is None:
        return RedirectResponse("/assess", status_code=303)

    STATE._last_answer = answer
    result, next_q = STATE.orchestrator.submit_answer(answer)
    STATE.pending_question = next_q

    # HTMX partial swap — just return the card
    if request and request.headers.get("hx-request"):
        return _result_card(result, next_q)
    return _page(_result_card(result, next_q), title="MasteryOS — Result")


@rt('/next', methods=['post'])
def next_question():
    """Show the next cached question."""
    if STATE.pending_question is None:
        return RedirectResponse("/assess", status_code=303)

    q = STATE.pending_question
    return _page(_question_card(q), title="MasteryOS — Assessment")


@rt(methods=['post'])
def end_session():
    """End the current session and redirect to coverage."""
    if STATE.orchestrator is not None:
        try:
            STATE.orchestrator.end_session()
        except RuntimeError:
            pass  # Session may not be active
    return RedirectResponse("/coverage", status_code=303)


@rt
def coverage():
    """Coverage dashboard — clean, actionable overview."""
    redir = _no_graph_redirect()
    if redir:
        return redir

    report = STATE.coverage.generate_report()

    # Zero-state message
    if report.assessed_count == 0:
        return _page(
            Card(
                Alert("No assessments yet — start one to track your progress.",
                      cls=AlertT.info),
                DivCentered(
                    A("Start Assessment", href=assess, cls=('uk-btn', ButtonT.primary)),
                    cls="pt-4",
                ),
            ),
            title="MasteryOS — Coverage",
        )

    return _page(
        Card(
            H2(f"{report.overall_score:.0%}"),
            P("Overall mastery score", cls=TextT.muted),
            header=H3(f"Coverage — {STATE.domain_name}"),
        ),
        Grid(
            _metric_card("How much covered",       report.breadth_percent, "book-open"),
            _metric_card("How well understood",     report.depth_score,    "brain"),
            _metric_card("How recently practised",  report.recency_score,  "clock"),
            cols_min=1, cols_max=3,
        ),
        Card(
            DivCentered(A("Practice Weak Concepts", href=assess, cls=('uk-btn', ButtonT.primary))),
        ) if report.overall_score < 1.0 else None,
        title="MasteryOS — Coverage",
    )

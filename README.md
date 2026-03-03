# MasteryOS

Turn any learning material into an adaptive quiz. Paste a URL, text, or upload a PDF — MasteryOS builds a knowledge graph of concepts and quizzes you with spaced-repetition scheduling until you've mastered the material.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

## Features

- **Multi-format ingestion** — web URLs, YouTube transcripts, PDFs, or raw text
- **AI-powered knowledge graphs** — Claude extracts concepts, prerequisites, and misconceptions in a two-pass pipeline
- **Adaptive assessment** — questions ramp in difficulty based on your performance
- **FSRS spaced repetition** — scientifically-backed scheduling decides when to re-test each concept
- **Prerequisite inference** — mastery signals propagate through the graph so the system knows what you know (and what's shaky)
- **Coverage dashboard** — five-level mastery tracking with hollow-concept detection
- **Interactive graph visualization** — color-coded SVG of your knowledge graph
- **Demo mode** — try it instantly with a pre-built graph, no API key needed

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

You also need [Graphviz](https://graphviz.org/download/) installed on your system for graph rendering, and optionally [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube transcript ingestion:

```bash
# macOS
brew install graphviz yt-dlp

# Ubuntu / Debian
sudo apt-get install graphviz
pip install yt-dlp

# Windows
choco install graphviz yt-dlp
```

### 2. Run

```bash
python -m src
```

Open [http://localhost:8000](http://localhost:8000).

### 3. Try demo mode

Click **📦 Load Demo Graph** on the home page to explore MasteryOS with a pre-built knowledge graph on *Spaced Repetition*. No API key required — all data is local.

### 4. Use with your own content

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Then paste a URL, enter text, or upload a PDF on the home page and click **Build Knowledge Graph**. MasteryOS will extract concepts, build a prerequisite DAG, and start quizzing you.

## How it works

```
Source material
    ↓  ingestion.py — URL/PDF/text → chunked text
    ↓  claude_calls.py — two-pass Claude extraction
    ↓  graph_builder.py — merge + validate → DAG
    ↓
KnowledgeGraph (NetworkX DAG)
    ↓
    ├─ assessment.py — adaptive question generation + evaluation
    ├─ scheduler.py — FSRS spaced-repetition scheduling
    ├─ inference.py — prerequisite mastery propagation
    ├─ coverage.py — five-level mastery + hollow detection
    └─ viz.py — color-coded SVG graph rendering
    ↓
FastHTML + MonsterUI web interface (ui.py)
```

## Architecture

| Module | Role |
|---|---|
| `models.py` | Dataclasses for concepts, assessments, coverage reports, and JSON persistence |
| `ingestion.py` | URL, YouTube, PDF, and text → `TextChunk` pipeline |
| `claude_calls.py` | Structured Claude API calls via claudette (graph building, question generation, answer evaluation) |
| `graph_builder.py` | Two-pass extraction → validated `KnowledgeGraph` DAG |
| `graph_store.py` | NetworkX DAG wrapper with CRUD, traversal, and JSON serialization |
| `scheduler.py` | FSRS-4.5 spaced-repetition scheduler |
| `inference.py` | Prerequisite mastery propagation and next-concept selection |
| `coverage.py` | Five-level coverage calculator with hollow-concept detection |
| `viz.py` | Graphviz SVG rendering with mastery color coding |
| `mock_claude.py` | Drop-in mock for demo mode using captured API responses |
| `ui.py` | FastHTML + MonsterUI web app (home, graph, assessment, dashboard) |

## Tech stack

- **[FastHTML](https://github.com/AnswerDotAI/fasthtml)** + **[MonsterUI](https://github.com/AnswerDotAI/MonsterUI)** — web framework
- **[claudette](https://github.com/AnswerDotAI/claudette)** — Claude API client with structured output
- **[NetworkX](https://networkx.org/)** — knowledge graph storage and traversal
- **[Graphviz](https://graphviz.org/)** — graph visualization
- **[tiktoken](https://github.com/openai/tiktoken)** — token counting for chunk sizing
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — PDF text extraction

## License

MIT

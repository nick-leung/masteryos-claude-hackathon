"""Source material → TextChunk pipeline.

Accepts YouTube URLs, PDFs, web URLs, and raw text.
Chunks text using tiktoken for downstream Claude processing.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import httpx
import tiktoken
from bs4 import BeautifulSoup
import html2text

from src.models import TextChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_YOUTUBE_RE = re.compile(
    r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+',
)
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
_TARGET_TOKENS = 4000
_OVERLAP_TOKENS = 200
_ENCODING_NAME = "cl100k_base"

# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------


def detect_source_type(source: str) -> str:
    """Auto-detect source type from input string.

    Returns one of: 'youtube', 'pdf', 'url', 'text'.
    """
    s = source.strip()
    if _YOUTUBE_RE.match(s):
        return "youtube"
    if s.lower().endswith(".pdf"):
        return "pdf"
    if re.match(r'https?://', s, re.IGNORECASE):
        return "url"
    return "text"


# ---------------------------------------------------------------------------
# Source handlers
# ---------------------------------------------------------------------------


def _extract_youtube(url: str) -> str:
    """Extract subtitle text from a YouTube video via yt-dlp."""
    

    with tempfile.TemporaryDirectory() as tmpdir:
        out_tpl = os.path.join(tmpdir, "subs")
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--output", out_tpl,
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

        # Find the VTT file
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            raise RuntimeError("No English subtitles found for this video")

        vtt_text = vtt_files[0].read_text(encoding="utf-8")
    return _parse_vtt(vtt_text)


def _parse_vtt(vtt_text: str) -> str:
    """Parse VTT subtitle text: strip headers, timestamps, tags; deduplicate lines."""
    lines: list[str] = []
    seen: set[str] = set()

    for raw_line in vtt_text.splitlines():
        line = raw_line.strip()
        # Skip empty, WEBVTT header, NOTE lines, and timestamp lines
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if line.startswith("NOTE"):
            continue
        if "-->" in line:
            continue
        # Strip position/alignment metadata lines (e.g. "align:start position:0%")
        if re.match(r'^(align|position|line|size)\b', line):
            continue
        # Strip VTT tags like <c>, </c>, <00:00:01.234>, etc.
        line = re.sub(r'<[^>]+>', '', line).strip()
        if not line:
            continue
        # Deduplicate
        if line not in seen:
            seen.add(line)
            lines.append(line)

    text = " ".join(lines)
    return text.strip()


def _extract_pdf(path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(path)
    pages: list[str] = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
    doc.close()

    if not pages:
        raise RuntimeError("PDF contains no extractable text (all pages may be image-only)")
    return "\n\n".join(pages)


def _extract_web(url: str) -> str:
    """Extract readable text from a web URL."""
    resp = httpx.get(url, headers={"User-Agent": _USER_AGENT}, follow_redirects=True, timeout=30)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")

    # Plain text / markdown — return directly (e.g. Jina Reader, raw markdown URLs)
    if "text/plain" in content_type or "text/markdown" in content_type:
        return resp.text.strip()

    if "text/html" not in content_type and "application/xhtml" not in content_type:
        raise RuntimeError(f"Non-HTML content type: {content_type}")

    soup = BeautifulSoup(resp.text, "lxml")
    # Strip non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0  # no wrapping
    text = h.handle(str(soup))
    return text.strip()


def _extract_text(raw: str) -> str:
    """Passthrough for raw text — just strip whitespace."""
    text = raw.strip()
    if not text:
        raise RuntimeError("Empty input text")
    return text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    source_type: str = "",
    source_id: str = "",
    target_tokens: int = _TARGET_TOKENS,
    overlap_tokens: int = _OVERLAP_TOKENS,
) -> list[TextChunk]:
    """Split text into overlapping chunks by token count.

    Returns a list of TextChunk with source metadata populated.
    """
    enc = tiktoken.get_encoding(_ENCODING_NAME)
    tokens = enc.encode(text)

    if len(tokens) <= target_tokens:
        return [TextChunk(
            text=text,
            source_type=source_type,
            source_id=source_id,
            chunk_index=0,
            total_chunks=1,
        )]

    chunks: list[TextChunk] = []
    start = 0
    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        chunks.append(TextChunk(
            text=chunk_text_str,
            source_type=source_type,
            source_id=source_id,
            chunk_index=len(chunks),
            total_chunks=0,  # filled in below
        ))
        if end >= len(tokens):
            break
        start = end - overlap_tokens

    # Set total_chunks
    for c in chunks:
        c.total_chunks = len(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest(source: str) -> list[TextChunk]:
    """Ingest source material and return a list of TextChunks.

    Auto-detects source type (youtube, pdf, url, text) and delegates
    to the appropriate handler, then chunks the result.
    """
    source_type = detect_source_type(source)
    source_id = source.strip()

    if source_type == "youtube":
        text = _extract_youtube(source_id)
    elif source_type == "pdf":
        text = _extract_pdf(source_id)
    elif source_type == "url":
        text = _extract_web(source_id)
    else:
        text = _extract_text(source_id)

    return chunk_text(text, source_type=source_type, source_id=source_id)

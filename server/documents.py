"""Whole-document translation, preserving whatever structure the format has.

The brief names businesses, schools and government departments. Those people
translate documents, not single sentences, so the app handles files directly:

    .txt .md    line by line, blank lines and indentation kept
    .srt .vtt   subtitle text only -- timestamps and cue numbers are never
                sent to a translation engine, let alone translated
    .docx       paragraph by paragraph, including table cells, written back
                into the original document so styling and layout survive
    .pdf        text extracted per page and returned as .docx or .txt

The PDF asymmetry is deliberate and worth stating plainly: rewriting a PDF in
place would mean re-flowing text into fixed-position boxes in a script whose
glyphs are a different width, which produces overlapping, clipped output. It
is more honest to hand back an editable document than a broken-looking PDF.

Two things make this cheap rather than slow: identical segments are collapsed
before translating (subtitles repeat constantly), and the router's cache means
a re-run of the same file costs almost nothing.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

#: Extensions we accept, mapped to the format handler name.
SUPPORTED = {
    ".txt": "text", ".md": "text", ".csv": "text", ".log": "text",
    ".srt": "srt", ".vtt": "vtt",
    ".docx": "docx",
    ".pdf": "pdf",
}

#: What a translated file of each type comes back as.
OUTPUT_FORMAT = {
    "text": "same", "srt": "same", "vtt": "same", "docx": "same",
    "pdf": "docx",
}

_MAX_SEGMENTS = 4000


@dataclass
class Document:
    """Segments pulled out of a file, plus how to put them back."""
    segments: list[str]
    rebuild: Callable[[list[str]], bytes]
    media_type: str
    suffix: str
    kind: str
    note: str | None = None
    meta: dict = field(default_factory=dict)


class UnsupportedDocument(ValueError):
    pass


# --- plain text and markdown ----------------------------------------------
def _load_text(data: bytes, suffix: str) -> Document:
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # Blank lines and lines with no letters (rules, numbering) are structure,
    # not content, so they are preserved verbatim and never translated.
    index = [i for i, line in enumerate(lines) if line.strip() and
             any(ch.isalpha() for ch in line)]
    segments = [lines[i] for i in index]

    def rebuild(translated: list[str]) -> bytes:
        out = list(lines)
        for position, value in zip(index, translated):
            # Keep the original indentation; translators drop it.
            leading = out[position][:len(out[position]) - len(out[position].lstrip())]
            out[position] = leading + value.strip()
        return ("\n".join(out) + "\n").encode("utf-8")

    return Document(segments, rebuild, "text/plain; charset=utf-8", suffix, "text")


# --- subtitles -------------------------------------------------------------
_SRT_TIME = re.compile(r"^\s*[\d:,.]+\s*-->\s*[\d:,.]+")
_CUE_NUMBER = re.compile(r"^\s*\d+\s*$")


def _load_subtitles(data: bytes, suffix: str, kind: str) -> Document:
    text = data.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()

    def is_content(line: str) -> bool:
        if not line.strip():
            return False
        if _SRT_TIME.match(line) or _CUE_NUMBER.match(line):
            return False
        if kind == "vtt" and line.strip().upper().startswith(("WEBVTT", "NOTE", "STYLE")):
            return False
        return any(ch.isalpha() for ch in line)

    index = [i for i, line in enumerate(lines) if is_content(line)]
    segments = [lines[i].strip() for i in index]

    def rebuild(translated: list[str]) -> bytes:
        out = list(lines)
        for position, value in zip(index, translated):
            out[position] = value.strip()
        return ("\n".join(out) + "\n").encode("utf-8")

    media = "application/x-subrip" if kind == "srt" else "text/vtt"
    return Document(segments, rebuild, media, suffix, kind,
                    meta={"cues": len(index)})


# --- Word ------------------------------------------------------------------
def _load_docx(data: bytes) -> Document:
    import docx

    document = docx.Document(io.BytesIO(data))

    holders = []          # paragraph objects, in a stable order
    for paragraph in document.paragraphs:
        holders.append(paragraph)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                holders.extend(cell.paragraphs)

    index = [i for i, p in enumerate(holders)
             if p.text.strip() and any(ch.isalpha() for ch in p.text)]
    segments = [holders[i].text for i in index]

    def rebuild(translated: list[str]) -> bytes:
        for position, value in zip(index, translated):
            paragraph = holders[position]
            runs = paragraph.runs
            if runs:
                # Put the whole translation in the first run and blank the
                # rest: that keeps the paragraph's font, size and colour,
                # which per-run splitting would scramble because the
                # translation does not align with the original run
                # boundaries.
                runs[0].text = value
                for run in runs[1:]:
                    run.text = ""
            else:
                paragraph.text = value
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()

    return Document(
        segments, rebuild,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx", "docx",
        note="Formatting is preserved per paragraph; mixed styling inside a "
             "paragraph collapses to the first run's style.")


# --- PDF -------------------------------------------------------------------
_SENTENCE_END = re.compile(r"[.!?:;।。！？]['\"”’)]?$")


def _pdf_paragraphs(raw: str) -> list[str]:
    """Rebuild paragraphs from extracted PDF text.

    PDF has no notion of a paragraph: extraction gives visual lines, broken
    wherever the text happened to wrap. Splitting only on blank lines turned a
    whole page into one segment; splitting on every newline would hand the
    translator half-sentences, which it translates badly.

    So consecutive lines are joined unless the previous one looks finished —
    it ended a sentence, or it is short enough to be a heading or a list item.
    """
    lines = [line.strip() for line in raw.splitlines()]
    paragraphs: list[str] = []
    current = ""

    for line in lines:
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            continue
        if not current:
            current = line
            continue
        finished = bool(_SENTENCE_END.search(current)) or len(current) < 40
        if finished:
            paragraphs.append(current)
            current = line
        else:
            current = f"{current} {line}"       # a wrapped line, rejoin it

    if current:
        paragraphs.append(current)
    return [p for p in paragraphs if any(ch.isalpha() for ch in p)]


def _load_pdf(data: bytes) -> Document:
    import docx
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[list[str]] = []
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            log.warning("could not read a PDF page: %s", exc)
            raw = ""
        pages.append(_pdf_paragraphs(raw))

    segments = [p for page in pages for p in page]

    def rebuild(translated: list[str]) -> bytes:
        document = docx.Document()
        cursor = 0
        for number, page in enumerate(pages, start=1):
            if number > 1:
                document.add_page_break()
            for _ in page:
                document.add_paragraph(translated[cursor] if cursor < len(translated)
                                       else "")
                cursor += 1
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()

    return Document(
        segments, rebuild,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx", "pdf",
        note="PDF text was extracted and returned as an editable Word file. "
             "Rewriting the PDF in place would overlap and clip text, because "
             "translated glyphs are a different width from the originals.",
        meta={"pages": len(pages)})


# --- entry point -----------------------------------------------------------
def load(data: bytes, filename: str) -> Document:
    """Parse an uploaded file into translatable segments."""
    suffix = Path(filename or "").suffix.lower()
    kind = SUPPORTED.get(suffix)
    if kind is None:
        raise UnsupportedDocument(
            f"cannot read {suffix or 'a file with no extension'}; "
            f"supported: {', '.join(sorted(SUPPORTED))}")

    if kind == "text":
        document = _load_text(data, suffix)
    elif kind in ("srt", "vtt"):
        document = _load_subtitles(data, suffix, kind)
    elif kind == "docx":
        document = _load_docx(data)
    elif kind == "pdf":
        document = _load_pdf(data)
    else:                                    # pragma: no cover - guarded above
        raise UnsupportedDocument(f"no handler for {kind}")

    if len(document.segments) > _MAX_SEGMENTS:
        raise UnsupportedDocument(
            f"document has {len(document.segments)} segments, over the "
            f"{_MAX_SEGMENTS} limit; split it first")
    return document


def dedupe(segments: list[str]) -> tuple[list[str], list[int]]:
    """Collapse repeated segments.

    Subtitles and forms repeat the same line constantly, and translating one
    string twenty times is twenty model passes for one answer.
    """
    unique: list[str] = []
    seen: dict[str, int] = {}
    mapping: list[int] = []
    for segment in segments:
        key = segment.strip()
        if key not in seen:
            seen[key] = len(unique)
            unique.append(segment)
        mapping.append(seen[key])
    return unique, mapping


def expand(translated_unique: list[str], mapping: list[int]) -> list[str]:
    return [translated_unique[i] for i in mapping]

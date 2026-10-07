"""Domain document ingestion: parsing and chunking.

Parses .md and .txt files into overlapping text chunks with metadata,
ready for indexing and retrieval. Part of the Agentic ML Engineer
domain knowledge layer (spec sections 13-14).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single chunk of text from a parsed document."""

    document_id: str
    chunk_id: str
    source: str  # filename or label
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DomainDocument(BaseModel):
    """A parsed domain document composed of text chunks."""

    document_id: str
    source: str
    chunks: list[DocumentChunk]
    created_at: str


def _generate_document_id(text: str, source: str) -> str:
    """Generate a deterministic document ID from content and source."""
    digest = hashlib.sha256(f"{source}:{text}".encode()).hexdigest()[:12]
    return f"doc_{digest}"


def _generate_chunk_id(document_id: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    return f"{document_id}_chunk_{index:04d}"


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks."""
    # Preserve paragraph breaks (double newlines) but normalise everything else
    text = re.sub(r"\r\n", "\n", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace within lines
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def _extract_metadata_from_markdown(text: str) -> dict[str, Any]:
    """Extract lightweight metadata from markdown content.

    Pulls out headings and any YAML-style key: value pairs found at
    the top of the document (a simplified front-matter parser).
    """
    metadata: dict[str, Any] = {}

    # Extract headings for context
    headings = re.findall(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE)
    if headings:
        metadata["headings"] = [h[1].strip() for h in headings]

    # Simple front-matter extraction (lines matching "key: value" at the top)
    lines = text.lstrip().split("\n")
    front_matter: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)\s*:\s*(.+)$", line.strip())
        if match and not line.startswith("#"):
            front_matter[match.group(1).strip().lower().replace(" ", "_")] = (
                match.group(2).strip()
            )
        elif line.strip():
            # Stop at first non-key-value, non-empty line
            break
    if front_matter:
        metadata["front_matter"] = front_matter

    return metadata


def _chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Parameters
    ----------
    text:
        The full document text.
    chunk_size:
        Target number of characters per chunk.
    overlap:
        Number of characters to overlap between consecutive chunks.

    Returns
    -------
    list[str]:
        Ordered list of text chunks.
    """
    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Try paragraph-aware splitting first
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Current buffer is big enough to emit
            if current:
                chunks.append(current)
            # If the paragraph itself exceeds chunk_size, split by characters
            if len(para) > chunk_size:
                sub_chunks = _hard_split(para, chunk_size=chunk_size, overlap=overlap)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # If we only got one chunk and it's over the limit, hard-split
    if len(chunks) == 1 and len(chunks[0]) > chunk_size:
        chunks = _hard_split(chunks[0], chunk_size=chunk_size, overlap=overlap)

    # Apply overlap: prepend the tail of the previous chunk to the next
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail} {chunks[i]}".strip())
        chunks = overlapped

    return chunks


def _hard_split(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Character-level splitting with overlap, breaking at word boundaries."""
    words = text.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        added_len = len(word) + (1 if current_words else 0)
        if current_len + added_len > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # Keep enough trailing words for overlap
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current_words):
                if overlap_len + len(w) + 1 > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
            current_words = overlap_words
            current_len = sum(len(w) for w in current_words) + max(
                0, len(current_words) - 1
            )

        current_words.append(word)
        current_len += added_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _assign_chunk_metadata(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    doc_metadata: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    """Build per-chunk metadata."""
    meta: dict[str, Any] = {
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "source_type": "markdown" if source.endswith(".md") else "text",
    }

    # Attach the nearest heading context from the document
    if "headings" in doc_metadata:
        meta["document_headings"] = doc_metadata["headings"]

    if "front_matter" in doc_metadata:
        meta["front_matter"] = doc_metadata["front_matter"]

    return meta


def ingest_document(
    path: str | Path,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> DomainDocument:
    """Parse a .md or .txt file into chunks.

    Parameters
    ----------
    path:
        Path to a markdown (.md) or plain-text (.txt) file.
    chunk_size:
        Target number of characters per chunk.
    overlap:
        Number of overlapping characters between adjacent chunks.

    Returns
    -------
    DomainDocument:
        The parsed document with its text chunks.

    Raises
    ------
    FileNotFoundError:
        If *path* does not exist.
    ValueError:
        If the file extension is not .md or .txt.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in (".md", ".txt"):
        raise ValueError(
            f"Unsupported file type '{suffix}'. Only .md and .txt are supported."
        )

    text = path.read_text(encoding="utf-8")
    source = path.name

    return _build_document(
        text,
        source=source,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def ingest_text(
    text: str,
    source: str = "inline",
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> DomainDocument:
    """Parse raw text into chunks.

    Parameters
    ----------
    text:
        The raw text content to ingest.
    source:
        A label for the text origin (e.g. a filename or description).
    chunk_size:
        Target number of characters per chunk.
    overlap:
        Number of overlapping characters between adjacent chunks.

    Returns
    -------
    DomainDocument:
        The parsed document with its text chunks.
    """
    return _build_document(
        text,
        source=source,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def _build_document(
    text: str,
    *,
    source: str,
    chunk_size: int,
    overlap: int,
) -> DomainDocument:
    """Internal helper that produces a DomainDocument from raw text."""
    text = _normalise_whitespace(text)
    doc_id = _generate_document_id(text, source)
    doc_metadata = _extract_metadata_from_markdown(text)

    raw_chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    chunks: list[DocumentChunk] = []
    for i, chunk_text in enumerate(raw_chunks):
        meta = _assign_chunk_metadata(
            chunk_text,
            chunk_index=i,
            total_chunks=len(raw_chunks),
            doc_metadata=doc_metadata,
            source=source,
        )
        chunks.append(
            DocumentChunk(
                document_id=doc_id,
                chunk_id=_generate_chunk_id(doc_id, i),
                source=source,
                text=chunk_text,
                metadata=meta,
            )
        )

    return DomainDocument(
        document_id=doc_id,
        source=source,
        chunks=chunks,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

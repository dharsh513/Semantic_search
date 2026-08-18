"""
Sentence-aware chunking.

PubMed abstracts are short but often structured (Background / Methods /
Results / Conclusions). Splitting them into overlapping passages gives the
retriever finer-grained evidence and lets the generator quote a specific
finding rather than a whole abstract.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List

from config import config

# Abbreviations that must not end a sentence.
_ABBREV = r"(?<!\bDr)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bNo)(?<!\bSt)(?<!\bapprox)"
_SENT_SPLIT = re.compile(rf"{_ABBREV}(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_SECTION = re.compile(
    r"^(background|objective[s]?|introduction|aim[s]?|purpose|method[s]?|"
    r"material[s]? and method[s]?|design|result[s]?|finding[s]?|discussion|"
    r"conclusion[s]?|significance|importance)\s*:",
    re.IGNORECASE,
)


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _chunk_id(pmid: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{pmid}:{ordinal}:{digest}"


def chunk_article(article: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turn one article into a list of chunk dicts.

    Chunk 0 is always title + MeSH headings so that a document is retrievable
    even when it has no abstract (roughly 20% of PubMed records).
    """
    pmid = str(article.get("pmid", ""))
    title = (article.get("title") or "").strip()
    abstract = (article.get("abstract") or "").strip()
    mesh = article.get("mesh_terms") or []

    chunks: List[Dict[str, Any]] = []
    ordinal = 0

    header_bits = [title]
    if mesh:
        header_bits.append("MeSH: " + "; ".join(mesh[:12]))
    header = " ".join(b for b in header_bits if b).strip()
    if header:
        chunks.append(
            {
                "chunk_id": _chunk_id(pmid, ordinal, header),
                "pmid": pmid,
                "ordinal": ordinal,
                "section": "Title",
                "text": header,
            }
        )
        ordinal += 1

    if not abstract:
        return chunks

    sentences = split_sentences(abstract)
    size = max(20, int(config.CHUNK_WORDS))
    overlap = max(0, min(int(config.CHUNK_OVERLAP_WORDS), size - 10))
    min_words = int(config.MIN_CHUNK_WORDS)

    buf: List[str] = []
    buf_words = 0
    current_section = ""   # most recent structured-abstract heading seen
    buf_section = ""       # heading in force when this buffer started

    def flush() -> None:
        nonlocal buf, buf_words, ordinal
        body = " ".join(buf).strip()
        if len(body.split()) >= min_words:
            # Prefix the title so an isolated passage keeps its topical anchor.
            text = f"{title} — {body}" if title else body
            chunks.append(
                {
                    "chunk_id": _chunk_id(pmid, ordinal, text),
                    "pmid": pmid,
                    "ordinal": ordinal,
                    "section": buf_section or "Abstract",
                    "text": text,
                }
            )
            ordinal += 1

    for sent in sentences:
        marker = _SECTION.match(sent)
        if marker:
            current_section = marker.group(1).title()
        if not buf:
            buf_section = current_section
        words = len(sent.split())
        if buf and buf_words + words > size:
            flush()
            buf_section = current_section
            # carry the tail over as overlap
            tail: List[str] = []
            tail_words = 0
            for prev in reversed(buf):
                pw = len(prev.split())
                if tail_words + pw > overlap:
                    break
                tail.insert(0, prev)
                tail_words += pw
            buf, buf_words = tail, tail_words
        buf.append(sent)
        buf_words += words

    if buf:
        flush()

    return chunks


def chunk_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for art in articles:
        out.extend(chunk_article(art))
    return out

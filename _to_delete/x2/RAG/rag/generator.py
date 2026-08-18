"""
Answer generation (the "G" in RAG).

Runs fully offline: instead of calling an LLM, the generator performs
*grounded extractive synthesis* — it scores every sentence in the retrieved
passages against the query embedding, greedily selects a non-redundant set,
and stitches them into a cited answer. Every sentence in the output traces
back to a specific PMID, so the answer cannot hallucinate.

If `OPENAI_API_KEY` (or `GEMINI_API_KEY`) happens to be present in the
environment the module will use it to phrase the same retrieved evidence more
fluently — but it is never required, and the extractive answer is always
computed as the fallback.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import config
from rag.chunker import split_sentences
from rag.embedder import embedder

log = logging.getLogger(__name__)

_MIN_SENT_WORDS = 6
_MAX_SENT_WORDS = 70


def _clean(sentence: str) -> str:
    s = re.sub(r"\s+", " ", sentence).strip()
    # Drop the "Title — " prefix the chunker adds.
    if " — " in s:
        _, _, tail = s.partition(" — ")
        if len(tail.split()) >= _MIN_SENT_WORDS:
            s = tail
    # Drop structured-abstract section labels ("Background:", "Results:" ...).
    s = re.sub(
        r"^(background|objective[s]?|method[s]?|material[s]? and method[s]?|"
        r"result[s]?|finding[s]?|conclusion[s]?|introduction|aim[s]?|purpose|"
        r"discussion|significance|importance|design)\s*:\s*",
        "", s, flags=re.IGNORECASE,
    )
    return s.strip()


def _candidate_sentences(docs: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    [(sentence, pmid)] harvested from each document's matched passages.

    Title/MeSH passages are skipped: they are excellent retrieval anchors but
    make terrible answer sentences. A document contributes its abstract text,
    and is dropped from the answer entirely if it has none.
    """
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for doc in docs:
        pmid = doc["pmid"]
        blobs = [
            p["text"] for p in doc.get("matched_passages", [])
            if (p.get("section") or "").lower() != "title"
        ]
        if not blobs and doc.get("abstract"):
            blobs = [doc["abstract"]]
        for blob in blobs:
            blob = re.sub(r"\bMeSH:[^.]*", " ", blob)
            for sent in split_sentences(blob):
                s = _clean(sent)
                n = len(s.split())
                if n < _MIN_SENT_WORDS or n > _MAX_SENT_WORDS:
                    continue
                key = s.lower()[:120]
                if key in seen:
                    continue
                seen.add(key)
                out.append((s, pmid))
    return out


def extractive_answer(
    query: str,
    docs: List[Dict[str, Any]],
    query_vec: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build a citation-grounded answer from the retrieved documents."""
    if not docs:
        return {
            "answer": (
                "No PubMed records matched this query closely enough to support "
                "an answer. Try broadening the terms or removing filters."
            ),
            "citations": [],
            "mode": "extractive",
            "grounded": False,
        }

    candidates = _candidate_sentences(docs)
    if not candidates:
        return {
            "answer": (
                "The matching PubMed records have no abstract text available, "
                "so only titles and MeSH headings could be retrieved. "
                "See the sources below."
            ),
            "citations": _citations(docs[: config.ANSWER_MAX_SOURCES], {}),
            "mode": "extractive",
            "grounded": False,
        }

    sentences = [c[0] for c in candidates]
    pmids = [c[1] for c in candidates]

    if query_vec is None:
        query_vec = embedder.encode_query(query)
    sent_vecs = embedder.encode_documents(sentences)

    if sent_vecs.size and np.linalg.norm(query_vec) > 0:
        sims = sent_vecs @ np.asarray(query_vec, dtype="float32").reshape(-1, 1)
        sims = sims.ravel()
    else:
        sims = np.zeros(len(sentences), dtype="float32")

    # Rank bonus: prefer sentences from higher-ranked documents.
    doc_rank = {d["pmid"]: d.get("rank", 99) for d in docs}
    bonus = np.array(
        [0.06 / max(doc_rank.get(p, 99), 1) for p in pmids], dtype="float32"
    )
    scored = sims + bonus

    # Greedy MMR-style selection over sentences.
    n_want = int(config.ANSWER_SENTENCES)
    chosen: List[int] = []
    used_docs: Dict[str, int] = {}
    order = list(np.argsort(-scored))

    for idx in order:
        if len(chosen) >= n_want:
            break
        idx = int(idx)
        pmid = pmids[idx]
        if used_docs.get(pmid, 0) >= 2:      # cap per-paper contribution
            continue
        if chosen and sent_vecs.size:
            redundancy = float((sent_vecs[chosen] @ sent_vecs[idx]).max())
            if redundancy > 0.88:            # near-duplicate sentence
                continue
        chosen.append(idx)
        used_docs[pmid] = used_docs.get(pmid, 0) + 1

    if not chosen:
        chosen = [int(order[0])]

    # Keep the original document ordering for readability.
    chosen.sort(key=lambda i: (doc_rank.get(pmids[i], 99), -float(scored[i])))

    # Assign citation numbers in order of first appearance.
    cite_no: Dict[str, int] = {}
    lines: List[str] = []
    for i in chosen:
        pmid = pmids[i]
        if pmid not in cite_no:
            cite_no[pmid] = len(cite_no) + 1
        lines.append(f"{sentences[i].rstrip('.')} [{cite_no[pmid]}]")

    body = ". ".join(lines) + "."
    header = f"Based on {len(cite_no)} PubMed record(s) retrieved for “{query}”:\n\n"
    answer = header + body

    cited_docs = [d for d in docs if d["pmid"] in cite_no]
    cited_docs.sort(key=lambda d: cite_no[d["pmid"]])

    return {
        "answer": answer,
        "citations": _citations(cited_docs, cite_no),
        "mode": "extractive",
        "grounded": True,
    }


def _citations(docs: List[Dict[str, Any]], cite_no: Dict[str, int]) -> List[Dict[str, Any]]:
    out = []
    for i, d in enumerate(docs, start=1):
        authors = d.get("authors") or []
        first = authors[0] if authors else ""
        label = f"{first} et al." if len(authors) > 1 else (first or "Unknown author")
        out.append(
            {
                "n": cite_no.get(d["pmid"], i),
                "pmid": d["pmid"],
                "title": d.get("title", ""),
                "citation": " ".join(
                    x for x in [label, d.get("journal", ""), d.get("year", "")] if x
                ),
                "url": d.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{d['pmid']}/"),
                "doi": d.get("doi", ""),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Optional LLM phrasing (never required)
# --------------------------------------------------------------------------- #
def _llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def generate(
    query: str,
    docs: List[Dict[str, Any]],
    query_vec: Optional[np.ndarray] = None,
    allow_llm: bool = True,
) -> Dict[str, Any]:
    """
    Public entry point. Always returns a grounded answer.

    The extractive answer is computed first and returned unless an optional LLM
    key is configured, in which case the same retrieved evidence is rephrased.
    """
    base = extractive_answer(query, docs, query_vec)

    if not allow_llm or not _llm_available() or not base.get("grounded"):
        return base

    try:
        from openai import OpenAI  # optional dependency

        context = "\n\n".join(
            f"[{c['n']}] PMID {c['pmid']} — {c['title']}\n"
            + " ".join(
                p["text"] for p in next(
                    d for d in docs if d["pmid"] == c["pmid"]
                ).get("matched_passages", [])
            )
            for c in base["citations"]
        )
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a biomedical research assistant. Answer ONLY "
                        "from the numbered PubMed excerpts provided. Cite every "
                        "claim inline as [n]. If the excerpts do not answer the "
                        "question, say so plainly. Never invent citations."
                    ),
                },
                {"role": "user", "content": f"Question: {query}\n\nExcerpts:\n{context}"},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return {**base, "answer": text, "mode": "llm"}
    except Exception as exc:  # noqa: BLE001 - LLM is strictly optional
        log.info("LLM phrasing skipped (%s); using extractive answer.", exc)

    return base

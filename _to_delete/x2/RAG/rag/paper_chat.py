"""
Per-paper chat — a miniature RAG loop scoped to one PubMed record.

When the user opens a result and asks a question, the answer is built ONLY
from that paper's own text (title, structured abstract, MeSH headings). The
paper is split into sentences, each sentence is embedded and scored against
the question, and the best non-redundant sentences are returned verbatim with
their section labels.

Consequences of the extractive design:
  * the assistant cannot invent a finding the paper does not contain,
  * every reply is quotable back to the abstract,
  * when the abstract genuinely does not cover the question, confidence is low
    and the assistant says so instead of guessing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rag.chunker import split_sentences
from rag.embedder import embedder
from rag.pubmed_client import STOPWORDS
from rag.store import store

# --------------------------------------------------------------------------- #
# Section intent — nudges the answer toward the right part of the abstract.
# --------------------------------------------------------------------------- #
_INTENTS: List[Tuple[str, List[str]]] = [
    ("Methods", [
        "method", "methods", "methodology", "how did", "how was", "design",
        "sample", "participants", "cohort", "population", "recruit", "measure",
        "assay", "protocol", "procedure", "technique", "instrument", "dose",
        "sample size", "how many", "randomis", "randomiz", "blinded", "control group",
    ]),
    ("Results", [
        "result", "results", "finding", "findings", "outcome", "effect",
        "significant", "correlation", "association", "difference", "increase",
        "decrease", "reduce", "improve", "risk", "odds", "hazard", "p value",
        "p-value", "confidence interval", "how much", "magnitude", "efficacy",
    ]),
    # Cues are matched as substrings, so stems cover their inflections
    # ("conclud" catches conclude / concluded / concludes / concluding).
    ("Conclusions", [
        "conclusion", "conclud", "implication", "takeaway", "take away",
        "so what", "clinical relevance", "recommend", "suggest",
        "mean for", "bottom line", "summary of findings",
    ]),
    ("Background", [
        "background", "why", "rationale", "context", "motivation", "problem",
        "objective", "aim", "purpose", "hypothesis", "research question",
    ]),
]

_SUMMARY_RE = re.compile(
    r"\b(summar\w*|tl;?dr|overview|in short|key points?|main points?|"
    r"what is this (paper|study|article) about|explain this paper)\b",
    re.IGNORECASE,
)

_LIMITATION_RE = re.compile(
    r"\b(limitation|weakness|caveat|drawback|bias|generaliz|generalis)\w*\b",
    re.IGNORECASE,
)

# Confidence below this and the assistant declines rather than guesses.
LOW_CONFIDENCE = 0.30
MIN_SENT_WORDS = 5


def _tokens(text: str) -> set:
    return {
        t for t in re.findall(r"[a-z0-9][a-z0-9\-']*", (text or "").lower())
        if len(t) > 2 and t not in STOPWORDS
    }


def detect_intent(question: str) -> Optional[str]:
    q = (question or "").lower()
    best, best_hits = None, 0
    for section, cues in _INTENTS:
        hits = sum(1 for cue in cues if cue in q)
        if hits > best_hits:
            best, best_hits = section, hits
    return best


# --------------------------------------------------------------------------- #
def _sentences(article: Dict[str, Any]) -> List[Dict[str, str]]:
    """Sentence-level view of the paper, each tagged with its section."""
    out: List[Dict[str, str]] = []
    section = "Abstract"
    label_re = re.compile(
        r"^(background|objective[s]?|introduction|aim[s]?|purpose|method[s]?|"
        r"material[s]? and method[s]?|design|result[s]?|finding[s]?|discussion|"
        r"conclusion[s]?|significance|importance)\s*:\s*",
        re.IGNORECASE,
    )

    for raw in split_sentences(article.get("abstract") or ""):
        marker = label_re.match(raw)
        if marker:
            section = marker.group(1).title()
            if section.startswith("Objective") or section in {"Aim", "Aims", "Purpose",
                                                              "Introduction"}:
                section = "Background"
            elif section.startswith("Finding"):
                section = "Results"
            elif section.startswith("Material"):
                section = "Methods"
            raw = label_re.sub("", raw).strip()
        if len(raw.split()) >= MIN_SENT_WORDS:
            out.append({"text": raw, "section": section})

    return out


def suggested_questions(article: Dict[str, Any]) -> List[str]:
    """Starter prompts tailored to what this particular abstract contains."""
    text = (article.get("abstract") or "").lower()
    qs = ["Summarise this paper in plain language"]
    if any(k in text for k in ("method", "we performed", "we conducted", "design",
                               "participants", "patients", "randomi")):
        qs.append("What methods and sample size were used?")
    if any(k in text for k in ("result", "showed", "found", "significant", "%",
                               "increase", "reduc", "associated")):
        qs.append("What were the main findings?")
    if any(k in text for k in ("conclusion", "suggest", "support", "indicate")):
        qs.append("What do the authors conclude?")
    if article.get("mesh_terms"):
        qs.append("What topics is this paper indexed under?")
    qs.append("What are the limitations of this study?")
    return qs[:5]


# --------------------------------------------------------------------------- #
def answer_question(pmid: str, question: str,
                    article: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Answer `question` using only the article identified by `pmid`.

    Returns {answer, evidence[], confidence, intent, grounded, suggestions[]}.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")

    if article is None:
        article = store.get_articles([pmid]).get(str(pmid))
    if not article:
        return {
            "answer": f"PMID {pmid} is not in the local cache. Open it from a "
                      "search result first.",
            "evidence": [], "confidence": 0.0, "intent": None,
            "grounded": False, "suggestions": [],
        }

    sentences = _sentences(article)

    # ---- MeSH questions are answered from the indexing metadata ----------
    if re.search(r"\b(mesh|index(ed)?|subject heading|categor(y|ies)|keyword)\b",
                 question, re.IGNORECASE):
        mesh = article.get("mesh_terms") or []
        kws = article.get("keywords") or []
        if mesh or kws:
            parts = []
            if mesh:
                parts.append("Indexed in PubMed under the MeSH headings: "
                             + "; ".join(mesh) + ".")
            if kws:
                parts.append("Author keywords: " + "; ".join(kws) + ".")
            return {
                "answer": " ".join(parts),
                "evidence": [{"section": "MeSH", "text": "; ".join(mesh or kws),
                              "score": 1.0}],
                "confidence": 1.0, "intent": "MeSH", "grounded": True,
                "suggestions": suggested_questions(article),
            }

    if not sentences:
        return {
            "answer": "This record has no abstract in PubMed, so there is no text "
                      "to answer from. Only its title and MeSH headings are "
                      "available — try asking what it is indexed under, or open "
                      "the full text on PubMed.",
            "evidence": [], "confidence": 0.0, "intent": None,
            "grounded": False, "suggestions": ["What topics is this paper indexed under?"],
        }

    # ---- score every sentence against the question -----------------------
    texts = [s["text"] for s in sentences]
    q_vec = embedder.encode_query(question)
    s_vecs = embedder.encode_documents(texts)

    if s_vecs.size and np.linalg.norm(q_vec) > 0:
        dense = (s_vecs @ np.asarray(q_vec, dtype="float32").reshape(-1, 1)).ravel()
    else:
        dense = np.zeros(len(texts), dtype="float32")

    # Lexical overlap keeps exact entities (drug names, genes) anchored.
    q_tokens = _tokens(question)
    lexical = np.array(
        [
            len(q_tokens & _tokens(t)) / max(len(q_tokens), 1)
            for t in texts
        ],
        dtype="float32",
    )

    scores = 0.68 * dense + 0.32 * lexical

    # Section intent bonus.
    intent = detect_intent(question)
    if intent:
        bonus = np.array(
            [0.12 if s["section"] == intent else 0.0 for s in sentences],
            dtype="float32",
        )
        scores = scores + bonus

    is_summary = bool(_SUMMARY_RE.search(question))
    is_limitation = bool(_LIMITATION_RE.search(question))

    # ---- select the sentences that make up the reply ---------------------
    want = 4 if is_summary else 3
    order = list(np.argsort(-scores))

    # Only keep sentences close to the best one. Without this a narrow factual
    # question ("how many patients?") pads its answer with whatever ranked
    # second and third, which reads as padding and dilutes the answer.
    peak = float(scores[order[0]]) if len(order) else 0.0
    floor = peak * 0.62 if peak > 0 else float("-inf")

    chosen: List[int] = []
    for idx in order:
        idx = int(idx)
        if len(chosen) >= want:
            break
        if chosen and not is_summary and float(scores[idx]) < floor:
            break  # everything after this is weaker still
        if chosen and s_vecs.size:
            if float((s_vecs[chosen] @ s_vecs[idx]).max()) > 0.9:
                continue  # near-duplicate
        chosen.append(idx)

    if is_summary:
        # A summary should walk the paper in order, one sentence per section.
        seen_sections, picked = set(), []
        for idx in order:
            idx = int(idx)
            sec = sentences[idx]["section"]
            if sec not in seen_sections:
                seen_sections.add(sec)
                picked.append(idx)
            if len(picked) >= 4:
                break
        chosen = picked or chosen

    # The single best-scoring sentence drives confidence and the fallback
    # quote. Capture it BEFORE reordering, because `chosen` is about to be
    # sorted back into the paper's narrative order for readability.
    best = int(chosen[0]) if chosen else int(np.argmax(scores))

    chosen.sort()  # restore the paper's own narrative order

    # ---- confidence ------------------------------------------------------
    # Raw cosine is NOT comparable across embedding models (a different model,
    # or a different vector dimension, shifts the whole similarity range), so
    # confidence is built from three scale-free signals instead:
    #
    #   discrimination  does one sentence actually stand out, or does the paper
    #                   answer this question no better than any other?
    #   lexical cover   how many of the question's own content words appear in
    #                   the sentence we are about to quote?
    #   intent hit      if the question asked for methods/results/conclusions,
    #                   did the winning sentence come from that section?
    #
    # An off-topic question produces a flat score profile, zero lexical overlap
    # and no intent match — which is exactly what should read as "I don't know".
    top_score = float(scores[best])

    if len(scores) >= 3:
        spread = float(scores.max() - scores.min())
        gap = float(scores[best] - np.median(scores))
        discrimination = float(np.clip(gap / spread, 0.0, 1.0)) if spread > 1e-6 else 0.0
    else:
        discrimination = 0.5

    lexical_cover = float(np.clip(lexical[best], 0.0, 1.0))
    intent_hit = 1.0 if (intent and sentences[best]["section"] == intent) else 0.0

    confidence = float(np.clip(
        0.42 * lexical_cover + 0.38 * discrimination + 0.20 * intent_hit, 0.0, 1.0
    ))

    # A summary is always well grounded — it is the paper's own sentences, and
    # the question's wording ("summarise this") deliberately shares no
    # vocabulary with the abstract, so the signals above do not apply.
    if is_summary:
        confidence = 1.0 if len({sentences[i]["section"] for i in chosen}) >= 2 else 0.75

    evidence = [
        {
            "section": sentences[i]["section"],
            "text": sentences[i]["text"],
            "score": round(float(scores[i]), 4),
        }
        for i in chosen
    ]

    # ---- compose the reply ----------------------------------------------
    body = " ".join(sentences[i]["text"].rstrip(".") + "." for i in chosen)

    if is_summary:
        reply = f"In short: {body}"
    elif confidence < LOW_CONFIDENCE:
        reply = (
            "The abstract does not directly address that. The closest thing it "
            f"says is: “{sentences[best]['text']}” — you may need the full "
            "text on PubMed, or try searching for it as a new query."
        )
    elif is_limitation:
        reply = (
            "The abstract does not contain an explicit limitations section — "
            "PubMed abstracts rarely do. Judging from what it reports: "
            + body
            + " For stated limitations you would need the full text."
        )
    elif intent:
        lead = {
            "Methods": "On methods, the paper states:",
            "Results": "The reported results:",
            "Conclusions": "The authors conclude:",
            "Background": "For background, the paper states:",
        }.get(intent, "From the abstract:")
        reply = f"{lead} {body}"
    else:
        reply = body

    return {
        "answer": reply,
        "evidence": evidence,
        "confidence": round(confidence, 3),
        "signals": {
            "discrimination": round(discrimination, 3),
            "lexical_cover": round(lexical_cover, 3),
            "intent_hit": bool(intent_hit),
            "top_score": round(top_score, 3),
        },
        "intent": intent,
        "grounded": confidence >= LOW_CONFIDENCE,
        "suggestions": suggested_questions(article),
    }


# --------------------------------------------------------------------------- #
def chat(pmid: str, question: str, persist: bool = True) -> Dict[str, Any]:
    """Answer a question and append both turns to the stored transcript."""
    article = store.get_articles([pmid]).get(str(pmid))
    result = answer_question(pmid, question, article)

    if persist:
        store.add_chat_message(pmid, "user", question)
        store.add_chat_message(
            pmid, "assistant", result["answer"],
            evidence=result.get("evidence"), confidence=result.get("confidence", 0.0),
        )

    return result

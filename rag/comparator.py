"""
Paper Comparison Engine.

Performs structured, multi-dimensional side-by-side comparison across 2–5 selected papers:
  1. Research Objective & Problem Statement
  2. Dataset, Cohort & Study Population
  3. Methodology, Models & Algorithms
  4. Experimental Setup & Validation
  5. Main Findings & Quantified Metrics (accuracy, AUC, p-values, hazard ratios, etc.)
  6. Strengths & Contributions
  7. Limitations & Caveats
  8. Future Work & Stated Directions
  9. Metadata (Year, Journal, Authors, DOI, PMID)

Followed by an AI Comparative Summary highlighting consensus, divergences, and complementary insights.
Strictly grounded in available paper text; unavailable dimensions report 'Not specified in the available paper content.'
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rag.chunker import split_sentences
from rag.store import store

log = logging.getLogger(__name__)

NOT_SPECIFIED = "Not specified in the available paper content."

_METRIC_RE = re.compile(
    r"(\b(auc|auroc|accuracy|sensitivity|specificity|precision|recall|f1[- ]score|"
    r"p\s*[<=<]\s*0?\.\d+|p-value|ci|odds ratio|or|hazard ratio|hr|relative risk|rr|"
    r"correlation|r\s*=\s*-?0?\.\d+|\d+(\.\d+)?%|\d+(\.\d+)?\s*(mg|g|ml|kg|years|months|days))\b)",
    re.IGNORECASE,
)

_COHORT_RE = re.compile(
    r"(\b(\d+[\d,]*\s*(patients|participants|subjects|cases|controls|individuals|samples|cohorts|scans|images|records|mice|rats|volunteers)|"
    r"n\s*=\s*\d+[\d,]*|sample size of \d+|cohort of \d+)\b)",
    re.IGNORECASE,
)

_LIMITATION_RE = re.compile(
    r"\b(limitation[s]?|limited by|caveat[s]?|weakness|drawback[s]?|small sample|single[- ]center|retrospective|lack of external|unvalidated)\b",
    re.IGNORECASE,
)

_FUTURE_RE = re.compile(
    r"\b(future (work|studies|research|investigations|trials)|further (work|studies|research|investigation|validation) (is|are)? needed|warrants further|remains to be investigated)\b",
    re.IGNORECASE,
)


def _sectioned_sentences(abstract: str) -> List[Tuple[str, str]]:
    """Splits abstract into (section, clean_sentence) tuples."""
    if not abstract:
        return []

    label_re = re.compile(
        r"^(background|objective[s]?|introduction|aim[s]?|purpose|method[s]?|"
        r"material[s]? and method[s]?|design|result[s]?|finding[s]?|discussion|"
        r"conclusion[s]?|significance|importance)\s*:\s*",
        re.IGNORECASE,
    )

    out = []
    current_sec = "Abstract"
    for s in split_sentences(abstract):
        m = label_re.match(s)
        if m:
            raw_sec = m.group(1).title()
            if raw_sec.startswith("Objective") or raw_sec in {"Aim", "Aims", "Purpose", "Introduction"}:
                current_sec = "Background"
            elif raw_sec.startswith("Material") or raw_sec.startswith("Design"):
                current_sec = "Methods"
            elif raw_sec.startswith("Finding"):
                current_sec = "Results"
            elif raw_sec.startswith("Significance") or raw_sec.startswith("Importance"):
                current_sec = "Conclusions"
            else:
                current_sec = raw_sec
            s = label_re.sub("", s).strip()
        if len(s.split()) >= 4:
            out.append((current_sec, s))
    return out


def extract_paper_dimensions(article: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts structured comparative dimensions from a single article."""
    pmid = str(article.get("pmid", ""))
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    year = article.get("year", "") or article.get("pub_date", "")
    journal = article.get("journal", "")
    authors = article.get("authors", [])
    doi = article.get("doi", "")
    mesh = article.get("mesh_terms", [])

    sentences = _sectioned_sentences(abstract)

    # 1. Objective / Problem
    obj_sents = [s for sec, s in sentences if sec in {"Background", "Objective", "Introduction"}]
    if not obj_sents and sentences:
        obj_sents = [sentences[0][1]]
    objective = " ".join(obj_sents[:2]).strip() if obj_sents else NOT_SPECIFIED

    # 2. Dataset / Population / Cohort
    cohort_matches = []
    for _, s in sentences:
        if _COHORT_RE.search(s) or any(k in s.lower() for k in ("cohort", "dataset", "patients", "participants", "sample")):
            cohort_matches.append(s)
    dataset = " ".join(cohort_matches[:2]).strip() if cohort_matches else (
        "Dataset details not explicitly specified in the abstract." if abstract else NOT_SPECIFIED
    )

    # 3. Methodology / Model / Algorithm
    method_sents = [s for sec, s in sentences if sec in {"Methods", "Design"}]
    if not method_sents:
        method_sents = [
            s for _, s in sentences
            if any(k in s.lower() for k in ("we used", "we developed", "we performed", "model", "algorithm", "deep learning", "trial", "assay", "technique"))
        ]
    methodology = " ".join(method_sents[:3]).strip() if method_sents else (
        "Methodological details not detailed in abstract." if abstract else NOT_SPECIFIED
    )

    # 4. Experimental Setup / Validation
    val_sents = [
        s for _, s in sentences
        if any(k in s.lower() for k in ("validation", "cross-validation", "evaluated on", "tested on", "baseline", "compared with", "control group", "blinded", "randomized"))
    ]
    experimental_setup = " ".join(val_sents[:2]).strip() if val_sents else (
        "Experimental validation setup not explicitly isolated in abstract." if abstract else NOT_SPECIFIED
    )

    # 5. Main Findings & Quantified Metrics
    result_sents = [s for sec, s in sentences if sec in {"Results", "Findings"}]
    if not result_sents:
        result_sents = [s for _, s in sentences if _METRIC_RE.search(s) or "found that" in s.lower() or "showed that" in s.lower()]
    findings = " ".join(result_sents[:3]).strip() if result_sents else (
        "Specific quantified findings not available in abstract text." if abstract else NOT_SPECIFIED
    )

    # 6. Strengths & Key Contribution
    concl_sents = [s for sec, s in sentences if sec in {"Conclusions", "Significance", "Discussion"}]
    if not concl_sents and sentences:
        concl_sents = [sentences[-1][1]]
    strengths = " ".join(concl_sents[:2]).strip() if concl_sents else (
        "Key contributions summarized in title and indexing." if not abstract else NOT_SPECIFIED
    )

    # 7. Limitations
    limit_sents = [s for _, s in sentences if _LIMITATION_RE.search(s)]
    if limit_sents:
        limitations = " ".join(limit_sents[:2]).strip()
    elif abstract:
        limitations = "Not explicitly stated in abstract (typical for PubMed abstracts; check full text)."
    else:
        limitations = NOT_SPECIFIED

    # 8. Future Work
    future_sents = [s for _, s in sentences if _FUTURE_RE.search(s)]
    if future_sents:
        future_work = " ".join(future_sents[:2]).strip()
    elif concl_sents and any("suggest" in s.lower() or "recommend" in s.lower() for s in concl_sents):
        future_work = [s for s in concl_sents if "suggest" in s.lower() or "recommend" in s.lower()][0]
    else:
        future_work = "Not explicitly stated in abstract."

    return {
        "pmid": pmid,
        "title": title,
        "year": str(year),
        "journal": journal,
        "authors": authors,
        "first_author": authors[0] if authors else "Unknown",
        "doi": doi,
        "mesh_terms": mesh,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "has_abstract": bool(abstract.strip()),
        "objective": objective,
        "dataset": dataset,
        "methodology": methodology,
        "experimental_setup": experimental_setup,
        "findings": findings,
        "strengths": strengths,
        "limitations": limitations,
        "future_work": future_work,
    }


def generate_comparative_summary(papers: List[Dict[str, Any]]) -> str:
    """Generates an evidence-backed comparative synthesis across the selected papers."""
    if not papers:
        return "No papers provided for comparison."

    n = len(papers)
    parts = []

    # Header / Scope
    labels = [f"“{p['title'][:60]}…” ({p.get('first_author', 'Unknown')}, {p.get('year', 'N/A')})" for p in papers]
    parts.append(f"Comparative synthesis across {n} selected biomedical studies: {'; '.join(labels)}.")

    # Objectives & Themes
    parts.append(
        "\n\n**Common Objectives & Focus Areas:**\n"
        + "\n".join(f"• **PMID {p['pmid']} ({p.get('year', 'N/A')}):** {p['objective']}" for p in papers if p['objective'] != NOT_SPECIFIED)
    )

    # Methodological Approaches
    parts.append(
        "\n\n**Methodological & Model Contrasts:**\n"
        + "\n".join(f"• **PMID {p['pmid']}:** {p['methodology']}" for p in papers if p['methodology'] != NOT_SPECIFIED)
    )

    # Findings & Outcomes
    parts.append(
        "\n\n**Key Findings & Reported Evidence:**\n"
        + "\n".join(f"• **PMID {p['pmid']}:** {p['findings']}" for p in papers if p['findings'] != NOT_SPECIFIED)
    )

    # Limitations & Complementary Value
    limits = [f"• **PMID {p['pmid']}:** {p['limitations']}" for p in papers if p['limitations'] != NOT_SPECIFIED and "typical for PubMed" not in p['limitations']]
    if limits:
        parts.append("\n\n**Noted Limitations & Caveats:**\n" + "\n".join(limits))

    return "".join(parts)


class PaperComparator:
    """Orchestrates structured multi-paper comparison."""

    def compare(self, pmids: Sequence[str]) -> Dict[str, Any]:
        pmids = [str(p).strip() for p in pmids if str(p).strip()]
        if len(pmids) < 2:
            raise ValueError("Comparison requires at least 2 papers.")
        if len(pmids) > 3:
            raise ValueError("Comparison supports a maximum of 3 papers at once.")

        cached_articles = store.get_articles(pmids)
        missing = [p for p in pmids if p not in cached_articles]
        if missing:
            from rag.pipeline import pipeline
            fetched = pipeline.article_batch(missing)
            cached_articles.update({a["pmid"]: a for a in fetched})

        papers_data = []
        for pmid in pmids:
            art = cached_articles.get(pmid)
            if art:
                papers_data.append(extract_paper_dimensions(art))

        if len(papers_data) < 2:
            raise ValueError("Could not retrieve enough valid papers for comparison.")

        # Build side-by-side dimension rows
        dimensions = [
            {"id": "objective", "label": "Research Objective / Problem", "values": {p["pmid"]: p["objective"] for p in papers_data}},
            {"id": "dataset", "label": "Dataset / Study Population", "values": {p["pmid"]: p["dataset"] for p in papers_data}},
            {"id": "methodology", "label": "Methodology / Model / Algorithm", "values": {p["pmid"]: p["methodology"] for p in papers_data}},
            {"id": "experimental_setup", "label": "Experimental Validation", "values": {p["pmid"]: p["experimental_setup"] for p in papers_data}},
            {"id": "findings", "label": "Main Findings & Quantified Metrics", "values": {p["pmid"]: p["findings"] for p in papers_data}},
            {"id": "strengths", "label": "Key Contributions & Strengths", "values": {p["pmid"]: p["strengths"] for p in papers_data}},
            {"id": "limitations", "label": "Identified Limitations", "values": {p["pmid"]: p["limitations"] for p in papers_data}},
            {"id": "future_work", "label": "Stated Future Directions", "values": {p["pmid"]: p["future_work"] for p in papers_data}},
            {"id": "year", "label": "Publication Year", "values": {p["pmid"]: p["year"] for p in papers_data}},
            {"id": "journal", "label": "Journal", "values": {p["pmid"]: p["journal"] for p in papers_data}},
        ]

        summary = generate_comparative_summary(papers_data)

        return {
            "pmids": pmids,
            "paper_count": len(papers_data),
            "papers": papers_data,
            "dimensions": dimensions,
            "summary": summary,
        }


comparator = PaperComparator()

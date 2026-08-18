"""
Citation formatting and reference-manager export.

Vancouver / NLM style is used throughout — it is the convention for biomedical
literature and matches what PubMed itself displays.

    Valles-Colomer M, Falony G, Raes J. The gut-brain axis: microbial
    regulation of depressive behaviour. Nat Microbiol. 2019;4(4):623-32.
    doi:10.1000/test. PMID: 31456127.

Vancouver lists up to six authors; beyond that the first six are given
followed by "et al." (NLM's rule).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

MAX_AUTHORS = 6


# --------------------------------------------------------------------------- #
def author_string(authors: List[str], max_authors: int = MAX_AUTHORS) -> str:
    authors = [a for a in (authors or []) if a]
    if not authors:
        return "[No authors listed]"
    if len(authors) <= max_authors:
        return ", ".join(authors)
    return ", ".join(authors[:max_authors]) + ", et al"


def _tidy_title(title: str) -> str:
    title = re.sub(r"\s+", " ", (title or "")).strip()
    return title if title.endswith((".", "?", "!")) else title + "."


def vancouver(article: Dict[str, Any], include_pmid: bool = True) -> str:
    """Render one article as a Vancouver/NLM reference string."""
    bits: List[str] = [author_string(article.get("authors") or [])]
    bits[-1] = bits[-1].rstrip(".") + "."

    bits.append(_tidy_title(article.get("title") or "[Title not available]"))

    journal = (article.get("journal") or "").strip()
    if journal:
        bits.append(journal.rstrip(".") + ".")

    # Year;Volume(Issue):Pages
    year = (article.get("year") or "").strip()
    volume = (article.get("volume") or "").strip()
    issue = (article.get("issue") or "").strip()
    pages = (article.get("pages") or "").strip()

    locator = year
    if volume:
        locator += f";{volume}"
        if issue:
            locator += f"({issue})"
    elif issue:
        locator += f";({issue})"
    if pages:
        locator += f":{pages}"
    if locator:
        bits.append(locator.rstrip(".") + ".")

    doi = (article.get("doi") or "").strip()
    if doi:
        bits.append(f"doi:{doi}.")

    if include_pmid and article.get("pmid"):
        bits.append(f"PMID: {article['pmid']}.")

    return " ".join(b for b in bits if b).strip()


def numbered_reference_list(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"n": i, "pmid": a.get("pmid", ""), "text": vancouver(a),
         "url": a.get("url", "")}
        for i, a in enumerate(articles, start=1)
    ]


# --------------------------------------------------------------------------- #
# BibTeX
# --------------------------------------------------------------------------- #
_BIB_ESCAPE = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _bib_escape(text: str) -> str:
    out = []
    for ch in str(text or ""):
        out.append(_BIB_ESCAPE.get(ch, ch))
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _bib_key(article: Dict[str, Any]) -> str:
    authors = article.get("authors") or []
    surname = re.sub(r"[^A-Za-z]", "", authors[0].split()[0]) if authors else "anon"
    year = re.sub(r"[^0-9]", "", article.get("year") or "") or "nd"
    return f"{surname.lower()}{year}pmid{article.get('pmid', '')}"


def bibtex(articles: List[Dict[str, Any]]) -> str:
    entries: List[str] = []
    for a in articles:
        fields = [
            ("author", " and ".join(a.get("authors") or []) or "Anonymous"),
            ("title", a.get("title") or ""),
            ("journal", a.get("journal") or ""),
            ("year", a.get("year") or ""),
            ("volume", a.get("volume") or ""),
            ("number", a.get("issue") or ""),
            ("pages", (a.get("pages") or "").replace("-", "--")),
            ("doi", a.get("doi") or ""),
            ("pmid", a.get("pmid") or ""),
            ("url", a.get("url") or ""),
            ("keywords", "; ".join(a.get("mesh_terms") or [])),
        ]
        body = ",\n".join(
            f"  {name} = {{{_bib_escape(value)}}}"
            for name, value in fields if str(value).strip()
        )
        entries.append(f"@article{{{_bib_key(a)},\n{body}\n}}")
    return "\n\n".join(entries) + "\n"


# --------------------------------------------------------------------------- #
# RIS (EndNote / Mendeley / Zotero)
# --------------------------------------------------------------------------- #
def ris(articles: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for a in articles:
        lines = ["TY  - JOUR"]
        for author in a.get("authors") or []:
            lines.append(f"AU  - {author}")
        if a.get("title"):
            lines.append(f"TI  - {a['title']}")
        if a.get("journal"):
            lines.append(f"JO  - {a['journal']}")
        if a.get("year"):
            lines.append(f"PY  - {a['year']}")
        if a.get("volume"):
            lines.append(f"VL  - {a['volume']}")
        if a.get("issue"):
            lines.append(f"IS  - {a['issue']}")
        pages = (a.get("pages") or "").strip()
        if pages:
            start, _, end = pages.partition("-")
            lines.append(f"SP  - {start}")
            if end:
                lines.append(f"EP  - {end}")
        if a.get("abstract"):
            lines.append("AB  - " + re.sub(r"\s+", " ", a["abstract"]))
        for term in a.get("mesh_terms") or []:
            lines.append(f"KW  - {term}")
        if a.get("doi"):
            lines.append(f"DO  - {a['doi']}")
        if a.get("url"):
            lines.append(f"UR  - {a['url']}")
        if a.get("pmid"):
            lines.append(f"AN  - {a['pmid']}")
        lines.append("ER  - ")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


# --------------------------------------------------------------------------- #
def plain_text(articles: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{i}. {vancouver(a)}" for i, a in enumerate(articles, start=1)
    ) + "\n"

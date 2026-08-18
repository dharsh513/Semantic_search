"""
NCBI E-utilities client.

PubMed data is fetched straight from the public NCBI Entrez endpoints:

    ESpell   -> spelling suggestions for the raw user query
    ESearch  -> PMID list + NCBI's own query translation (exposes MeSH mapping)
    EFetch   -> full records (title, abstract, authors, journal, MeSH headings)

No API key is required. NCBI allows 3 requests/second anonymously, so every
call goes through a token-bucket rate limiter. An optional key can be set in
config to lift the limit to 10 req/s.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Optional

import requests

from config import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class _RateLimiter:
    """Simple thread-safe minimum-interval limiter."""

    def __init__(self, per_second: float):
        self._min_interval = 1.0 / max(per_second, 0.1)
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _text(node: Optional[ET.Element]) -> str:
    """Flatten an element and all of its descendants into plain text."""
    if node is None:
        return ""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "study", "studies", "paper", "papers", "article", "articles",
    "research", "please", "find", "show", "tell", "give", "explain", "what's",
    "latest", "recent", "role", "effect", "effects", "impact",
}


def keywords(text: str, limit: int = 12) -> List[str]:
    """Content words from a free-text query, order preserved, de-duplicated."""
    out, seen = [], set()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-']{1,}", text.lower()):
        if tok in STOPWORDS or len(tok) < 3 or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class PubMedError(RuntimeError):
    """Raised when NCBI cannot be reached or returns an unusable response."""


class PubMedClient:
    def __init__(self) -> None:
        self.base = config.NCBI_BASE_URL.rstrip("/")
        self.limiter = _RateLimiter(config.NCBI_RATE_LIMIT)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": f"{config.NCBI_TOOL}/1.0 (+python-requests)"}
        )

    # ------------------------------------------------------------------ #
    # low-level
    # ------------------------------------------------------------------ #
    def _common_params(self) -> Dict[str, str]:
        params = {"tool": config.NCBI_TOOL}
        if config.NCBI_EMAIL:
            params["email"] = config.NCBI_EMAIL
        if config.NCBI_API_KEY:
            params["api_key"] = config.NCBI_API_KEY
        return params

    def _get(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        url = f"{self.base}/{endpoint}"
        merged = {**self._common_params(), **params}
        last_err: Optional[Exception] = None

        for attempt in range(1, config.NCBI_MAX_RETRIES + 1):
            self.limiter.acquire()
            try:
                resp = self.session.get(
                    url, params=merged, timeout=config.NCBI_TIMEOUT
                )
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise PubMedError(f"NCBI HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001 - retry on any transport error
                last_err = exc
                log.warning("NCBI %s attempt %s/%s failed: %s",
                            endpoint, attempt, config.NCBI_MAX_RETRIES, exc)
                time.sleep(min(2 ** attempt * 0.4, 4.0))

        raise PubMedError(f"NCBI request to {endpoint} failed: {last_err}")

    # ------------------------------------------------------------------ #
    # ESpell
    # ------------------------------------------------------------------ #
    def espell(self, term: str) -> Optional[str]:
        """Return NCBI's spelling correction for `term`, or None."""
        if not term.strip():
            return None
        try:
            resp = self._get("espell.fcgi", {"db": "pubmed", "term": term})
            root = ET.fromstring(resp.content)
            corrected = _text(root.find("CorrectedQuery"))
            if corrected and corrected.lower() != term.lower():
                return corrected
        except Exception as exc:  # noqa: BLE001 - spelling help is best-effort
            log.debug("espell failed: %s", exc)
        return None

    # ------------------------------------------------------------------ #
    # ESearch
    # ------------------------------------------------------------------ #
    def esearch(
        self,
        term: str,
        retmax: Optional[int] = None,
        sort: str = "relevance",
        mindate: Optional[str] = None,
        maxdate: Optional[str] = None,
        field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a PubMed search.

        Returns a dict with: ids, count, query_translation, mesh_terms,
        translation_stack info and any NCBI warnings.
        """
        params: Dict[str, Any] = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": retmax or config.CANDIDATE_POOL,
            "sort": sort,
        }
        if field:
            params["field"] = field
        if mindate or maxdate:
            params["datetype"] = "pdat"
            params["mindate"] = mindate or "1800/01/01"
            params["maxdate"] = maxdate or "3000/12/31"

        payload = self._get("esearch.fcgi", params).json()
        result = payload.get("esearchresult", {}) or {}

        query_translation = result.get("querytranslation", "") or ""
        translation_set = result.get("translationset", []) or []

        return {
            "ids": [str(i) for i in result.get("idlist", []) or []],
            "count": int(result.get("count", 0) or 0),
            "query_translation": query_translation,
            "translations": [
                {"from": t.get("from", ""), "to": t.get("to", "")}
                for t in translation_set
            ],
            "mesh_terms": self._mesh_from_translation(query_translation),
            "warnings": result.get("warninglist", {}) or {},
            "errors": result.get("errorlist", {}) or {},
        }

    @staticmethod
    def _mesh_from_translation(translation: str) -> List[str]:
        """Pull the MeSH descriptors NCBI mapped the query onto."""
        found, seen = [], set()
        for match in re.findall(r'"([^"]+)"\[MeSH Terms\]', translation or ""):
            key = match.lower()
            if key not in seen:
                seen.add(key)
                found.append(match)
        return found

    # ------------------------------------------------------------------ #
    # EFetch
    # ------------------------------------------------------------------ #
    def efetch(self, pmids: Iterable[str]) -> List[Dict[str, Any]]:
        """Fetch full PubMed records for the given PMIDs."""
        pmids = [str(p).strip() for p in pmids if str(p).strip()]
        if not pmids:
            return []

        articles: List[Dict[str, Any]] = []
        batch = max(1, int(config.EFETCH_BATCH))
        for start in range(0, len(pmids), batch):
            chunk = pmids[start:start + batch]
            resp = self._get(
                "efetch.fcgi",
                {"db": "pubmed", "id": ",".join(chunk),
                 "retmode": "xml", "rettype": "abstract"},
            )
            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError as exc:
                raise PubMedError(f"Malformed XML from EFetch: {exc}") from exc
            for node in root.findall(".//PubmedArticle"):
                parsed = self._parse_article(node)
                if parsed:
                    articles.append(parsed)
        return articles

    # ------------------------------------------------------------------ #
    # XML parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_article(node: ET.Element) -> Optional[Dict[str, Any]]:
        pmid = _text(node.find(".//MedlineCitation/PMID"))
        if not pmid:
            return None

        article = node.find(".//MedlineCitation/Article")
        if article is None:
            return None

        title = _text(article.find("ArticleTitle"))

        # --- abstract (may be structured with Label attributes) ---
        parts: List[str] = []
        for seg in article.findall(".//Abstract/AbstractText"):
            body = _text(seg)
            if not body:
                continue
            label = (seg.get("Label") or seg.get("NlmCategory") or "").strip()
            parts.append(f"{label.title()}: {body}" if label else body)
        abstract = " ".join(parts).strip()

        # --- authors ---
        authors: List[str] = []
        for a in article.findall(".//AuthorList/Author"):
            last = _text(a.find("LastName"))
            initials = _text(a.find("Initials"))
            collective = _text(a.find("CollectiveName"))
            if last:
                authors.append(f"{last} {initials}".strip())
            elif collective:
                authors.append(collective)

        # --- journal / date ---
        journal = _text(article.find(".//Journal/ISOAbbreviation")) or _text(
            article.find(".//Journal/Title")
        )
        pub = article.find(".//Journal/JournalIssue/PubDate")
        year = _text(pub.find("Year")) if pub is not None else ""
        month = _text(pub.find("Month")) if pub is not None else ""
        if not year and pub is not None:
            medline = _text(pub.find("MedlineDate"))
            m = re.search(r"(\d{4})", medline)
            year = m.group(1) if m else ""
        pub_date = " ".join(x for x in (month, year) if x).strip()

        # --- identifiers ---
        doi = ""
        for ident in node.findall(".//ArticleIdList/ArticleId"):
            if ident.get("IdType") == "doi":
                doi = _text(ident)
                break

        # --- MeSH headings & keywords ---
        mesh = [
            _text(d)
            for d in node.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if _text(d)
        ]
        kws = [_text(k) for k in node.findall(".//KeywordList/Keyword") if _text(k)]

        pub_types = [
            _text(p) for p in article.findall(".//PublicationTypeList/PublicationType")
            if _text(p)
        ]

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "pub_date": pub_date,
            "year": year,
            "doi": doi,
            "mesh_terms": mesh,
            "keywords": kws,
            "publication_types": pub_types,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }


pubmed_client = PubMedClient()

"""
Research Landscape Analytics Engine.

Provides macro-level discovery and synthesis over large literature sets (1,000+ matches):
  1. Publication trends over time (by year)
  2. Semantic topic clustering with human-readable labels and representative papers
  3. Research methodology breakdown (Clinical Trial, Meta-Analysis, Deep Learning, etc.)
  4. MeSH heading & author keyword distributions
  5. Top publishing journals and frequent authors
  6. Macro-level research landscape summary
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from rag.embedder import embedder
from rag.pubmed_client import STOPWORDS

log = logging.getLogger(__name__)

# Known research methodology cues for classification
METHOD_PATTERNS = [
    (
        "Clinical Trial / RCT",
        re.compile(
            r"\b(randomized controlled trial|clinical trial|phase [1234]|placebo-controlled|double-blind|rct)\b",
            re.IGNORECASE,
        ),
        {"clinical trial", "randomized controlled trial", "clinical study"},
    ),
    (
        "Systematic Review / Meta-Analysis",
        re.compile(
            r"\b(systematic review|meta-analysis|prisma|cochrane|pooled analysis)\b",
            re.IGNORECASE,
        ),
        {"systematic review", "meta-analysis", "review literature as topic"},
    ),
    (
        "Deep Learning / AI",
        re.compile(
            r"\b(deep learning|neural network|transformer|convolutional|cnn|lstm|machine learning|artificial intelligence|reinforcement learning|bert|large language model|llm)\b",
            re.IGNORECASE,
        ),
        {"deep learning", "machine learning", "artificial intelligence", "neural networks, computer"},
    ),
    (
        "Cohort / Observational Study",
        re.compile(
            r"\b(cohort study|prospective study|retrospective study|case-control|cross-sectional|longitudinal|observational study|population-based)\b",
            re.IGNORECASE,
        ),
        {"cohort studies", "prospective studies", "retrospective studies", "case-control studies", "cross-sectional studies"},
    ),
    (
        "Comparative / Experimental Study",
        re.compile(
            r"\b(comparative study|in vitro|in vivo|animal model|murine|xenograft|biomarker analysis|benchmarking)\b",
            re.IGNORECASE,
        ),
        {"comparative study", "models, animal", "in vitro techniques"},
    ),
    (
        "Narrative Review / Perspective",
        re.compile(
            r"\b(review|overview|perspective|editorial|commentary|state of the art|current advances)\b",
            re.IGNORECASE,
        ),
        {"review", "editorial", "comment"},
    ),
]


def classify_methodology(article: Dict[str, Any]) -> str:
    """Classifies an article into a primary research methodology category."""
    pub_types = [p.lower() for p in article.get("publication_types", [])]
    mesh_terms = {m.lower() for m in article.get("mesh_terms", [])}
    text = " ".join([
        article.get("title", ""),
        article.get("abstract", "")[:500],
    ])

    for name, pattern, mesh_cues in METHOD_PATTERNS:
        # Check publication types
        if any(cue in " ".join(pub_types) for cue in mesh_cues):
            return name
        # Check MeSH headings
        if mesh_terms & mesh_cues:
            return name
        # Check text regex
        if pattern.search(text):
            return name

    return "General Experimental / Other"


def extract_topic_label(articles: List[Dict[str, Any]], query: str = "") -> str:
    """Derives a concise, readable topic name from a cluster of articles."""
    mesh_counts = Counter()
    kw_counts = Counter()
    word_counts = Counter()
    query_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9\-']+", query)} if query else set()

    for a in articles:
        for m in a.get("mesh_terms", []):
            if m.lower() not in {"humans", "male", "female", "adult", "middle aged", "aged", "animals", "treatment outcome"}:
                mesh_counts[m] += 1
        for kw in a.get("keywords", []):
            kw_counts[kw.title()] += 1
        for word in re.findall(r"[A-Za-z][A-Za-z0-9\-']{2,}", (a.get("title") or "").lower()):
            if word not in STOPWORDS and word not in query_tokens:
                word_counts[word.title()] += 1

    # Top candidates
    top_mesh = [m for m, _ in mesh_counts.most_common(2)]
    top_kw = [k for k, _ in kw_counts.most_common(2)]
    top_words = [w for w, _ in word_counts.most_common(2)]

    if top_kw and len(top_kw[0]) > 3:
        if len(top_kw) > 1 and top_kw[0] != top_kw[1]:
            return f"{top_kw[0]} & {top_kw[1]}"
        return top_kw[0]
    if top_mesh:
        if len(top_mesh) > 1 and top_mesh[0] != top_mesh[1]:
            return f"{top_mesh[0]} & {top_mesh[1]}"
        return top_mesh[0]
    if top_words:
        return " & ".join(top_words)

    return "Biomedical Literature Cluster"


def cluster_articles(
    articles: List[Dict[str, Any]],
    vectors: Optional[np.ndarray] = None,
    n_clusters: int = 5,
    query: str = "",
) -> List[Dict[str, Any]]:
    """
    Groups articles into semantic clusters using embeddings or TF-IDF.
    Returns cluster items with titles, counts, and representative papers.
    """
    if not articles:
        return []

    n = len(articles)
    k = max(1, min(n_clusters, n, 6))

    if n <= 2:
        return [{
            "id": 1,
            "title": extract_topic_label(articles, query) or "Primary Focus Area",
            "paper_count": n,
            "percentage": 100.0,
            "representative_pmids": [a["pmid"] for a in articles],
            "representative_papers": [
                {
                    "pmid": a["pmid"],
                    "title": a.get("title", ""),
                    "year": a.get("year", ""),
                    "journal": a.get("journal", ""),
                }
                for a in articles[:2]
            ],
            "top_terms": [m for m, _ in Counter(m for a in articles for m in a.get("mesh_terms", [])).most_common(4)],
        }]

    # Try KMeans clustering
    labels = np.zeros(n, dtype=int)
    centroids = None
    try:
        from sklearn.cluster import KMeans

        if vectors is None or len(vectors) != n or vectors.size == 0:
            texts = [f"{a.get('title', '')} {a.get('abstract', '')[:300]}" for a in articles]
            vectors = embedder.encode_documents(texts)

        if vectors.size and vectors.shape[0] == n:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
            labels = kmeans.fit_predict(vectors)
            centroids = kmeans.cluster_centers_
    except Exception as exc:
        log.warning("Clustering fallback to equal division: %s", exc)
        labels = np.array([i % k for i in range(n)])

    cluster_groups: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        cluster_groups[int(lab)].append(idx)

    clusters: List[Dict[str, Any]] = []
    for c_id, indices in cluster_groups.items():
        c_articles = [articles[i] for i in indices]
        count = len(c_articles)
        pct = round((count / n) * 100, 1)
        title = extract_topic_label(c_articles, query)

        # Pick representative papers (closest to centroid if available, else first few)
        rep_indices = indices[:3]
        if centroids is not None and vectors is not None and vectors.size:
            centroid = centroids[c_id]
            dists = np.linalg.norm(vectors[indices] - centroid, axis=1)
            sorted_idx = np.argsort(dists)
            rep_indices = [indices[i] for i in sorted_idx[:3]]

        rep_papers = [
            {
                "pmid": articles[i]["pmid"],
                "title": articles[i].get("title", ""),
                "year": articles[i].get("year", ""),
                "journal": articles[i].get("journal", ""),
            }
            for i in rep_indices
        ]

        top_terms = [
            term
            for term, _ in Counter(
                m for a in c_articles for m in (a.get("mesh_terms") or a.get("keywords") or [])
                if m.lower() not in {"humans", "male", "female", "adult", "animals"}
            ).most_common(5)
        ]

        clusters.append({
            "id": c_id + 1,
            "title": title,
            "paper_count": count,
            "percentage": pct,
            "representative_pmids": [articles[i]["pmid"] for i in rep_indices],
            "representative_papers": rep_papers,
            "top_terms": top_terms,
        })

    clusters.sort(key=lambda c: -c["paper_count"])
    # Renumber IDs sequentially
    for idx, c in enumerate(clusters, start=1):
        c["id"] = idx

    return clusters


class LandscapeEngine:
    """Generates a complete research landscape analysis for a literature set."""

    def build_landscape(
        self,
        articles: List[Dict[str, Any]],
        query: str = "",
        total_matches: Optional[int] = None,
        vectors: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if not articles:
            return {
                "total_papers": 0,
                "total_matches": total_matches or 0,
                "query": query,
                "year_range": {"min": "", "max": ""},
                "publication_trend": [],
                "topic_clusters": [],
                "methodologies": [],
                "top_mesh_terms": [],
                "top_keywords": [],
                "top_journals": [],
                "top_authors": [],
                "summary": "No articles available for landscape analysis.",
            }

        total_sample = len(articles)
        total_found = total_matches or total_sample

        # 1. Publication Trend by Year
        years = [a.get("year") for a in articles if a.get("year") and str(a.get("year")).isdigit()]
        year_counts = Counter(years)
        sorted_years = sorted(year_counts.keys())
        publication_trend = [{"year": str(y), "count": year_counts[y]} for y in sorted_years]

        min_year = sorted_years[0] if sorted_years else ""
        max_year = sorted_years[-1] if sorted_years else ""

        # 2. Topic Clusters
        topic_clusters = cluster_articles(articles, vectors=vectors, query=query)

        # 3. Methodologies
        method_counts = Counter()
        for a in articles:
            method_counts[classify_methodology(a)] += 1
        methodologies = [
            {
                "method": method,
                "count": count,
                "percentage": round((count / total_sample) * 100, 1),
            }
            for method, count in method_counts.most_common()
        ]

        # 4. MeSH Terms Distribution
        mesh_counter = Counter()
        for a in articles:
            for m in a.get("mesh_terms", []):
                if m.lower() not in {"humans", "male", "female", "animals", "adult", "middle aged", "aged"}:
                    mesh_counter[m] += 1
        top_mesh_terms = [
            {"term": term, "count": cnt, "percentage": round((cnt / total_sample) * 100, 1)}
            for term, cnt in mesh_counter.most_common(12)
        ]

        # 5. Author Keywords
        kw_counter = Counter()
        for a in articles:
            for kw in a.get("keywords", []):
                kw_counter[kw.strip()] += 1
        top_keywords = [
            {"keyword": kw, "count": cnt}
            for kw, cnt in kw_counter.most_common(12) if kw
        ]

        # 6. Top Journals
        journal_counter = Counter(a.get("journal") for a in articles if a.get("journal"))
        top_journals = [
            {"journal": j, "count": cnt}
            for j, cnt in journal_counter.most_common(8)
        ]

        # 7. Top Authors
        author_counter = Counter(
            author for a in articles for author in a.get("authors", []) if author
        )
        top_authors = [
            {"author": author, "count": cnt}
            for author, cnt in author_counter.most_common(8)
        ]

        # 8. Macro-level Landscape Summary
        summary_parts = [
            f"Analysis of {total_sample} retrieved studies across {min_year}–{max_year} (from {total_found:,} total PubMed matches)."
        ]
        if topic_clusters:
            top_areas = ", ".join(f"“{c['title']}” ({c['paper_count']} papers)" for c in topic_clusters[:3])
            summary_parts.append(f"Major research clusters include: {top_areas}.")
        if methodologies:
            primary_method = methodologies[0]
            summary_parts.append(
                f"Predominant study design: {primary_method['method']} ({primary_method['percentage']}%)."
            )

        return {
            "total_papers": total_sample,
            "total_matches": total_found,
            "query": query,
            "year_range": {"min": min_year, "max": max_year},
            "publication_trend": publication_trend,
            "topic_clusters": topic_clusters,
            "methodologies": methodologies,
            "top_mesh_terms": top_mesh_terms,
            "top_keywords": top_keywords,
            "top_journals": top_journals,
            "top_authors": top_authors,
            "summary": " ".join(summary_parts),
        }


landscape_engine = LandscapeEngine()

"""
Hybrid retriever.

Dense (semantic) similarity alone can drift away from exact biomedical
entities; BM25 alone is exactly the keyword brittleness this project sets out
to fix. Combining both, then diversifying with MMR, gives the best of each:

    final = w_dense * cosine(query, chunk) + w_lex * bm25_norm(query, chunk)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from config import config
from rag.pubmed_client import STOPWORDS
from rag.vector_index import VectorIndex, mmr

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")


def tokenize(text: str) -> List[str]:
    return [
        t for t in _TOKEN.findall((text or "").lower())
        if len(t) > 1 and t not in STOPWORDS
    ]


class BM25:
    """Okapi BM25 over an in-memory passage set."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in corpus]
        self.n = len(self.docs)
        self.lengths = np.array([len(d) or 1 for d in self.docs], dtype="float32")
        self.avgdl = float(self.lengths.mean()) if self.n else 1.0

        self.tf: List[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype="float32")
        if self.n == 0:
            return out
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counter in enumerate(self.tf):
                freq = counter.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.lengths[i] / self.avgdl
                )
                out[i] += idf * (freq * (self.k1 + 1)) / denom
        return out


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class HybridRetriever:
    """Ranks chunks for one query against a freshly built per-query index."""

    def __init__(self, chunks: List[Dict[str, Any]], vectors: np.ndarray):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25(texts)
        dim = int(vectors.shape[1]) if vectors.size else 1
        self.index = VectorIndex(dim, use_faiss=False)
        if len(chunks):
            self.index.add([c["chunk_id"] for c in chunks], vectors, chunks)

    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query_text: str,
        query_vec: np.ndarray,
        top_k_chunks: int = None,
        use_mmr: bool = True,
    ) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        top_k_chunks = int(top_k_chunks or config.TOP_K_CHUNKS)

        # --- dense ---
        dense = np.zeros(len(self.chunks), dtype="float32")
        if query_vec is not None and np.linalg.norm(query_vec) > 0:
            for row, score in self.index.search(query_vec, k=len(self.chunks)):
                dense[row] = score

        # --- lexical ---
        lexical = self.bm25.scores(query_text)

        # --- fusion ---
        combined = (
            config.DENSE_WEIGHT * _minmax(dense)
            + config.LEXICAL_WEIGHT * _minmax(lexical)
        )

        # Title+MeSH passages are dense keyword bags and score artificially high.
        # They stay in the pool (they are the only handle on abstract-less
        # records) but yield to real abstract evidence on a tie.
        title_mask = np.array(
            [1.0 if (c.get("section") or "").lower() == "title" else 0.0
             for c in self.chunks],
            dtype="float32",
        )
        combined = combined * (1.0 - config.TITLE_CHUNK_PENALTY * title_mask)

        order = np.argsort(-combined)
        pool = [int(i) for i in order[: max(top_k_chunks * 3, top_k_chunks)]]

        if use_mmr and len(pool) > 1 and self.index.backend == "numpy":
            vecs = self.index.vectors_for(pool)
            picked = mmr(
                query_vec, vecs, combined[pool],
                k=top_k_chunks, lambda_mult=config.MMR_LAMBDA,
            )
            pool = [pool[p] for p in picked]
        else:
            pool = pool[:top_k_chunks]

        results = []
        for rank, i in enumerate(pool, start=1):
            chunk = self.chunks[i]
            results.append(
                {
                    **chunk,
                    "rank": rank,
                    "score": round(float(combined[i]), 5),
                    "dense_score": round(float(dense[i]), 5),
                    "lexical_score": round(float(lexical[i]), 5),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    @staticmethod
    def rollup_documents(
        chunk_hits: List[Dict[str, Any]],
        articles: Dict[str, Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Collapse chunk-level hits into ranked documents.

        A document's score is its best passage plus a small bonus for having
        several independently-relevant passages.
        """
        top_k = int(top_k or config.TOP_K)
        buckets: Dict[str, Dict[str, Any]] = {}

        for hit in chunk_hits:
            pmid = hit["pmid"]
            bucket = buckets.setdefault(
                pmid, {"pmid": pmid, "best": 0.0, "passages": []}
            )
            bucket["best"] = max(bucket["best"], hit["score"])
            bucket["passages"].append(hit)

        docs: List[Dict[str, Any]] = []
        for pmid, bucket in buckets.items():
            article = articles.get(pmid)
            if not article:
                continue
            extra = sorted((p["score"] for p in bucket["passages"]), reverse=True)[1:3]
            score = bucket["best"] + 0.12 * sum(extra)
            passages = sorted(
                bucket["passages"], key=lambda p: -p["score"]
            )[:3]
            docs.append(
                {
                    "pmid": pmid,
                    "title": article.get("title", ""),
                    "abstract": article.get("abstract", ""),
                    "authors": article.get("authors", []),
                    "journal": article.get("journal", ""),
                    "pub_date": article.get("pub_date", ""),
                    "year": article.get("year", ""),
                    "doi": article.get("doi", ""),
                    "mesh_terms": article.get("mesh_terms", []),
                    "keywords": article.get("keywords", []),
                    "publication_types": article.get("publication_types", []),
                    "url": article.get(
                        "url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    ),
                    "score": round(float(score), 5),
                    "relevance": round(min(float(score), 1.0) * 100, 1),
                    "matched_passages": [
                        {
                            "section": p.get("section", ""),
                            "text": p["text"],
                            "score": p["score"],
                        }
                        for p in passages
                    ],
                }
            )

        docs.sort(key=lambda d: -d["score"])
        docs = [d for d in docs if d["score"] >= config.MIN_SCORE] or docs
        for rank, doc in enumerate(docs[:top_k], start=1):
            doc["rank"] = rank
        return docs[:top_k]

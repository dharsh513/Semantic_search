"""
End-to-end RAG pipeline.

    user query
        │
        ├─ 1. Query understanding ─ ESpell correction, intent/field detection,
        │                            MeSH-aware boolean expansion
        ├─ 2. Retrieval (sparse)  ─ ESearch against PubMed  -> candidate PMIDs
        ├─ 3. Ingestion           ─ EFetch (cache-first) -> SQLite -> chunks
        ├─ 4. Embedding           ─ local sentence-transformer vectors
        ├─ 5. Retrieval (dense)   ─ hybrid cosine + BM25, MMR diversification
        ├─ 6. Document rollup     ─ passage scores -> ranked articles
        └─ 7. Generation          ─ grounded, citation-bearing answer

Step 2 is what keeps the corpus current; steps 4–7 are what make the search
*semantic* rather than a keyword match.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

from config import config
from rag.chunker import chunk_articles
from rag.embedder import embedder
from rag.generator import generate
from rag.pubmed_client import PubMedError, keywords, pubmed_client
from rag.retriever import HybridRetriever
from rag.store import store

log = logging.getLogger(__name__)

# Queries that look like a paper title are searched against the [Title] field,
# exactly as the requirement describes for "exact title searches".
_TITLE_HINT = re.compile(r"^[\"“](.+)[\"”]$")


class SearchPipeline:
    # ------------------------------------------------------------------ #
    # 1. Query understanding
    # ------------------------------------------------------------------ #
    def understand(self, raw_query: str, search_field: str = "auto") -> Dict[str, Any]:
        query = (raw_query or "").strip()
        info: Dict[str, Any] = {
            "original": query,
            "corrected": None,
            "field": None,
            "terms": keywords(query),
            "notes": [],
        }

        quoted = _TITLE_HINT.match(query)
        if search_field == "title" or quoted:
            core = quoted.group(1) if quoted else query
            info["field"] = "title"
            info["pubmed_query"] = f'"{core}"[Title]'
            info["notes"].append(
                "Exact-title mode: matched against the [Title] field."
            )
            return info

        if search_field == "author":
            info["field"] = "author"
            info["pubmed_query"] = f"{query}[Author]"
            return info

        # Spelling help (best effort, never fatal).
        corrected = pubmed_client.espell(query) if config.ENABLE_ESPELL else None
        if corrected:
            info["corrected"] = corrected
            info["notes"].append(f'NCBI spelling suggestion: "{corrected}".')
            query = corrected
            info["terms"] = keywords(query)

        # Natural-language questions are stripped down to their content words
        # and OR-joined so PubMed's own ATM can map each onto MeSH.
        terms = info["terms"]
        if len(query.split()) > 6 and terms:
            core = " AND ".join(terms[:6])
            info["pubmed_query"] = core
            info["notes"].append(
                "Long natural-language query reduced to its content terms; "
                "PubMed's Automatic Term Mapping expands each onto MeSH."
            )
        else:
            info["pubmed_query"] = query or "medicine"

        return info

    # ------------------------------------------------------------------ #
    # 2. Sparse retrieval + progressive relaxation
    # ------------------------------------------------------------------ #
    def _esearch_with_fallback(
        self, understanding: Dict[str, Any], filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        attempts: List[str] = [understanding["pubmed_query"]]

        terms = understanding.get("terms") or []
        if len(terms) > 3:
            attempts.append(" AND ".join(terms[:3]))
        if len(terms) > 1:
            attempts.append(" OR ".join(terms[:6]))
        if terms:
            attempts.append(terms[0])

        last: Dict[str, Any] = {}
        for i, term in enumerate(attempts):
            try:
                res = pubmed_client.esearch(
                    term,
                    retmax=config.CANDIDATE_POOL,
                    sort=filters.get("sort", "relevance"),
                    mindate=filters.get("mindate"),
                    maxdate=filters.get("maxdate"),
                )
            except PubMedError:
                raise
            last = res
            last["used_query"] = term
            last["relaxation_step"] = i
            if res["ids"]:
                if i > 0:
                    understanding["notes"].append(
                        f'No hits for the strict query; broadened to "{term}".'
                    )
                return last
        return last

    # ------------------------------------------------------------------ #
    # 3. Ingestion (cache-first)
    # ------------------------------------------------------------------ #
    def ingest(self, pmids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not pmids:
            return {}
        fresh = store.fresh_pmids(pmids)
        missing = [p for p in pmids if p not in fresh]

        if missing:
            fetched = pubmed_client.efetch(missing)
            store.upsert_articles(fetched)
            for art in fetched:
                store.replace_chunks(art["pmid"], chunk_articles([art]))

        articles = store.get_articles(pmids)

        # Make sure everything has chunks (older cache rows may predate a
        # chunker change).
        have = {c["pmid"] for c in store.get_chunks(pmids)}
        for pmid, art in articles.items():
            if pmid not in have:
                store.replace_chunks(pmid, chunk_articles([art]))

        return articles

    # ------------------------------------------------------------------ #
    # 4-5. Embed + hybrid rank
    # ------------------------------------------------------------------ #
    def _embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, max(embedder.dim, 1)), dtype="float32")

        model = embedder.name
        cached_raw = store.load_embeddings(model, [c["chunk_id"] for c in chunks])

        vectors: List[Optional[np.ndarray]] = []
        to_encode: List[int] = []
        for i, chunk in enumerate(chunks):
            blob = cached_raw.get(chunk["chunk_id"])
            if blob:
                vec = np.frombuffer(blob, dtype="float32")
                if embedder.dim and vec.shape[0] == embedder.dim:
                    vectors.append(vec)
                    continue
            vectors.append(None)
            to_encode.append(i)

        if to_encode:
            fresh = embedder.encode_documents([chunks[i]["text"] for i in to_encode])
            for slot, i in enumerate(to_encode):
                vectors[i] = fresh[slot]
            store.save_embeddings(
                model, [(chunks[i]["chunk_id"], vectors[i]) for i in to_encode]
            )

        dim = max((v.shape[0] for v in vectors if v is not None), default=1)
        return np.vstack(
            [v if v is not None else np.zeros(dim, dtype="float32") for v in vectors]
        ).astype("float32")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def search(
        self,
        raw_query: str,
        top_k: Optional[int] = None,
        search_field: str = "auto",
        filters: Optional[Dict[str, Any]] = None,
        with_answer: bool = True,
        record_history: bool = True,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        filters = filters or {}
        top_k = int(top_k or config.TOP_K)

        if not (raw_query or "").strip():
            raise ValueError("Query must not be empty.")

        understanding = self.understand(raw_query, search_field)
        sparse = self._esearch_with_fallback(understanding, filters)
        pmids = sparse.get("ids", [])

        stages: Dict[str, Any] = {
            "pubmed_query": sparse.get("used_query", understanding["pubmed_query"]),
            "query_translation": sparse.get("query_translation", ""),
            "mesh_terms": sparse.get("mesh_terms", []),
            "translations": sparse.get("translations", []),
            "total_matches": sparse.get("count", 0),
            "candidates_fetched": len(pmids),
        }

        if not pmids:
            took = int((time.perf_counter() - started) * 1000)
            payload = {
                "query": raw_query,
                "understanding": understanding,
                "stages": {**stages, "chunks_indexed": 0, "embedding": embedder.info()},
                "results": [],
                "answer": {
                    "answer": "PubMed returned no records for this query.",
                    "citations": [],
                    "mode": "extractive",
                    "grounded": False,
                },
                "took_ms": took,
            }
            payload["search_id"] = (
                self._record_history(payload, search_field) if record_history else None
            )
            return payload

        articles = self.ingest(pmids)
        chunks = [c for c in store.get_chunks(pmids) if c.get("text")]
        vectors = self._embed_chunks(chunks)

        query_text = understanding.get("corrected") or raw_query
        query_vec = embedder.encode_query(query_text)

        retriever = HybridRetriever(chunks, vectors)
        chunk_hits = retriever.retrieve(
            query_text, query_vec, top_k_chunks=max(config.TOP_K_CHUNKS, top_k * 2)
        )
        docs = HybridRetriever.rollup_documents(chunk_hits, articles, top_k=top_k)

        answer = (
            generate(query_text, docs, query_vec)
            if with_answer
            else {"answer": "", "citations": [], "mode": "skipped", "grounded": False}
        )

        took = int((time.perf_counter() - started) * 1000)

        payload = {
            "query": raw_query,
            "understanding": understanding,
            "stages": {
                **stages,
                "chunks_indexed": len(chunks),
                "chunks_retrieved": len(chunk_hits),
                "embedding": embedder.info(),
                "retrieval": {
                    "dense_weight": config.DENSE_WEIGHT,
                    "lexical_weight": config.LEXICAL_WEIGHT,
                    "mmr_lambda": config.MMR_LAMBDA,
                },
            },
            "results": docs,
            "answer": answer,
            "took_ms": took,
        }
        payload["search_id"] = self._record_history(payload, search_field)
        return payload

    # ------------------------------------------------------------------ #
    @staticmethod
    def _record_history(payload: Dict[str, Any], field: str) -> Optional[int]:
        """
        Persist the search so it can be reopened later without re-querying NCBI.

        History must never be able to break a search, so any storage failure is
        swallowed and the search still returns.
        """
        try:
            stages = payload.get("stages", {}) or {}
            return store.save_search(
                {
                    "query": payload.get("query", ""),
                    "translated": stages.get("pubmed_query", ""),
                    "field": field,
                    "n_results": len(payload.get("results", [])),
                    "total_hits": int(stages.get("total_matches", 0) or 0),
                    "took_ms": int(payload.get("took_ms", 0) or 0),
                    "pmids": [d["pmid"] for d in payload.get("results", [])],
                    "mesh_terms": stages.get("mesh_terms", []),
                    "snapshot": {
                        "understanding": payload.get("understanding", {}),
                        "stages": stages,
                        "results": payload.get("results", []),
                        "answer": payload.get("answer", {}),
                        "took_ms": payload.get("took_ms", 0),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not write search history: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    def article(self, pmid: str) -> Optional[Dict[str, Any]]:
        cached = store.get_articles([pmid]).get(str(pmid))
        if cached and (cached.get("abstract") or cached.get("title")):
            return cached
        fetched = pubmed_client.efetch([pmid])
        if not fetched:
            return None
        store.upsert_articles(fetched)
        store.replace_chunks(fetched[0]["pmid"], chunk_articles([fetched[0]]))
        return fetched[0]

    # ------------------------------------------------------------------ #
    def article_batch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch several records at once, caching and chunking each."""
        pmids = [str(p) for p in pmids if str(p).strip()]
        if not pmids:
            return []
        fetched = pubmed_client.efetch(pmids)
        if fetched:
            store.upsert_articles(fetched)
            for art in fetched:
                store.replace_chunks(art["pmid"], chunk_articles([art]))
        return fetched

    # ------------------------------------------------------------------ #
    def similar(self, pmid: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """'More like this' — semantic neighbours of one article."""
        art = self.article(pmid)
        if not art:
            return []
        seed = " ".join(
            [art.get("title", ""), " ".join((art.get("mesh_terms") or [])[:8])]
        ).strip()
        res = self.search(seed or art.get("title", ""), top_k=top_k + 1,
                          with_answer=False, record_history=False)
        return [d for d in res["results"] if d["pmid"] != str(pmid)][:top_k]


pipeline = SearchPipeline()

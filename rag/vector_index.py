"""
Vector index.

The working set for a single query is small (tens to a few hundred passages),
so an exact numpy cosine search is both faster and simpler than an ANN index.
If `faiss` is installed it is used automatically for the persistent corpus
index, which matters once the local cache grows to tens of thousands of chunks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:  # optional dependency
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore
    _HAS_FAISS = False


class VectorIndex:
    """Cosine-similarity index over L2-normalised vectors."""

    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        self.dim = int(dim)
        self.ids: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self._faiss = None
        if use_faiss and _HAS_FAISS and self.dim > 0:
            self._faiss = faiss.IndexFlatIP(self.dim)

    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def __len__(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------------ #
    def add(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        if len(ids) == 0:
            return
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(ids):
            raise ValueError("ids and vectors length mismatch")

        self.ids.extend(str(i) for i in ids)
        self.meta.extend(metadata or [{} for _ in ids])

        if self._faiss is not None:
            self._faiss.add(vectors)
        self._matrix = (
            vectors if self._matrix is None
            else np.vstack([self._matrix, vectors])
        )

    # ------------------------------------------------------------------ #
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Return [(row_index, cosine_score)] sorted by descending score."""
        if self._matrix is None or len(self.ids) == 0:
            return []
        q = np.asarray(query, dtype="float32").reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        k = int(min(max(k, 1), len(self.ids)))

        if self._faiss is not None:
            scores, idx = self._faiss.search(np.ascontiguousarray(q), k)
            return [
                (int(i), float(s))
                for i, s in zip(idx[0], scores[0]) if i >= 0
            ]

        sims = (self._matrix @ q.T).ravel()
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    # ------------------------------------------------------------------ #
    def vectors_for(self, rows: Sequence[int]) -> np.ndarray:
        if self._matrix is None or not len(rows):
            return np.zeros((0, self.dim), dtype="float32")
        return self._matrix[np.asarray(list(rows), dtype=int)]

    def entry(self, row: int) -> Dict[str, Any]:
        return {"id": self.ids[row], **(self.meta[row] or {})}

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        if self._matrix is None:
            return
        import json

        np.savez_compressed(
            path,
            matrix=self._matrix,
            ids=np.array(self.ids, dtype=object),
            meta=np.array([json.dumps(m) for m in self.meta], dtype=object),
            dim=np.array([self.dim]),
        )

    @classmethod
    def load(cls, path: str) -> Optional["VectorIndex"]:
        import json
        import os

        if not os.path.exists(path):
            return None
        try:
            data = np.load(path, allow_pickle=True)
            idx = cls(int(data["dim"][0]))
            idx.add(
                [str(i) for i in data["ids"].tolist()],
                data["matrix"],
                [json.loads(m) for m in data["meta"].tolist()],
            )
            return idx
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load index at %s: %s", path, exc)
            return None


def mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    scores: Sequence[float],
    k: int,
    lambda_mult: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance.

    Balances relevance against redundancy so the answer is not built from five
    near-identical passages of the same paper.
    """
    n = len(scores)
    if n == 0:
        return []
    k = int(min(k, n))
    doc_vecs = np.asarray(doc_vecs, dtype="float32")
    scores = np.asarray(scores, dtype="float32")

    selected: List[int] = []
    remaining = list(range(n))

    while remaining and len(selected) < k:
        if not selected:
            best = int(remaining[int(np.argmax(scores[remaining]))])
        else:
            sel_mat = doc_vecs[selected]
            cand = np.asarray(remaining, dtype=int)
            redundancy = (doc_vecs[cand] @ sel_mat.T).max(axis=1)
            mmr_scores = lambda_mult * scores[cand] - (1 - lambda_mult) * redundancy
            best = int(cand[int(np.argmax(mmr_scores))])
        selected.append(best)
        remaining.remove(best)

    return selected

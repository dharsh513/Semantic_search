"""
Embeddings.

Primary backend  : sentence-transformers (`all-MiniLM-L6-v2`, 384-d).
                   Runs entirely on the local machine, no API key, no network
                   after the first model download.
Fallback backend : scikit-learn TF-IDF + TruncatedSVD, used automatically if
                   sentence-transformers is unavailable or the model cannot be
                   downloaded. The rest of the pipeline is unchanged.

Vectors are L2-normalised, so a dot product is cosine similarity.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Sequence

import numpy as np

from config import config

log = logging.getLogger(__name__)


class BaseEmbedder:
    name: str = "base"
    dim: int = 0
    kind: str = "base"

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float32")
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder(BaseEmbedder):
    kind = "sentence-transformer"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # local import

        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        with self._lock:
            vecs = self.model.encode(
                list(texts),
                batch_size=int(config.EMBEDDING_BATCH),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vecs, dtype="float32")


class TfidfEmbedder(BaseEmbedder):
    """
    Deterministic lexical-semantic fallback.

    Fitted lazily on the first non-query corpus it sees, then reused. Uses
    character + word n-grams so it degrades gracefully on biomedical jargon.
    """

    kind = "tfidf-svd"

    def __init__(self, dim: int = 256) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name = "tfidf-svd"
        self.dim = dim
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=60_000,
        )
        self._svd_cls = TruncatedSVD
        self._svd = None
        self._fitted = False
        self._lock = threading.Lock()

    def fit(self, corpus: Sequence[str]) -> None:
        with self._lock:
            corpus = [c for c in corpus if c and c.strip()]
            if len(corpus) < 2:
                corpus = list(corpus) + ["placeholder biomedical text"] * 2
            tfidf = self._vectorizer.fit_transform(corpus)
            n_comp = int(min(self.dim, max(2, min(tfidf.shape) - 1)))
            self._svd = self._svd_cls(n_components=n_comp, random_state=42)
            self._svd.fit(tfidf)
            self.dim = n_comp
            self._fitted = True

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, max(self.dim, 1)), dtype="float32")
        if not self._fitted:
            if is_query:
                # Nothing indexed yet — return a zero vector of the right shape.
                return np.zeros((len(texts), max(self.dim, 1)), dtype="float32")
            self.fit(texts)
        tfidf = self._vectorizer.transform(list(texts))
        return _normalize(self._svd.transform(tfidf))


class Embedder:
    """Facade that picks the best available backend once, then delegates."""

    def __init__(self) -> None:
        self._backend: Optional[BaseEmbedder] = None
        self._lock = threading.Lock()

    @property
    def backend(self) -> BaseEmbedder:
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    self._backend = self._build()
        return self._backend

    def _build(self) -> BaseEmbedder:
        try:
            log.info("Loading embedding model %s ...", config.EMBEDDING_MODEL)
            be = SentenceTransformerEmbedder(config.EMBEDDING_MODEL)
            log.info("Embedding backend ready: %s (dim=%d)", be.name, be.dim)
            return be
        except Exception as exc:  # noqa: BLE001
            if not config.ALLOW_TFIDF_FALLBACK:
                raise
            log.warning(
                "sentence-transformers unavailable (%s). "
                "Falling back to TF-IDF+SVD embeddings.", exc
            )
            return TfidfEmbedder()

    # -------------------------------------------------------------- #
    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def kind(self) -> str:
        return self.backend.kind

    @property
    def dim(self) -> int:
        return self.backend.dim

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.backend.encode(texts, is_query=False)

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.backend.encode([text], is_query=True)
        return vec[0] if len(vec) else np.zeros(max(self.dim, 1), dtype="float32")

    def info(self) -> dict:
        return {"model": self.name, "kind": self.kind, "dim": self.dim}


embedder = Embedder()

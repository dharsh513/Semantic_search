"""
Central configuration for the PubMed Semantic Search (RAG) project.

Every value can be overridden with an environment variable or a `.env` file
placed next to this module.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except Exception:  # python-dotenv is optional
    pass


def _env(key: str, default):
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            return default
    return raw


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(_env("RAG_DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    # ---------------- Flask ----------------
    HOST = _env("FLASK_HOST", "127.0.0.1")
    PORT = _env("FLASK_PORT", 5000)
    DEBUG = _env("FLASK_DEBUG", True)
    SECRET_KEY = _env("SECRET_KEY", "pubmed-rag-dev-key")

    # Flask's auto-reloader watches every module in sys.modules. Importing
    # sentence-transformers pulls in hundreds of `transformers` submodules
    # LAZILY — i.e. after the server has already started — and the reloader
    # reads those as "files changed" and restarts the server mid-request.
    # The visible symptom is the engine badge flipping to "offline" and
    # searches failing at random. Off by default; the exclude patterns below
    # make it safe if you switch it back on.
    USE_RELOADER = _env("USE_RELOADER", False)
    # Load the embedding model at boot instead of during the first search.
    PRELOAD_EMBEDDER = _env("PRELOAD_EMBEDDER", True)

    # ---------------- Authentication ----------------
    SESSION_COOKIE = _env("SESSION_COOKIE", "pubmed_rag_session")
    SESSION_DAYS = _env("SESSION_DAYS", 7)              # normal sign-in
    SESSION_REMEMBER_DAYS = _env("SESSION_REMEMBER_DAYS", 30)  # "keep me signed in"
    # Set true only when serving over HTTPS — a secure cookie is never sent
    # over plain http, which would silently break local development.
    SESSION_COOKIE_SECURE = _env("SESSION_COOKIE_SECURE", False)
    # Set false to close signups once your accounts exist.
    ALLOW_SIGNUP = _env("ALLOW_SIGNUP", True)

    # ---------------- NCBI E-utilities ----------------
    # No API key is required. Supplying one only raises the rate limit
    # from 3 req/s to 10 req/s.
    NCBI_BASE_URL = _env(
        "NCBI_BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    )
    NCBI_API_KEY = _env("NCBI_API_KEY", "")  # optional, blank = anonymous access
    NCBI_TOOL = _env("NCBI_TOOL", "pubmed-semantic-rag")
    NCBI_EMAIL = _env("NCBI_EMAIL", "")  # optional, NCBI politeness header
    NCBI_TIMEOUT = _env("NCBI_TIMEOUT", 30)
    NCBI_MAX_RETRIES = _env("NCBI_MAX_RETRIES", 3)
    # Requests per second. NCBI allows 3/s anonymously; stay a little under.
    NCBI_RATE_LIMIT = _env("NCBI_RATE_LIMIT", 2.5)
    # How many PubMed records to pull as the candidate pool before reranking.
    CANDIDATE_POOL = _env("CANDIDATE_POOL", 60)
    EFETCH_BATCH = _env("EFETCH_BATCH", 100)
    # ESpell costs one extra round trip per query; set false to skip it.
    ENABLE_ESPELL = _env("ENABLE_ESPELL", True)

    # ---------------- Storage ----------------
    DB_PATH = str(DATA_DIR / _env("DB_NAME", "pubmed_cache.sqlite3"))
    INDEX_PATH = str(DATA_DIR / "vector_index.npz")
    CACHE_TTL_DAYS = _env("CACHE_TTL_DAYS", 30)

    # ---------------- Embeddings ----------------
    EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_BATCH = _env("EMBEDDING_BATCH", 32)
    # If the transformer model cannot be loaded (offline / not installed) the
    # pipeline silently degrades to a TF-IDF vectoriser instead of crashing.
    ALLOW_TFIDF_FALLBACK = _env("ALLOW_TFIDF_FALLBACK", True)

    # ---------------- Chunking ----------------
    CHUNK_WORDS = _env("CHUNK_WORDS", 110)
    CHUNK_OVERLAP_WORDS = _env("CHUNK_OVERLAP_WORDS", 30)
    MIN_CHUNK_WORDS = _env("MIN_CHUNK_WORDS", 12)

    # ---------------- Retrieval ----------------
    TOP_K = _env("TOP_K", 10)             # documents returned to the UI
    TOP_K_CHUNKS = _env("TOP_K_CHUNKS", 24)  # chunks pulled before doc rollup
    DENSE_WEIGHT = _env("DENSE_WEIGHT", 0.72)   # semantic similarity weight
    LEXICAL_WEIGHT = _env("LEXICAL_WEIGHT", 0.28)  # BM25 weight
    MMR_LAMBDA = _env("MMR_LAMBDA", 0.72)   # 1.0 = pure relevance, 0.0 = pure diversity
    MIN_SCORE = _env("MIN_SCORE", 0.05)
    # Title+MeSH passages are keyword-dense and over-score; damp them so real
    # abstract evidence wins ties. 0.0 disables the penalty.
    TITLE_CHUNK_PENALTY = _env("TITLE_CHUNK_PENALTY", 0.18)

    # ---------------- Generation ----------------
    ANSWER_SENTENCES = _env("ANSWER_SENTENCES", 6)
    ANSWER_MAX_SOURCES = _env("ANSWER_MAX_SOURCES", 5)


config = Config()

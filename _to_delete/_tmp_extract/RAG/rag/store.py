"""
SQLite cache for PubMed records.

Every article fetched from NCBI is persisted locally so that:
  * repeat searches do not re-hit E-utilities (respectful of NCBI limits),
  * the corpus grows into a reusable offline knowledge base for the RAG index,
  * embeddings can be cached alongside the text they were computed from.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from config import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    pmid              TEXT PRIMARY KEY,
    title             TEXT NOT NULL DEFAULT '',
    abstract          TEXT NOT NULL DEFAULT '',
    authors           TEXT NOT NULL DEFAULT '[]',
    journal           TEXT NOT NULL DEFAULT '',
    pub_date          TEXT NOT NULL DEFAULT '',
    year              TEXT NOT NULL DEFAULT '',
    doi               TEXT NOT NULL DEFAULT '',
    mesh_terms        TEXT NOT NULL DEFAULT '[]',
    keywords          TEXT NOT NULL DEFAULT '[]',
    publication_types TEXT NOT NULL DEFAULT '[]',
    url               TEXT NOT NULL DEFAULT '',
    fetched_at        REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    pmid       TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,
    section    TEXT NOT NULL DEFAULT '',
    text       TEXT NOT NULL,
    FOREIGN KEY (pmid) REFERENCES articles(pmid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id  TEXT PRIMARY KEY,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS query_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT NOT NULL,
    translated  TEXT NOT NULL DEFAULT '',
    n_results   INTEGER NOT NULL DEFAULT 0,
    took_ms     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_pmid ON chunks(pmid);
CREATE INDEX IF NOT EXISTS idx_articles_year ON articles(year);
"""

_JSON_FIELDS = ("authors", "mesh_terms", "keywords", "publication_types")


class Store:
    """Thread-safe thin wrapper around a SQLite file."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or config.DB_PATH
        self._lock = threading.RLock()
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------ #
    # articles
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_article(row: sqlite3.Row) -> Dict[str, Any]:
        art = dict(row)
        for field in _JSON_FIELDS:
            try:
                art[field] = json.loads(art.get(field) or "[]")
            except (TypeError, ValueError):
                art[field] = []
        return art

    def upsert_articles(self, articles: Iterable[Dict[str, Any]]) -> int:
        rows = []
        now = time.time()
        for a in articles:
            if not a.get("pmid"):
                continue
            rows.append(
                (
                    a["pmid"], a.get("title", ""), a.get("abstract", ""),
                    json.dumps(a.get("authors", [])), a.get("journal", ""),
                    a.get("pub_date", ""), a.get("year", ""), a.get("doi", ""),
                    json.dumps(a.get("mesh_terms", [])),
                    json.dumps(a.get("keywords", [])),
                    json.dumps(a.get("publication_types", [])),
                    a.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/"),
                    now,
                )
            )
        if not rows:
            return 0
        with self._lock:
            conn = self._conn()
            conn.executemany(
                """
                INSERT INTO articles
                    (pmid, title, abstract, authors, journal, pub_date, year,
                     doi, mesh_terms, keywords, publication_types, url, fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pmid) DO UPDATE SET
                    title=excluded.title,
                    abstract=excluded.abstract,
                    authors=excluded.authors,
                    journal=excluded.journal,
                    pub_date=excluded.pub_date,
                    year=excluded.year,
                    doi=excluded.doi,
                    mesh_terms=excluded.mesh_terms,
                    keywords=excluded.keywords,
                    publication_types=excluded.publication_types,
                    url=excluded.url,
                    fetched_at=excluded.fetched_at
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def get_articles(self, pmids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        pmids = [str(p) for p in pmids]
        if not pmids:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        conn = self._conn()
        for start in range(0, len(pmids), 400):
            batch = pmids[start:start + 400]
            marks = ",".join("?" * len(batch))
            for row in conn.execute(
                f"SELECT * FROM articles WHERE pmid IN ({marks})", batch
            ):
                art = self._row_to_article(row)
                out[art["pmid"]] = art
        return out

    def fresh_pmids(self, pmids: Iterable[str]) -> set:
        """PMIDs already cached and still inside the TTL window."""
        cutoff = time.time() - config.CACHE_TTL_DAYS * 86400
        cached = self.get_articles(pmids)
        return {
            p for p, a in cached.items()
            if a.get("fetched_at", 0) >= cutoff and (a.get("abstract") or a.get("title"))
        }

    def all_articles(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM articles ORDER BY fetched_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._row_to_article(r) for r in self._conn().execute(sql)]

    # ------------------------------------------------------------------ #
    # chunks
    # ------------------------------------------------------------------ #
    def replace_chunks(self, pmid: str, chunks: List[Dict[str, Any]]) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute("DELETE FROM chunks WHERE pmid = ?", (pmid,))
            conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id, pmid, ordinal, section, text) VALUES (?,?,?,?,?)",
                [
                    (c["chunk_id"], pmid, c["ordinal"], c.get("section", ""), c["text"])
                    for c in chunks
                ],
            )
            conn.commit()

    def get_chunks(self, pmids: Iterable[str]) -> List[Dict[str, Any]]:
        pmids = [str(p) for p in pmids]
        if not pmids:
            return []
        rows: List[Dict[str, Any]] = []
        conn = self._conn()
        for start in range(0, len(pmids), 400):
            batch = pmids[start:start + 400]
            marks = ",".join("?" * len(batch))
            rows.extend(
                dict(r) for r in conn.execute(
                    f"SELECT * FROM chunks WHERE pmid IN ({marks}) "
                    f"ORDER BY pmid, ordinal", batch
                )
            )
        return rows

    # ------------------------------------------------------------------ #
    # embeddings
    # ------------------------------------------------------------------ #
    def save_embeddings(self, model: str, items: Iterable) -> None:
        """items: iterable of (chunk_id, numpy_vector)."""
        rows = [
            (cid, model, int(vec.shape[0]), vec.astype("float32").tobytes())
            for cid, vec in items
        ]
        if not rows:
            return
        with self._lock:
            conn = self._conn()
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (chunk_id, model, dim, vector) "
                "VALUES (?,?,?,?)",
                rows,
            )
            conn.commit()

    def load_embeddings(self, model: str, chunk_ids: Iterable[str]) -> Dict[str, bytes]:
        chunk_ids = [str(c) for c in chunk_ids]
        if not chunk_ids:
            return {}
        out: Dict[str, bytes] = {}
        conn = self._conn()
        for start in range(0, len(chunk_ids), 400):
            batch = chunk_ids[start:start + 400]
            marks = ",".join("?" * len(batch))
            for row in conn.execute(
                f"SELECT chunk_id, vector FROM embeddings "
                f"WHERE model = ? AND chunk_id IN ({marks})",
                [model, *batch],
            ):
                out[row["chunk_id"]] = row["vector"]
        return out

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #
    def log_query(self, query: str, translated: str, n: int, took_ms: int) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO query_log (query, translated, n_results, took_ms, created_at) "
                "VALUES (?,?,?,?,?)",
                (query, translated, n, took_ms, time.time()),
            )
            conn.commit()

    def recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        return [
            dict(r) for r in self._conn().execute(
                "SELECT query, n_results, took_ms, created_at FROM query_log "
                "ORDER BY id DESC LIMIT ?", (limit,)
            )
        ]

    def stats(self) -> Dict[str, Any]:
        conn = self._conn()
        one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "articles": one("SELECT COUNT(*) FROM articles"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "embeddings": one("SELECT COUNT(*) FROM embeddings"),
            "queries": one("SELECT COUNT(*) FROM query_log"),
            "db_path": self.path,
        }


store = Store()

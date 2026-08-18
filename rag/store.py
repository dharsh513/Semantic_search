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
    volume            TEXT NOT NULL DEFAULT '',
    issue             TEXT NOT NULL DEFAULT '',
    pages             TEXT NOT NULL DEFAULT '',
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

-- Accounts. Passwords are never stored — only a PBKDF2-SHA256 digest and its
-- per-user salt (see rag/auth.py).
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL,
    last_login_at REAL NOT NULL DEFAULT 0,
    is_active     INTEGER NOT NULL DEFAULT 1
);

-- Server-side sessions. The cookie carries only an opaque random token; every
-- piece of state lives here, so logging out really does revoke access.
CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    user_agent TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Full search history. Each row stores enough to restore the result set in
-- the UI without hitting NCBI again, and to drive citation export.
CREATE TABLE IF NOT EXISTS searches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL DEFAULT 0,
    query       TEXT NOT NULL,
    translated  TEXT NOT NULL DEFAULT '',
    field       TEXT NOT NULL DEFAULT 'auto',
    n_results   INTEGER NOT NULL DEFAULT 0,
    total_hits  INTEGER NOT NULL DEFAULT 0,
    took_ms     INTEGER NOT NULL DEFAULT 0,
    pinned      INTEGER NOT NULL DEFAULT 0,
    pmids       TEXT NOT NULL DEFAULT '[]',
    mesh_terms  TEXT NOT NULL DEFAULT '[]',
    snapshot    TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);

-- Per-paper chat transcripts.
CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL DEFAULT 0,
    pmid        TEXT NOT NULL,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    evidence    TEXT NOT NULL DEFAULT '[]',
    confidence  REAL NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_pmid ON chunks(pmid);
CREATE INDEX IF NOT EXISTS idx_articles_year ON articles(year);
CREATE INDEX IF NOT EXISTS idx_searches_created ON searches(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_pmid ON chat_messages(user_id, pmid, id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);
"""

# Columns added after earlier releases. Applied to existing databases on
# startup so an older data/pubmed_cache.sqlite3 keeps working instead of
# throwing. Adding a column is safe and idempotent; existing rows take the
# DEFAULT, and user_id 0 means "created before accounts existed".
_MIGRATIONS = {
    "articles": [
        ("volume", "TEXT NOT NULL DEFAULT ''"),
        ("issue", "TEXT NOT NULL DEFAULT ''"),
        ("pages", "TEXT NOT NULL DEFAULT ''"),
    ],
    "searches": [
        ("user_id", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "chat_messages": [
        ("user_id", "INTEGER NOT NULL DEFAULT 0"),
    ],
}

_JSON_FIELDS = ("authors", "mesh_terms", "keywords", "publication_types")


class Store:
    """Thread-safe thin wrapper around a SQLite file."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or config.DB_PATH
        self._lock = threading.RLock()
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
        self._migrate()

    # ------------------------------------------------------------------ #
    def _migrate(self) -> None:
        """Add columns introduced after the initial release."""
        conn = self._conn()
        with self._lock:
            for table, columns in _MIGRATIONS.items():
                existing = {
                    row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
                }
                for name, ddl in columns:
                    if name not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            conn.commit()

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
                    a.get("pub_date", ""), a.get("year", ""),
                    a.get("volume", ""), a.get("issue", ""), a.get("pages", ""),
                    a.get("doi", ""),
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
                     volume, issue, pages, doi, mesh_terms, keywords,
                     publication_types, url, fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pmid) DO UPDATE SET
                    title=excluded.title,
                    abstract=excluded.abstract,
                    authors=excluded.authors,
                    journal=excluded.journal,
                    pub_date=excluded.pub_date,
                    year=excluded.year,
                    volume=excluded.volume,
                    issue=excluded.issue,
                    pages=excluded.pages,
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
    def save_search(self, record: Dict[str, Any]) -> int:
        """
        Persist a completed search. `snapshot` holds everything the UI needs to
        redisplay the result set later without another NCBI round trip.
        """
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                """
                INSERT INTO searches
                    (user_id, query, translated, field, n_results, total_hits,
                     took_ms, pmids, mesh_terms, snapshot, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    int(record.get("user_id", 0) or 0),
                    record.get("query", ""),
                    record.get("translated", ""),
                    record.get("field", "auto"),
                    int(record.get("n_results", 0)),
                    int(record.get("total_hits", 0)),
                    int(record.get("took_ms", 0)),
                    json.dumps(record.get("pmids", [])),
                    json.dumps(record.get("mesh_terms", [])),
                    json.dumps(record.get("snapshot", {})),
                    time.time(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def history(self, user_id: int, limit: int = 50, offset: int = 0,
                search: str = "") -> List[Dict[str, Any]]:
        """
        One user's recent searches, newest first. `snapshot` is omitted for
        lightness — call get_search() to restore a specific one.
        """
        sql = (
            "SELECT id, query, translated, field, n_results, total_hits, took_ms, "
            "pinned, pmids, mesh_terms, created_at FROM searches WHERE user_id = ? "
        )
        params: List[Any] = [int(user_id)]
        if search.strip():
            sql += "AND query LIKE ? "
            params.append(f"%{search.strip()}%")
        sql += "ORDER BY pinned DESC, id DESC LIMIT ? OFFSET ?"
        params += [int(limit), int(offset)]

        out = []
        for row in self._conn().execute(sql, params):
            item = dict(row)
            for field in ("pmids", "mesh_terms"):
                try:
                    item[field] = json.loads(item[field] or "[]")
                except (TypeError, ValueError):
                    item[field] = []
            out.append(item)
        return out

    def get_search(self, search_id: int,
                   user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        One saved search. When `user_id` is given the row must belong to that
        user — a mismatch returns None, which the API surfaces as 404 so one
        account cannot probe another's history by guessing ids.
        """
        if user_id is None:
            row = self._conn().execute(
                "SELECT * FROM searches WHERE id = ?", (int(search_id),)
            ).fetchone()
        else:
            row = self._conn().execute(
                "SELECT * FROM searches WHERE id = ? AND user_id = ?",
                (int(search_id), int(user_id)),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        for field in ("pmids", "mesh_terms"):
            try:
                item[field] = json.loads(item[field] or "[]")
            except (TypeError, ValueError):
                item[field] = []
        try:
            item["snapshot"] = json.loads(item["snapshot"] or "{}")
        except (TypeError, ValueError):
            item["snapshot"] = {}
        return item

    def delete_search(self, search_id: int, user_id: int) -> bool:
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "DELETE FROM searches WHERE id = ? AND user_id = ?",
                (int(search_id), int(user_id)),
            )
            conn.commit()
            return cur.rowcount > 0

    def clear_history(self, user_id: int, keep_pinned: bool = True) -> int:
        with self._lock:
            conn = self._conn()
            sql = "DELETE FROM searches WHERE user_id = ?"
            if keep_pinned:
                sql += " AND pinned = 0"
            cur = conn.execute(sql, (int(user_id),))
            conn.commit()
            return cur.rowcount

    def set_pinned(self, search_id: int, pinned: bool, user_id: int) -> bool:
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "UPDATE searches SET pinned = ? WHERE id = ? AND user_id = ?",
                (1 if pinned else 0, int(search_id), int(user_id)),
            )
            conn.commit()
            return cur.rowcount > 0

    def recent_queries(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        return [
            dict(r) for r in self._conn().execute(
                "SELECT query, n_results, took_ms, created_at FROM searches "
                "WHERE user_id = ? ORDER BY id DESC LIMIT ?", (int(user_id), limit)
            )
        ]

    # ------------------------------------------------------------------ #
    # chat
    # ------------------------------------------------------------------ #
    def add_chat_message(
        self,
        user_id: int,
        pmid: str,
        role: str,
        text: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.0,
    ) -> int:
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "INSERT INTO chat_messages (user_id, pmid, role, text, evidence, "
                "confidence, created_at) VALUES (?,?,?,?,?,?,?)",
                (int(user_id), str(pmid), role, text, json.dumps(evidence or []),
                 float(confidence), time.time()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def chat_history(self, user_id: int, pmid: str,
                     limit: int = 100) -> List[Dict[str, Any]]:
        out = []
        for row in self._conn().execute(
            "SELECT id, role, text, evidence, confidence, created_at "
            "FROM chat_messages WHERE user_id = ? AND pmid = ? ORDER BY id LIMIT ?",
            (int(user_id), str(pmid), int(limit)),
        ):
            item = dict(row)
            try:
                item["evidence"] = json.loads(item["evidence"] or "[]")
            except (TypeError, ValueError):
                item["evidence"] = []
            out.append(item)
        return out

    def clear_chat(self, user_id: int, pmid: str) -> int:
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "DELETE FROM chat_messages WHERE user_id = ? AND pmid = ?",
                (int(user_id), str(pmid)),
            )
            conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------ #
    # users
    # ------------------------------------------------------------------ #
    @staticmethod
    def _public_user(row: sqlite3.Row) -> Dict[str, Any]:
        """A user record with the password digest stripped out."""
        user = dict(row)
        user.pop("password_hash", None)
        user["is_active"] = bool(user.get("is_active", 1))
        return user

    def create_user(self, email: str, name: str, password_hash: str) -> int:
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "INSERT INTO users (email, name, password_hash, created_at) "
                "VALUES (?,?,?,?)",
                (email.strip().lower(), name.strip(), password_hash, time.time()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def user_by_email(self, email: str, with_hash: bool = False) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        if not row:
            return None
        return dict(row) if with_hash else self._public_user(row)

    def user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM users WHERE id = ?", (int(user_id),)
        ).fetchone()
        return self._public_user(row) if row else None

    def count_users(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def touch_login(self, user_id: int) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "UPDATE users SET last_login_at = ? WHERE id = ?",
                (time.time(), int(user_id)),
            )
            conn.commit()

    def set_password_hash(self, user_id: int, password_hash: str) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (password_hash, int(user_id)),
            )
            conn.commit()

    def adopt_orphan_data(self, user_id: int) -> Dict[str, int]:
        """
        Hand pre-accounts rows (user_id = 0) to a user.

        Called once, for the first account created on a database that already
        had history from before logins existed — so upgrading does not appear
        to wipe someone's saved searches.
        """
        with self._lock:
            conn = self._conn()
            searches = conn.execute(
                "UPDATE searches SET user_id = ? WHERE user_id = 0", (int(user_id),)
            ).rowcount
            chats = conn.execute(
                "UPDATE chat_messages SET user_id = ? WHERE user_id = 0",
                (int(user_id),),
            ).rowcount
            conn.commit()
            return {"searches": searches, "chat_messages": chats}

    # ------------------------------------------------------------------ #
    # sessions
    # ------------------------------------------------------------------ #
    def create_session(self, token: str, user_id: int, expires_at: float,
                       user_agent: str = "") -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(token, user_id, created_at, expires_at, user_agent) VALUES (?,?,?,?,?)",
                (token, int(user_id), time.time(), float(expires_at), user_agent[:300]),
            )
            conn.commit()

    def session_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Resolve a session token to its user, or None if missing/expired."""
        if not token:
            return None
        row = self._conn().execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token = ? AND s.expires_at > ? AND u.is_active = 1",
            (token, time.time()),
        ).fetchone()
        return self._public_user(row) if row else None

    def delete_session(self, token: str) -> bool:
        with self._lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            return cur.rowcount > 0

    def delete_user_sessions(self, user_id: int) -> int:
        with self._lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM sessions WHERE user_id = ?", (int(user_id),))
            conn.commit()
            return cur.rowcount

    def purge_expired_sessions(self) -> int:
        with self._lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (time.time(),))
            conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        conn = self._conn()
        one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "articles": one("SELECT COUNT(*) FROM articles"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "embeddings": one("SELECT COUNT(*) FROM embeddings"),
            "queries": one("SELECT COUNT(*) FROM searches"),
            "searches": one("SELECT COUNT(*) FROM searches"),
            "chat_messages": one("SELECT COUNT(*) FROM chat_messages"),
            "users": one("SELECT COUNT(*) FROM users"),
            "db_path": self.path,
        }

    def user_stats(self, user_id: int) -> Dict[str, Any]:
        conn = self._conn()
        one = lambda sql: conn.execute(sql, (int(user_id),)).fetchone()[0]  # noqa: E731
        return {
            "searches": one("SELECT COUNT(*) FROM searches WHERE user_id = ?"),
            "pinned": one("SELECT COUNT(*) FROM searches WHERE user_id = ? AND pinned = 1"),
            "chat_messages": one("SELECT COUNT(*) FROM chat_messages WHERE user_id = ?"),
            "papers_discussed": one(
                "SELECT COUNT(DISTINCT pmid) FROM chat_messages WHERE user_id = ?"
            ),
        }


store = Store()

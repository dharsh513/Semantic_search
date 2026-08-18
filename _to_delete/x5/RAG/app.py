"""
Flask server for PubMed Semantic Search (RAG).

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import logging
import os

from flask import Flask, Response, g, jsonify, redirect, render_template, request, url_for

from config import config
from rag import auth
from rag import export as exporter
from rag import paper_chat
from rag.auth import AuthError, login_required
from rag.embedder import embedder
from rag.pipeline import pipeline
from rag.pubmed_client import PubMedError
from rag.store import store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pubmed-rag")

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["JSON_SORT_KEYS"] = False

try:
    from flask_cors import CORS

    CORS(app)
except Exception:  # noqa: BLE001 - CORS is a convenience, not a requirement
    pass


APP_VERSION = "2.0.0"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.get("/")
@login_required
def index():
    return render_template("index.html", app_version=APP_VERSION, user=g.user)


def _render_auth(start_mode: str):
    if auth.current_user():
        return redirect(url_for("index"))
    return render_template(
        "auth.html",
        app_version=APP_VERSION,
        allow_signup=bool(config.ALLOW_SIGNUP),
        # A fresh install has no accounts — open on the sign-up panel.
        first_run=store.count_users() == 0,
        start_mode=start_mode,
        next_url=request.args.get("next", "/"),
    )


# Two endpoints rather than two rules on one, so url_for("auth_page") always
# resolves to /login — otherwise redirects for anonymous visitors could land
# on /signup, which is the wrong invitation for someone who has an account.
@app.get("/login")
def auth_page():
    """The split-screen login / sign-up screen."""
    return _render_auth("login")


@app.get("/signup")
def auth_page_signup():
    return _render_auth("signup")


# --------------------------------------------------------------------------- #
# Auth API
# --------------------------------------------------------------------------- #
def _safe_next(target: str) -> str:
    """Only ever redirect within this app — never to an attacker's URL."""
    target = (target or "/").strip()
    if not target.startswith("/") or target.startswith("//"):
        return "/"
    return target


@app.post("/api/auth/signup")
def api_signup():
    if not config.ALLOW_SIGNUP:
        return jsonify({"error": "Sign-ups are closed on this instance."}), 403

    payload = request.get_json(silent=True) or {}
    try:
        result = auth.signup(
            payload.get("email", ""),
            payload.get("name", ""),
            payload.get("password", ""),
        )
    except AuthError as exc:
        return jsonify({"error": exc.message, "field": exc.field}), exc.status

    user = result["user"]
    token, max_age = auth.start_session(user["id"], bool(payload.get("remember")))
    body = {
        "user": user,
        "next": _safe_next(payload.get("next", "/")),
        "first_account": result["first_account"],
        "adopted": result["adopted"],
    }
    log.info("New account: %s (id=%s)", user["email"], user["id"])
    return auth.set_session_cookie(jsonify(body), token, max_age)


@app.post("/api/auth/login")
def api_login():
    payload = request.get_json(silent=True) or {}
    try:
        result = auth.login(payload.get("email", ""), payload.get("password", ""))
    except AuthError as exc:
        return jsonify({"error": exc.message, "field": exc.field}), exc.status

    user = result["user"]
    token, max_age = auth.start_session(user["id"], bool(payload.get("remember")))
    body = {"user": user, "next": _safe_next(payload.get("next", "/"))}
    return auth.set_session_cookie(jsonify(body), token, max_age)


@app.post("/api/auth/logout")
def api_logout():
    auth.end_session(request.cookies.get(config.SESSION_COOKIE, ""))
    return auth.clear_session_cookie(jsonify({"ok": True}))


@app.get("/logout")
def logout_page():
    auth.end_session(request.cookies.get(config.SESSION_COOKIE, ""))
    return auth.clear_session_cookie(redirect(url_for("auth_page")))


@app.get("/api/auth/me")
def api_me():
    user = auth.current_user()
    if not user:
        return jsonify({"user": None, "authenticated": False}), 200
    return jsonify({
        "user": user,
        "authenticated": True,
        "stats": store.user_stats(user["id"]),
    })


@app.post("/api/auth/password")
@login_required
def api_change_password():
    payload = request.get_json(silent=True) or {}
    try:
        auth.change_password(
            g.user["id"],
            payload.get("current_password", ""),
            payload.get("new_password", ""),
        )
    except AuthError as exc:
        return jsonify({"error": exc.message, "field": exc.field}), exc.status
    # change_password revokes every session, including this one.
    return auth.clear_session_cookie(
        jsonify({"ok": True, "message": "Password changed. Please sign in again."})
    )


@app.post("/api/auth/strength")
def api_password_strength():
    """Server-side mirror of the meter, so both agree on what counts as strong."""
    payload = request.get_json(silent=True) or {}
    password = payload.get("password", "")
    result = auth.password_strength(password)
    result["problem"] = auth.password_problems(
        password, payload.get("email", ""), payload.get("name", "")
    )
    return jsonify(result)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
@login_required
def health():
    return jsonify(
        {
            "status": "ok",
            "embedding": embedder.info(),
            "ncbi_key_configured": bool(config.NCBI_API_KEY),
            "store": store.stats(),
            "user": g.user,
        }
    )


@app.get("/api/stats")
@login_required
def stats():
    return jsonify(
        {
            "store": store.stats(),
            "user_stats": store.user_stats(g.user["id"]),
            "recent_queries": store.recent_queries(g.user["id"], 8),
            "embedding": embedder.info(),
            "retrieval": {
                "candidate_pool": config.CANDIDATE_POOL,
                "top_k": config.TOP_K,
                "top_k_chunks": config.TOP_K_CHUNKS,
                "dense_weight": config.DENSE_WEIGHT,
                "lexical_weight": config.LEXICAL_WEIGHT,
                "mmr_lambda": config.MMR_LAMBDA,
                "chunk_words": config.CHUNK_WORDS,
            },
        }
    )


def _run_search(force_answer: bool = False):
    """
    Shared implementation behind /api/search and /api/ask.

    Both routes read the same request, so this must NOT be reached by opening a
    second request context — that would drop the session cookie and the
    resolved user along with it.
    """
    payload = request.get_json(silent=True) or {}
    args = request.args

    query = (payload.get("query") or args.get("q") or args.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'query'."}), 400

    def _int(name, default):
        raw = payload.get(name, args.get(name))
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    top_k = max(1, min(_int("top_k", config.TOP_K), 50))
    field = (payload.get("field") or args.get("field") or "auto").lower()
    if field not in {"auto", "title", "author"}:
        field = "auto"

    if force_answer:
        with_answer = True
    else:
        raw = payload.get("with_answer", args.get("with_answer", "true"))
        with_answer = str(raw).lower() not in {"0", "false", "no"}

    filters = {
        "sort": (payload.get("sort") or args.get("sort") or "relevance"),
        "mindate": payload.get("mindate") or args.get("mindate"),
        "maxdate": payload.get("maxdate") or args.get("maxdate"),
    }
    if filters["sort"] not in {"relevance", "date", "pub_date", "most+recent"}:
        filters["sort"] = "relevance"
    if filters["sort"] == "date":
        filters["sort"] = "pub_date"

    try:
        result = pipeline.search(
            query,
            top_k=top_k,
            search_field=field,
            filters=filters,
            with_answer=with_answer,
            user_id=g.user["id"],
        )
        return jsonify(result)
    except PubMedError as exc:
        log.exception("NCBI failure")
        return jsonify({"error": f"Could not reach NCBI: {exc}"}), 502
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        log.exception("Search failed")
        return jsonify({"error": f"Search failed: {exc}"}), 500


@app.route("/api/search", methods=["GET", "POST"])
@login_required
def search():
    return _run_search()


@app.post("/api/ask")
@login_required
def ask():
    """Alias of /api/search that always returns a generated answer."""
    return _run_search(force_answer=True)


@app.get("/api/article/<pmid>")
@login_required
def article(pmid: str):
    try:
        art = pipeline.article(pmid)
    except PubMedError as exc:
        return jsonify({"error": f"Could not reach NCBI: {exc}"}), 502
    if not art:
        return jsonify({"error": f"PMID {pmid} not found."}), 404
    return jsonify(art)


@app.get("/api/similar/<pmid>")
@login_required
def similar(pmid: str):
    try:
        return jsonify({"pmid": pmid, "results": pipeline.similar(pmid)})
    except PubMedError as exc:
        return jsonify({"error": f"Could not reach NCBI: {exc}"}), 502
    except Exception as exc:  # noqa: BLE001
        log.exception("similar failed")
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------------- #
# Search history
# --------------------------------------------------------------------------- #
@app.get("/api/history")
@login_required
def history_list():
    def _int(name, default):
        try:
            return int(request.args.get(name, default))
        except (TypeError, ValueError):
            return default

    return jsonify(
        {
            "items": store.history(
                g.user["id"],
                limit=max(1, min(_int("limit", 50), 200)),
                offset=max(0, _int("offset", 0)),
                search=request.args.get("q", ""),
            ),
            "total": store.user_stats(g.user["id"])["searches"],
        }
    )


@app.get("/api/history/<int:search_id>")
@login_required
def history_get(search_id: int):
    """Reopen a past search from its stored snapshot — no NCBI call."""
    # Scoped by user, so a guessed id from another account reads as 404.
    record = store.get_search(search_id, user_id=g.user["id"])
    if not record:
        return jsonify({"error": f"Search {search_id} not found."}), 404

    snapshot = record.get("snapshot") or {}
    return jsonify(
        {
            "search_id": record["id"],
            "query": record["query"],
            "created_at": record["created_at"],
            "pinned": bool(record["pinned"]),
            "from_history": True,
            "understanding": snapshot.get("understanding", {}),
            "stages": snapshot.get("stages", {}),
            "results": snapshot.get("results", []),
            "answer": snapshot.get("answer", {}),
            "took_ms": snapshot.get("took_ms", record["took_ms"]),
        }
    )


@app.post("/api/history/<int:search_id>/pin")
@login_required
def history_pin(search_id: int):
    payload = request.get_json(silent=True) or {}
    pinned = bool(payload.get("pinned", True))
    if not store.set_pinned(search_id, pinned, g.user["id"]):
        return jsonify({"error": f"Search {search_id} not found."}), 404
    return jsonify({"search_id": search_id, "pinned": pinned})


@app.delete("/api/history/<int:search_id>")
@login_required
def history_delete(search_id: int):
    if not store.delete_search(search_id, g.user["id"]):
        return jsonify({"error": f"Search {search_id} not found."}), 404
    return jsonify({"deleted": search_id})


@app.delete("/api/history")
@login_required
def history_clear():
    keep = str(request.args.get("keep_pinned", "true")).lower() not in {"0", "false", "no"}
    return jsonify({"deleted": store.clear_history(g.user["id"], keep_pinned=keep)})


# --------------------------------------------------------------------------- #
# Citation export
# --------------------------------------------------------------------------- #
@app.post("/api/export")
@login_required
def export():
    """
    Export selected results as PDF / BibTeX / RIS / plain text.

    Body: {
      pmids:     ["31456127", ...],      # required unless search_id is given
      search_id: 12,                     # pull answer + passages from history
      format:    "pdf" | "bib" | "ris" | "txt",
      mode:      "references" | "report",
      query:     "..."                   # used in the PDF header
    }
    """
    payload = request.get_json(silent=True) or {}
    fmt = str(payload.get("format", "pdf")).lower()
    mode = str(payload.get("mode", "references")).lower()
    if fmt not in exporter.FORMATS:
        return jsonify({"error": f"Unsupported format '{fmt}'."}), 400
    if mode not in {"references", "report"}:
        mode = "references"

    pmids = [str(p) for p in (payload.get("pmids") or []) if str(p).strip()]
    query = payload.get("query", "") or ""
    answer = payload.get("answer") or None
    stages = payload.get("stages") or None
    enriched: dict = {}

    # A history record supplies the answer, stages and matched passages, so the
    # client only has to send the id and the subset of PMIDs it wants.
    if payload.get("search_id") is not None:
        record = store.get_search(int(payload["search_id"]), user_id=g.user["id"])
        if not record:
            return jsonify({"error": "Search not found."}), 404
        snapshot = record.get("snapshot") or {}
        query = query or record.get("query", "")
        answer = answer or snapshot.get("answer")
        stages = stages or snapshot.get("stages")
        enriched = {d["pmid"]: d for d in snapshot.get("results", [])}
        if not pmids:
            pmids = [str(p) for p in record.get("pmids", [])]

    if not pmids:
        return jsonify({"error": "No records selected for export."}), 400

    cached = store.get_articles(pmids)
    missing = [p for p in pmids if p not in cached]
    if missing:
        try:
            fetched = pipeline.article_batch(missing)
            cached.update({a["pmid"]: a for a in fetched})
        except PubMedError as exc:
            return jsonify({"error": f"Could not reach NCBI: {exc}"}), 502

    articles = []
    for pmid in pmids:                      # preserve the caller's ordering
        art = cached.get(pmid)
        if not art:
            continue
        extra = enriched.get(pmid) or {}
        articles.append({**art, "matched_passages": extra.get("matched_passages", [])})

    if not articles:
        return jsonify({"error": "None of the selected records could be resolved."}), 404

    # If the caller narrowed the selection, renumber the answer's citations so
    # the markers in the PDF still line up with the reference list.
    if answer and mode == "report":
        keep = {a["pmid"] for a in articles}
        cites = [c for c in (answer.get("citations") or []) if c.get("pmid") in keep]
        if len(cites) != len(answer.get("citations") or []):
            answer = {**answer, "citations": cites}

    try:
        blob = exporter.build(fmt, articles, mode=mode, query=query,
                              answer=answer, stages=stages)
    except Exception as exc:  # noqa: BLE001
        log.exception("Export failed")
        return jsonify({"error": f"Export failed: {exc}"}), 500

    name = exporter.filename(fmt, query, mode)
    return Response(
        blob,
        mimetype=exporter.FORMATS[fmt][0],
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Content-Length": str(len(blob)),
            "X-Export-Filename": name,
        },
    )


@app.post("/api/citations")
@login_required
def citation_preview():
    """Formatted Vancouver strings for a set of PMIDs — used by the UI preview."""
    from rag import citations as cite

    payload = request.get_json(silent=True) or {}
    pmids = [str(p) for p in (payload.get("pmids") or []) if str(p).strip()]
    if not pmids:
        return jsonify({"error": "No PMIDs given."}), 400
    cached = store.get_articles(pmids)
    ordered = [cached[p] for p in pmids if p in cached]
    return jsonify({"citations": cite.numbered_reference_list(ordered)})


# --------------------------------------------------------------------------- #
# Per-paper chat
# --------------------------------------------------------------------------- #
@app.get("/api/chat/<pmid>")
@login_required
def chat_get(pmid: str):
    try:
        article = pipeline.article(pmid)
    except PubMedError as exc:
        return jsonify({"error": f"Could not reach NCBI: {exc}"}), 502
    if not article:
        return jsonify({"error": f"PMID {pmid} not found."}), 404
    return jsonify(
        {
            "pmid": pmid,
            "messages": store.chat_history(g.user["id"], pmid),
            "suggestions": paper_chat.suggested_questions(article),
            "has_abstract": bool((article.get("abstract") or "").strip()),
        }
    )


@app.post("/api/chat/<pmid>")
@login_required
def chat_post(pmid: str):
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'."}), 400

    try:
        pipeline.article(pmid)  # ensure it is cached before answering
        result = paper_chat.chat(g.user["id"], pmid, question)
    except PubMedError as exc:
        return jsonify({"error": f"Could not reach NCBI: {exc}"}), 502
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        log.exception("Chat failed")
        return jsonify({"error": f"Chat failed: {exc}"}), 500

    return jsonify({"pmid": pmid, "question": question, **result})


@app.delete("/api/chat/<pmid>")
@login_required
def chat_clear(pmid: str):
    return jsonify({"pmid": pmid, "deleted": store.clear_chat(g.user["id"], pmid)})


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return redirect(url_for("index"))


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #
# Never let the reloader watch installed packages. sentence-transformers
# imports most of `transformers` lazily, so those modules appear in sys.modules
# only after the server is already serving — and the stat reloader treats every
# newly-seen file as a change, restarting the server in a loop.
RELOADER_EXCLUDES = [
    "*site-packages*", "*dist-packages*", "*/lib/python*", "*\\Lib\\*",
    "*anaconda3*", "*miniconda3*", "*.venv*", "*node_modules*",
]


def _warm_embedder() -> None:
    """
    Load the embedding model once, in the background, at startup.

    Without this the model loads inside the first search request, which makes
    that request appear to hang for several seconds.
    """
    try:
        info = embedder.info()
        log.info("Embedding backend ready: %s (%s, %s-d)",
                 info["model"], info["kind"], info["dim"])
    except Exception as exc:  # noqa: BLE001 - never block startup
        log.warning("Could not preload the embedding model: %s", exc)


if __name__ == "__main__":
    import threading

    # With the reloader on, this module runs twice; only the child actually
    # serves requests, so only the child should warm the model.
    is_serving_process = (
        not config.USE_RELOADER or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    )

    if is_serving_process:
        log.info("Data directory : %s", os.path.dirname(config.DB_PATH))
        log.info("Embedding model: %s", config.EMBEDDING_MODEL)
        if store.count_users() == 0:
            log.info("No accounts yet — the first visit opens the sign-up screen.")
        else:
            log.info("Accounts       : %s", store.count_users())
        log.info("Auto-reloader  : %s",
                 "on (installed packages excluded)" if config.USE_RELOADER
                 else "off — set USE_RELOADER=true to enable")
        log.info("Serving on      http://%s:%s", config.HOST, config.PORT)

        if config.PRELOAD_EMBEDDER:
            threading.Thread(target=_warm_embedder, name="warm-embedder",
                             daemon=True).start()

    app.run(
        host=config.HOST,
        port=int(config.PORT),
        debug=bool(config.DEBUG),
        use_reloader=bool(config.USE_RELOADER),
        exclude_patterns=RELOADER_EXCLUDES,
    )

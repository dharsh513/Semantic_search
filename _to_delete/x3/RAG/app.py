"""
Flask server for PubMed Semantic Search (RAG).

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import logging
import os

from flask import Flask, Response, jsonify, render_template, request

from config import config
from rag import export as exporter
from rag import paper_chat
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


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template("index.html", app_version="1.0.0")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "embedding": embedder.info(),
            "ncbi_key_configured": bool(config.NCBI_API_KEY),
            "store": store.stats(),
        }
    )


@app.get("/api/stats")
def stats():
    return jsonify(
        {
            "store": store.stats(),
            "recent_queries": store.recent_queries(8),
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


@app.route("/api/search", methods=["GET", "POST"])
def search():
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

    with_answer = payload.get("with_answer", args.get("with_answer", "true"))
    with_answer = str(with_answer).lower() not in {"0", "false", "no"}

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


@app.post("/api/ask")
def ask():
    """Alias of /api/search that always returns a generated answer."""
    payload = request.get_json(silent=True) or {}
    payload["with_answer"] = True
    with app.test_request_context("/api/search", json=payload):
        return search()


@app.get("/api/article/<pmid>")
def article(pmid: str):
    try:
        art = pipeline.article(pmid)
    except PubMedError as exc:
        return jsonify({"error": f"Could not reach NCBI: {exc}"}), 502
    if not art:
        return jsonify({"error": f"PMID {pmid} not found."}), 404
    return jsonify(art)


@app.get("/api/similar/<pmid>")
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
def history_list():
    def _int(name, default):
        try:
            return int(request.args.get(name, default))
        except (TypeError, ValueError):
            return default

    return jsonify(
        {
            "items": store.history(
                limit=max(1, min(_int("limit", 50), 200)),
                offset=max(0, _int("offset", 0)),
                search=request.args.get("q", ""),
            ),
            "total": store.stats()["searches"],
        }
    )


@app.get("/api/history/<int:search_id>")
def history_get(search_id: int):
    """Reopen a past search from its stored snapshot — no NCBI call."""
    record = store.get_search(search_id)
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
def history_pin(search_id: int):
    payload = request.get_json(silent=True) or {}
    pinned = bool(payload.get("pinned", True))
    if not store.set_pinned(search_id, pinned):
        return jsonify({"error": f"Search {search_id} not found."}), 404
    return jsonify({"search_id": search_id, "pinned": pinned})


@app.delete("/api/history/<int:search_id>")
def history_delete(search_id: int):
    if not store.delete_search(search_id):
        return jsonify({"error": f"Search {search_id} not found."}), 404
    return jsonify({"deleted": search_id})


@app.delete("/api/history")
def history_clear():
    keep = str(request.args.get("keep_pinned", "true")).lower() not in {"0", "false", "no"}
    return jsonify({"deleted": store.clear_history(keep_pinned=keep)})


# --------------------------------------------------------------------------- #
# Citation export
# --------------------------------------------------------------------------- #
@app.post("/api/export")
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
        record = store.get_search(int(payload["search_id"]))
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
            "messages": store.chat_history(pmid),
            "suggestions": paper_chat.suggested_questions(article),
            "has_abstract": bool((article.get("abstract") or "").strip()),
        }
    )


@app.post("/api/chat/<pmid>")
def chat_post(pmid: str):
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'."}), 400

    try:
        pipeline.article(pmid)  # ensure it is cached before answering
        result = paper_chat.chat(pmid, question)
    except PubMedError as exc:
        return jsonify({"error": f"Could not reach NCBI: {exc}"}), 502
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        log.exception("Chat failed")
        return jsonify({"error": f"Chat failed: {exc}"}), 500

    return jsonify({"pmid": pmid, "question": question, **result})


@app.delete("/api/chat/<pmid>")
def chat_clear(pmid: str):
    return jsonify({"pmid": pmid, "deleted": store.clear_chat(pmid)})


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    log.info("Data directory : %s", os.path.dirname(config.DB_PATH))
    log.info("Embedding model: %s", config.EMBEDDING_MODEL)
    log.info("Serving on      http://%s:%s", config.HOST, config.PORT)
    app.run(host=config.HOST, port=int(config.PORT), debug=bool(config.DEBUG))

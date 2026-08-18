"""
Flask server for PubMed Semantic Search (RAG).

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, render_template, request

from config import config
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


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    log.info("Data directory : %s", os.path.dirname(config.DB_PATH))
    log.info("Embedding model: %s", config.EMBEDDING_MODEL)
    log.info("Serving on      http://%s:%s", config.HOST, config.PORT)
    app.run(host=config.HOST, port=int(config.PORT), debug=bool(config.DEBUG))

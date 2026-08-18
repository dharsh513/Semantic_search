"""
Bulk corpus builder.

Pre-loads the local SQLite cache + embedding store with PubMed records for a
set of topics, so the first live search is instant and the app keeps working
offline afterwards.

    python tools/ingest.py "gut microbiota depression" "CRISPR sickle cell" -n 100
    python tools/ingest.py --file topics.txt -n 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402
from rag.chunker import chunk_articles  # noqa: E402
from rag.embedder import embedder  # noqa: E402
from rag.pubmed_client import PubMedError, pubmed_client  # noqa: E402
from rag.store import store  # noqa: E402


def ingest_topic(topic: str, n: int) -> int:
    print(f"\n▸ {topic}")
    try:
        res = pubmed_client.esearch(topic, retmax=n)
    except PubMedError as exc:
        print(f"  ! ESearch failed: {exc}")
        return 0

    pmids = res["ids"]
    print(f"  {res['count']:,} total matches · fetching {len(pmids)}")
    if res["mesh_terms"]:
        print(f"  MeSH: {', '.join(res['mesh_terms'])}")
    if not pmids:
        return 0

    known = store.fresh_pmids(pmids)
    todo = [p for p in pmids if p not in known]
    print(f"  {len(known)} already cached · {len(todo)} to fetch")

    if todo:
        try:
            arts = pubmed_client.efetch(todo)
        except PubMedError as exc:
            print(f"  ! EFetch failed: {exc}")
            return 0
        store.upsert_articles(arts)
        for art in arts:
            store.replace_chunks(art["pmid"], chunk_articles([art]))
        print(f"  + {len(arts)} records stored")

    # Embed anything not yet embedded.
    chunks = store.get_chunks(pmids)
    cached = store.load_embeddings(embedder.name, [c["chunk_id"] for c in chunks])
    pending = [c for c in chunks if c["chunk_id"] not in cached]
    if pending:
        print(f"  embedding {len(pending)} passages with {embedder.name} …")
        vecs = embedder.encode_documents([c["text"] for c in pending])
        store.save_embeddings(
            embedder.name, [(c["chunk_id"], vecs[i]) for i, c in enumerate(pending)]
        )
    print(f"  ✓ {len(chunks)} passages ready")
    return len(pmids)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-load the local PubMed corpus.")
    ap.add_argument("topics", nargs="*", help="Search topics.")
    ap.add_argument("--file", help="Text file with one topic per line.")
    ap.add_argument("-n", "--per-topic", type=int, default=100,
                    help="Records to fetch per topic (default 100).")
    args = ap.parse_args()

    topics = list(args.topics)
    if args.file:
        topics += [
            ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    if not topics:
        ap.error("Give at least one topic, or use --file.")

    print("=" * 66)
    print("PubMed corpus ingestion")
    print(f"Embedding backend: {embedder.kind} ({embedder.name}, dim={embedder.dim})")
    print(f"Database         : {config.DB_PATH}")
    print("=" * 66)

    started = time.time()
    total = sum(ingest_topic(t, args.per_topic) for t in topics)

    st = store.stats()
    print("\n" + "=" * 66)
    print(f"Done in {time.time() - started:.1f}s · {total} PMIDs touched")
    print(f"Corpus: {st['articles']:,} articles · {st['chunks']:,} passages · "
          f"{st['embeddings']:,} embeddings")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())

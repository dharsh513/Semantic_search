# PubMed Semantic Search — a RAG application

Semantic search and grounded question answering over **PubMed**, built with a
Retrieval-Augmented Generation pipeline in Python, served by **Flask**, with a
plain **HTML / CSS / JavaScript** frontend.

Records are pulled live from NCBI's public **E-utilities** endpoints —
**no API key is required**. Embeddings are computed **locally**; nothing is sent
to a third-party AI service.

---

## The problem this solves

PubMed's default search is lexical. The requirement brief lists four specific
pain points; here is how each is handled.

| Challenge | How this project addresses it |
|---|---|
| **Keyword overload** — users dump long natural-language questions and get nothing back | The query is reduced to its content words (stopwords stripped), and if the strict query returns zero hits it is *progressively relaxed* (`AND` → fewer terms → `OR` → single term) until PubMed responds |
| **Boolean operators** — most users don't know them | The user never types Boolean syntax. `pipeline.understand()` builds the Boolean expression, and the exact string sent to ESearch is shown in the pipeline trace so the user can learn it |
| **Truncation** — wildcards are unintuitive | ESpell supplies spelling corrections, and semantic embedding matches morphological and synonym variants (`probiotic` ↔ `live bacteria supplement`) without any wildcard |
| **MeSH terms** — standardized vocabulary users don't know | NCBI's Automatic Term Mapping is exploited deliberately: the `querytranslation` field is parsed for `[MeSH Terms]` descriptors, they are shown as chips in the UI, and each record's own MeSH headings are folded into its embedded text |

On top of that, results are **re-ranked semantically** rather than returned in
PubMed's order, and a **cited answer** is synthesised from the retrieved
abstracts.

## Beyond search

| Feature | What it does |
|---|---|
| **Citation export** | Tick any subset of the results and export them as a **PDF** — either a compact numbered reference list or a full report (search parameters → grounded answer with its `[n]` markers → references → every abstract with MeSH headings). Also exports **BibTeX**, **RIS** and plain text. Citations are **Vancouver/NLM**, the biomedical standard. Narrowing the selection renumbers the answer's citation markers so they still line up. |
| **Search history** | Every search is saved automatically with a full snapshot of its results. Reopen one from the history drawer and it is restored **without another NCBI call**. Filter, pin (pinned searches survive "Clear"), delete individually or in bulk. |
| **Paper view + chat** | Click any result to open it: the full abstract with your query terms highlighted, MeSH headings, author keywords, and links. Alongside it, a chat panel scoped to **that paper only**. |

---

## Architecture

```
                       ┌──────────────────────────────────────────┐
  Browser              │  templates/index.html                    │
  (HTML/CSS/JS) ◄──────┤  static/css/style.css                    │
                       │  static/js/app.js                        │
                       └───────────────┬──────────────────────────┘
                                       │  fetch()  JSON
                       ┌───────────────▼──────────────────────────┐
  Flask                │  app.py                                  │
                       │  /api/search  /api/ask  /api/article/…    │
                       │  /api/similar/…  /api/health  /api/stats  │
                       └───────────────┬──────────────────────────┘
                                       │
                       ┌───────────────▼──────────────────────────┐
  RAG pipeline         │  rag/pipeline.py                         │
                       └───┬───────┬───────┬───────┬───────┬──────┘
                           │       │       │       │       │
       pubmed_client ──────┘       │       │       │       └────── generator
       (ESpell/ESearch/EFetch)     │       │       │        (grounded answer
                                   │       │       │         + citations)
                store ─────────────┘       │       └─────── retriever
       (SQLite cache: articles,            │               (dense + BM25 + MMR)
        chunks, embeddings, query log)     │
                                    chunker + embedder
                              (sentence-aware passages,
                               local MiniLM vectors)
```

### Request flow

1. **Query understanding** — ESpell correction, exact-title / author mode
   detection, stopword removal, Boolean construction.
2. **Sparse retrieval** — `ESearch` returns candidate PMIDs plus NCBI's own
   query translation, from which MeSH descriptors are extracted.
3. **Ingestion** — `EFetch` pulls full records (cache-first: anything already in
   SQLite and inside the TTL is not re-fetched). Records are parsed from XML,
   including structured-abstract labels, authors, journal, DOI and MeSH.
4. **Chunking** — each abstract becomes overlapping, sentence-aware passages.
   Chunk 0 is always *title + MeSH*, so records with no abstract stay findable.
5. **Embedding** — passages are encoded with a local sentence-transformer and
   cached in SQLite, so repeat queries never re-embed.
6. **Dense retrieval** — hybrid scoring:
   `0.72 × cosine + 0.28 × BM25`, then **MMR** diversification so the answer is
   not five near-identical passages from the same paper.
7. **Document rollup** — passage scores collapse into ranked articles
   (best passage + a bonus for multiple independently-relevant passages).
8. **Generation** — sentences from the retrieved passages are scored against the
   query embedding, greedily de-duplicated, and stitched into an answer where
   **every sentence carries a `[n]` citation** back to a PMID.
9. **Persistence** — the whole result set is snapshotted into SQLite so the
   search can be reopened later, and exported to PDF/BibTeX/RIS on demand.

### Per-paper chat

`rag/paper_chat.py` runs the same idea at the scale of a single record. The
abstract is split into sentences tagged with their structured section, each is
scored against the question (`0.68 × cosine + 0.32 × lexical overlap`, plus a
section-intent bonus), and the best non-redundant sentences are returned
**verbatim** with their section labels as visible evidence.

Because raw cosine is not comparable across embedding models, **confidence is
built from three scale-free signals** rather than the similarity value itself:

| Signal | Question it answers |
|---|---|
| *Discrimination* | Does one sentence actually stand out, or does this paper answer the question no better than any other sentence? |
| *Lexical cover* | How many of the question's own content words appear in the sentence about to be quoted? |
| *Intent hit* | If the question asked for methods/results/conclusions, did the winning sentence come from that section? |

`confidence = 0.42 × lexical_cover + 0.38 × discrimination + 0.20 × intent_hit`

Ask something the abstract does not cover and all three collapse — so instead
of confidently quoting an unrelated sentence, the assistant says the abstract
does not address it and shows you the nearest thing it does say. This is
verified in the self-test on both embedding backends.

---

## Quick start

### Windows (the easy way)

Double-click **`run.bat`**. It creates a virtual environment, installs
everything, and starts the server. Then open <http://127.0.0.1:5000>.

### Any platform

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>.

> **First run** downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB) from
> Hugging Face. After that the app works with only NCBI access. If the download
> fails the app **does not crash** — it automatically falls back to a TF-IDF+SVD
> embedder and tells you so in the `engine:` badge.

---

## Verifying it works

```bash
python tools/selftest.py
```

Runs 148 checks over the whole stack — XML parsing, chunking, MeSH extraction,
hybrid ranking, citation integrity, caching, export formats, history
persistence, per-paper chat, and every Flask route — against canned NCBI
responses. **No network needed.**

```
==========================================================================
PubMed RAG — offline self-test
Embedding backend : sentence-transformer  (all-MiniLM-L6-v2, dim=384)
==========================================================================
[1]  NCBI XML parsing                    ...  9 passed
[2]  Chunking                            ...  4 passed
[3]  Query understanding                 ...  3 passed
[4]  End-to-end search                   ...  8 passed
[5]  Grounded answer generation          ...  5 passed
[6]  Caching                             ...  5 passed
[7]  Semantic behaviour                  ...  2 passed
[8]  Article + similar endpoints         ...  3 passed
[9]  Flask routes                        ... 11 passed
[10] Citation formatting (Vancouver/NLM) ... 11 passed
[11] Export formats                      ... 13 passed
[12] Search history                      ... 12 passed
[13] Per-paper chat                      ... 28 passed
[14] Flask routes — export/history/chat  ... 22 passed
[15] Static assets                       ... 12 passed
  148 passed, 0 failed
```

Some of those checks are the ones that matter for the project's premise:

* A query phrased with **completely different vocabulary** from the target
  paper (`"nerve pathway linking bacteria to anxiety in rodents"` → the *vagal
  signalling* paper) still ranks it first. Plain keyword search does not do that.
* Every sentence of a chat answer is asserted to appear **verbatim** in the
  source abstract — the test would fail if the assistant ever paraphrased.
* Asking an off-topic question ("the boiling point of liquid helium") must
  produce low confidence and a refusal, on **both** embedding backends.
* Narrowing an export selection must renumber the answer's citation markers so
  they still match the reference list.

There is also an offline UI sandbox:

```bash
python tools/demo_server.py     # http://127.0.0.1:5001, PubMed mocked
```

---

## Pre-loading a corpus (optional)

```bash
python tools/ingest.py "gut microbiota depression" "CRISPR sickle cell" -n 100
python tools/ingest.py --file topics.txt -n 200
```

Fetches, chunks and embeds records ahead of time so the first search is instant.

---

## API

All endpoints return JSON.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | The web UI |
| `POST` | `/api/search` | Main search. Body: `query`, `top_k`, `field`, `sort`, `mindate`, `maxdate`, `with_answer` |
| `GET` | `/api/search?q=…` | Same, via query string |
| `POST` | `/api/ask` | Search that always returns a generated answer |
| `GET` | `/api/article/<pmid>` | Full record (cache-first) |
| `GET` | `/api/similar/<pmid>` | Semantic "more like this" |
| `GET` | `/api/health` | Embedding backend + cache status |
| `GET` | `/api/stats` | Corpus size, recent queries, retrieval settings |
| `GET` | `/api/history` | Past searches. `?q=` filters, `?limit=`/`?offset=` paginate |
| `GET` | `/api/history/<id>` | Restore a past search from its snapshot |
| `POST` | `/api/history/<id>/pin` | Pin/unpin so "Clear" spares it |
| `DELETE` | `/api/history/<id>` | Delete one search |
| `DELETE` | `/api/history` | Clear history (`?keep_pinned=false` to wipe everything) |
| `POST` | `/api/export` | Download PDF / BibTeX / RIS / text for selected PMIDs |
| `POST` | `/api/citations` | Formatted Vancouver strings (used by the export preview) |
| `GET` | `/api/chat/<pmid>` | Stored transcript + suggested questions for a paper |
| `POST` | `/api/chat/<pmid>` | Ask a question about that paper |
| `DELETE` | `/api/chat/<pmid>` | Clear that paper's conversation |

```bash
# search
curl -X POST http://127.0.0.1:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"how does gut microbiota influence depression?","top_k":5}'

# export three records as a full PDF report
curl -X POST http://127.0.0.1:5000/api/export -o report.pdf \
  -H "Content-Type: application/json" \
  -d '{"pmids":["31456127","36527918","29276734"],
       "format":"pdf","mode":"report","query":"gut microbiota depression"}'

# ask a question about one paper
curl -X POST http://127.0.0.1:5000/api/chat/31456127 \
  -H "Content-Type: application/json" \
  -d '{"question":"how many patients were studied?"}'
```

Response shape:

```jsonc
{
  "query": "...",
  "understanding": { "original", "corrected", "field", "terms", "notes" },
  "stages": {
    "pubmed_query": "...",              // exact string sent to ESearch
    "query_translation": "...",         // NCBI's Automatic Term Mapping output
    "mesh_terms": ["gastrointestinal microbiome", "depression"],
    "total_matches": 18452,
    "chunks_indexed": 214,
    "embedding": { "model", "kind", "dim" },
    "retrieval": { "dense_weight", "lexical_weight", "mmr_lambda" }
  },
  "results": [
    {
      "rank": 1, "pmid": "...", "title": "...", "authors": [...],
      "journal": "...", "pub_date": "...", "doi": "...",
      "mesh_terms": [...], "score": 0.83, "relevance": 83.0,
      "matched_passages": [ { "section": "Results", "text": "...", "score": 0.81 } ]
    }
  ],
  "answer": {
    "answer": "… [1]. … [2].",
    "citations": [ { "n": 1, "pmid": "...", "title": "...", "url": "..." } ],
    "grounded": true
  },
  "took_ms": 1840
}
```

---

## Configuration

Everything is optional — copy `.env.example` to `.env` to change anything.

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Try `all-mpnet-base-v2` for quality, or `pritamdeka/S-PubMedBert-MS-MARCO` for a biomedical-domain model |
| `CANDIDATE_POOL` | `60` | PMIDs pulled from PubMed before semantic re-ranking |
| `DENSE_WEIGHT` / `LEXICAL_WEIGHT` | `0.72` / `0.28` | Hybrid scoring mix |
| `MMR_LAMBDA` | `0.72` | 1.0 = pure relevance, 0.0 = pure diversity |
| `CHUNK_WORDS` | `110` | Passage size |
| `CACHE_TTL_DAYS` | `30` | How long a cached record is considered fresh |
| `NCBI_API_KEY` | *(blank)* | **Not required.** Only raises the rate limit from 3/s to 10/s |
| `OPENAI_API_KEY` | *(blank)* | **Not required.** If present, the same retrieved evidence is rephrased by an LLM instead of extracted |

---

## Project layout

```
RAG/
├── app.py                  Flask server + JSON API
├── config.py               All settings (env-overridable)
├── requirements.txt
├── run.bat                 Windows one-click launcher
├── .env.example
├── rag/
│   ├── pubmed_client.py    ESpell / ESearch / EFetch, rate limiting, XML parsing
│   ├── store.py            SQLite: articles, chunks, embeddings, history, chat
│   ├── chunker.py          Sentence-aware overlapping passages
│   ├── embedder.py         Local sentence-transformer + TF-IDF fallback
│   ├── vector_index.py     Cosine index (optional FAISS) + MMR
│   ├── retriever.py        BM25 + hybrid fusion + document rollup
│   ├── generator.py        Grounded extractive answer with citations
│   ├── citations.py        Vancouver/NLM formatting, BibTeX, RIS
│   ├── export.py           PDF report + reference list (ReportLab)
│   ├── paper_chat.py       Extractive Q&A scoped to a single paper
│   └── pipeline.py         Orchestration
├── templates/index.html
├── static/css/style.css
├── static/js/app.js
├── tools/
│   ├── selftest.py         52 offline checks over the whole stack
│   ├── demo_server.py      UI sandbox with PubMed mocked
│   └── ingest.py           Bulk corpus pre-loader
└── data/                   SQLite cache (created on first run)
```

---

## Design notes

**Why hybrid instead of pure vector search?** Biomedical text is full of exact
entities — gene symbols, drug names, PMIDs — where lexical matching is simply
correct. Pure dense retrieval drifts on those. BM25 alone is the keyword
brittleness the project exists to fix. The 72/28 mix keeps precise entity
matching while still bridging vocabulary gaps.

**Why extractive generation?** Every sentence in the answer is lifted verbatim
from a retrieved abstract and tagged with its PMID, so the answer physically
cannot hallucinate a finding or invent a citation — a hard requirement for
clinical literature. An optional LLM path exists for fluency, but the grounded
answer is always computed first.

**Why cache in SQLite?** It respects NCBI's rate limits, makes repeat searches
near-instant (second identical search issues zero EFetch calls), and turns the
app into a growing offline corpus rather than a stateless proxy. It is also what
makes history free: the snapshot needed to redisplay a past search is already
there.

**Why is the paper chat extractive too?** The same reason as the main answer,
only sharper: a chat interface invites open-ended questions, which is exactly
where a generative model would start filling gaps. Quoting the abstract makes
the failure mode visible — you see the sentences it used and a confidence score,
so a wrong answer looks wrong rather than fluent.

**Keyboard:** `H` opens history, `Esc` closes any overlay.

---

## Adding an LLM later (optional)

Nothing here needs one, but if you set `OPENAI_API_KEY` in `.env` the main
search answer is rephrased from the *same retrieved passages* — retrieval,
citations and grounding are unchanged. The per-paper chat stays extractive by
design. Without a key everything works exactly as documented.

---

## Attribution

Literature data courtesy of the **National Library of Medicine (NLM)** via NCBI
E-utilities. This project is not endorsed by NLM/NCBI. Please respect the
[E-utilities usage guidelines](https://www.ncbi.nlm.nih.gov/books/NBK25497/):
no more than 3 requests/second without an API key (enforced in
`rag/pubmed_client.py`).

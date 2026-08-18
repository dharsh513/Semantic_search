"""
rag — Retrieval-Augmented Generation package for semantic search over PubMed.

Modules
-------
pubmed_client : NCBI E-utilities access (ESpell / ESearch / EFetch). No API key required.
store         : SQLite cache of fetched PubMed records.
chunker       : Splits abstracts into overlapping, sentence-aware passages.
embedder      : Local sentence-transformer embeddings (TF-IDF fallback).
vector_index  : In-memory cosine-similarity index (optional FAISS backend).
retriever     : Hybrid dense + BM25 retrieval with MMR diversification.
generator     : Extractive, citation-grounded answer synthesis.
pipeline      : Orchestrates the full query -> answer flow.
"""

__version__ = "1.0.0"

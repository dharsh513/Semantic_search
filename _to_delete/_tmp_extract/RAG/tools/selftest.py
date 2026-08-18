"""
Offline self-test.

Runs the entire RAG pipeline against canned NCBI responses, so you can verify
chunking, embedding, hybrid retrieval, document rollup, answer generation and
every Flask route without touching the network.

Usage (from the project root):
    python tools/selftest.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use a throwaway database so the real cache is never polluted.
_TMP = tempfile.mkdtemp(prefix="pubmed_rag_selftest_")
os.environ["RAG_DATA_DIR"] = _TMP

from config import config  # noqa: E402
from rag import pubmed_client as pc  # noqa: E402

# --------------------------------------------------------------------------- #
# Canned NCBI payloads (shape-identical to the real E-utilities responses)
# --------------------------------------------------------------------------- #
ESEARCH_JSON = {
    "esearchresult": {
        "count": "18452",
        "retmax": "4",
        "idlist": ["31456127", "36527918", "29276734", "34567890"],
        "translationset": [
            {"from": "gut microbiota",
             "to": '"gastrointestinal microbiome"[MeSH Terms] OR gut microbiota[All Fields]'},
            {"from": "depression",
             "to": '"depression"[MeSH Terms] OR "depressive disorder"[MeSH Terms]'},
        ],
        "querytranslation": (
            '("gastrointestinal microbiome"[MeSH Terms] OR gut microbiota[All Fields]) '
            'AND ("depression"[MeSH Terms] OR "depressive disorder"[MeSH Terms])'
        ),
    }
}


def _article(pmid, title, abstract, journal, year, mesh, authors):
    mesh_xml = "".join(
        f"<MeshHeading><DescriptorName MajorTopicYN='N'>{m}</DescriptorName></MeshHeading>"
        for m in mesh
    )
    auth_xml = "".join(
        f"<Author><LastName>{a[0]}</LastName><Initials>{a[1]}</Initials></Author>"
        for a in authors
    )
    return f"""
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">{pmid}</PMID>
      <Article PubModel="Print">
        <Journal>
          <ISOAbbreviation>{journal}</ISOAbbreviation>
          <JournalIssue><PubDate><Year>{year}</Year><Month>Mar</Month></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>{title}</ArticleTitle>
        <Abstract>{abstract}</Abstract>
        <AuthorList>{auth_xml}</AuthorList>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>{mesh_xml}</MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">{pmid}</ArticleId>
        <ArticleId IdType="doi">10.1000/test.{pmid}</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>"""


EFETCH_XML = ("<?xml version='1.0'?><PubmedArticleSet>" + "".join([
    _article(
        "31456127",
        "The gut-brain axis: microbial regulation of depressive behaviour",
        "<AbstractText Label='BACKGROUND' NlmCategory='BACKGROUND'>Accumulating evidence "
        "links the intestinal microbiome to mood regulation through the gut-brain axis."
        "</AbstractText>"
        "<AbstractText Label='METHODS' NlmCategory='METHODS'>We performed 16S rRNA "
        "sequencing on faecal samples from 212 patients with major depressive disorder "
        "and 210 matched healthy controls, and correlated taxa abundance with HAM-D "
        "scores.</AbstractText>"
        "<AbstractText Label='RESULTS' NlmCategory='RESULTS'>Patients showed reduced "
        "Faecalibacterium and Coprococcus abundance. Lower short-chain fatty acid "
        "producers were associated with higher symptom severity (r = -0.41, p &lt; 0.001). "
        "Serum inflammatory markers partially mediated this association.</AbstractText>"
        "<AbstractText Label='CONCLUSIONS' NlmCategory='CONCLUSIONS'>Depleted "
        "butyrate-producing bacteria are associated with depressive symptom severity, "
        "supporting a microbiota-inflammation-mood pathway.</AbstractText>",
        "Nat Microbiol", "2019",
        ["Gastrointestinal Microbiome", "Depressive Disorder, Major", "Brain-Gut Axis",
         "Fatty Acids, Volatile"],
        [("Valles-Colomer", "M"), ("Falony", "G"), ("Raes", "J")],
    ),
    _article(
        "36527918",
        "Probiotic supplementation and depressive symptoms: a randomised controlled trial",
        "<AbstractText Label='OBJECTIVE'>To test whether an eight-week multi-strain "
        "probiotic reduces depressive symptoms as an adjunct to standard antidepressant "
        "therapy.</AbstractText>"
        "<AbstractText Label='RESULTS'>In 147 randomised adults, the probiotic arm "
        "showed a greater reduction in HAM-D score than placebo (mean difference -2.8, "
        "95% CI -4.1 to -1.5). Effects were largest in participants with elevated "
        "baseline inflammation.</AbstractText>"
        "<AbstractText Label='CONCLUSIONS'>Adjunctive probiotics produced a modest but "
        "significant improvement in depressive symptoms.</AbstractText>",
        "JAMA Psychiatry", "2023",
        ["Probiotics", "Depression", "Randomized Controlled Trial as Topic"],
        [("Schaub", "A"), ("Lang", "U")],
    ),
    _article(
        "29276734",
        "Vagal signalling mediates microbiota effects on anxiety-like behaviour in mice",
        "<AbstractText>Germ-free mice display altered anxiety-like behaviour that is "
        "normalised by colonisation with a conventional microbiota. Subdiaphragmatic "
        "vagotomy abolished the behavioural effect of Lactobacillus rhamnosus, "
        "demonstrating that vagal afferents are required for microbial modulation of "
        "central GABA receptor expression and stress reactivity.</AbstractText>",
        "Proc Natl Acad Sci U S A", "2018",
        ["Vagus Nerve", "Gastrointestinal Microbiome", "Anxiety", "Mice"],
        [("Bravo", "J"), ("Cryan", "J")],
    ),
    # Record with no abstract — must still be indexed via title + MeSH.
    _article(
        "34567890",
        "Faecal microbiota transplantation in treatment-resistant depression: a case series",
        "", "Transl Psychiatry", "2021",
        ["Fecal Microbiota Transplantation", "Depressive Disorder, Treatment-Resistant"],
        [("Doll", "J")],
    ),
]) + "</PubmedArticleSet>").encode("utf-8")

ESPELL_XML = b"<?xml version='1.0'?><eSpellResult><CorrectedQuery></CorrectedQuery></eSpellResult>"


class FakeResponse:
    def __init__(self, content: bytes, payload=None):
        self.content = content
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


CALLS = {"espell": 0, "esearch": 0, "efetch": 0}


def fake_get(self, endpoint, params):  # noqa: ARG001
    if endpoint.startswith("espell"):
        CALLS["espell"] += 1
        return FakeResponse(ESPELL_XML)
    if endpoint.startswith("esearch"):
        CALLS["esearch"] += 1
        return FakeResponse(b"", ESEARCH_JSON)
    if endpoint.startswith("efetch"):
        CALLS["efetch"] += 1
        return FakeResponse(EFETCH_XML)
    raise AssertionError(f"unexpected endpoint {endpoint}")


pc.PubMedClient._get = fake_get  # monkeypatch before the pipeline is imported

from rag.chunker import chunk_article  # noqa: E402
from rag.embedder import embedder  # noqa: E402
from rag.pipeline import pipeline  # noqa: E402
from rag.store import store  # noqa: E402

# --------------------------------------------------------------------------- #
PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "  PASS" if condition else "  FAIL"
    print(f"{mark}  {name}" + (f"   -> {detail}" if detail and not condition else ""))


def main() -> int:
    print("=" * 74)
    print("PubMed RAG — offline self-test")
    print(f"Embedding backend : {embedder.kind}  ({embedder.name}, dim={embedder.dim})")
    print(f"Scratch DB        : {config.DB_PATH}")
    print("=" * 74)

    # ---------------- 1. XML parsing -----------------------------------
    print("\n[1] NCBI XML parsing")
    arts = pc.pubmed_client.efetch(["31456127", "36527918", "29276734", "34567890"])
    check("parses 4 PubmedArticle records", len(arts) == 4, f"got {len(arts)}")
    a0 = next(a for a in arts if a["pmid"] == "31456127")
    check("title extracted", a0["title"].startswith("The gut-brain axis"))
    check("structured abstract labels kept", "Results:" in a0["abstract"])
    check("html entity decoded (&lt; -> <)", "p < 0.001" in a0["abstract"])
    check("authors parsed", a0["authors"][:1] == ["Valles-Colomer M"], str(a0["authors"]))
    check("journal + year parsed", a0["journal"] == "Nat Microbiol" and a0["year"] == "2019")
    check("doi parsed", a0["doi"] == "10.1000/test.31456127")
    check("mesh headings parsed", "Brain-Gut Axis" in a0["mesh_terms"])

    # ---------------- 2. Chunking --------------------------------------
    print("\n[2] Chunking")
    chunks = chunk_article(a0)
    check("produces multiple chunks", len(chunks) >= 2, f"got {len(chunks)}")
    check("chunk 0 is title+MeSH", chunks[0]["section"] == "Title")
    check("chunk ids unique", len({c["chunk_id"] for c in chunks}) == len(chunks))
    no_abs = next(a for a in arts if a["pmid"] == "34567890")
    check("abstract-less record still chunked", len(chunk_article(no_abs)) == 1)

    # ---------------- 3. MeSH extraction -------------------------------
    print("\n[3] Query understanding")
    mesh = pc.PubMedClient._mesh_from_translation(
        ESEARCH_JSON["esearchresult"]["querytranslation"]
    )
    check("MeSH terms pulled from translation",
          "gastrointestinal microbiome" in mesh and "depression" in mesh, str(mesh))
    u = pipeline.understand('"The gut-brain axis"')
    check("quoted query -> exact title mode",
          u["field"] == "title" and "[Title]" in u["pubmed_query"], u["pubmed_query"])
    u2 = pipeline.understand("how does gut microbiota influence depression in adults?")
    check("stopwords removed from long query",
          "how" not in u2["terms"] and "microbiota" in u2["terms"], str(u2["terms"]))

    # ---------------- 4. End-to-end search -----------------------------
    print("\n[4] End-to-end search")
    res = pipeline.search("how does gut microbiota influence depression?", top_k=5)
    check("returns ranked results", len(res["results"]) >= 3, str(len(res["results"])))
    check("results sorted by score",
          all(res["results"][i]["score"] >= res["results"][i + 1]["score"]
              for i in range(len(res["results"]) - 1)))
    check("ranks assigned 1..n",
          [d["rank"] for d in res["results"]] == list(range(1, len(res["results"]) + 1)))
    check("MeSH surfaced to UI", len(res["stages"]["mesh_terms"]) >= 2,
          str(res["stages"]["mesh_terms"]))
    check("chunks indexed", res["stages"]["chunks_indexed"] >= 6,
          str(res["stages"]["chunks_indexed"]))
    check("total match count surfaced", res["stages"]["total_matches"] == 18452)
    check("matched passages attached",
          all(d["matched_passages"] for d in res["results"]))
    check("hybrid sub-scores present",
          "dense_weight" in res["stages"]["retrieval"])

    # ---------------- 5. Generation ------------------------------------
    print("\n[5] Grounded answer generation")
    ans = res["answer"]
    check("answer is non-empty", len(ans["answer"]) > 80)
    check("answer is grounded", ans["grounded"] is True)
    check("answer carries citations", len(ans["citations"]) >= 1)
    cited = {c["pmid"] for c in ans["citations"]}
    retrieved = {d["pmid"] for d in res["results"]}
    check("every citation traces to a retrieved doc", cited.issubset(retrieved),
          f"{cited - retrieved}")
    import re as _re
    markers = {int(m) for m in _re.findall(r"\[(\d+)\]", ans["answer"])}
    numbers = {c["n"] for c in ans["citations"]}
    check("every [n] marker has a matching citation", markers.issubset(numbers),
          f"markers={markers} citations={numbers}")

    # ---------------- 6. Caching ---------------------------------------
    print("\n[6] Caching")
    before = dict(CALLS)
    res2 = pipeline.search("how does gut microbiota influence depression?", top_k=5)
    check("second search hits zero EFetch calls", CALLS["efetch"] == before["efetch"],
          f"{before['efetch']} -> {CALLS['efetch']}")
    check("cached run is not slower", res2["took_ms"] <= max(res["took_ms"], 1) * 3,
          f"{res['took_ms']}ms -> {res2['took_ms']}ms")
    st = store.stats()
    check("articles persisted", st["articles"] == 4, str(st))
    check("embeddings persisted", st["embeddings"] >= 6, str(st))
    check("query log written", st["queries"] >= 2, str(st))

    # ---------------- 7. Semantic behaviour ----------------------------
    print("\n[7] Semantic behaviour")
    r_vagus = pipeline.search("nerve pathway linking bacteria to anxiety in rodents",
                              top_k=4, with_answer=False)
    top_pmid = r_vagus["results"][0]["pmid"] if r_vagus["results"] else None
    check("vocabulary-mismatch query ranks the vagus paper first",
          top_pmid == "29276734",
          f"top was {top_pmid} ({(r_vagus['results'][0]['title'][:60] if r_vagus['results'] else '-')})")
    r_trial = pipeline.search("randomised trial of live bacteria supplements for mood",
                              top_k=4, with_answer=False)
    top2 = r_trial["results"][0]["pmid"] if r_trial["results"] else None
    check("RCT-intent query ranks the probiotic trial first", top2 == "36527918",
          f"top was {top2}")

    # ---------------- 8. Similar / article ------------------------------
    print("\n[8] Article + similar endpoints")
    art = pipeline.article("36527918")
    check("article lookup works", art is not None and art["pmid"] == "36527918")
    sim = pipeline.similar("31456127", top_k=3)
    check("similar excludes the seed", all(s["pmid"] != "31456127" for s in sim))
    check("similar returns neighbours", len(sim) >= 1, str(len(sim)))

    # ---------------- 9. Flask routes -----------------------------------
    print("\n[9] Flask routes")
    from app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        r = client.get("/")
        check("GET / renders", r.status_code == 200 and b"PubMed Semantic Search" in r.data)

        r = client.get("/api/health")
        check("GET /api/health", r.status_code == 200 and r.get_json()["status"] == "ok")

        r = client.post("/api/search", json={"query": "gut microbiota depression", "top_k": 3})
        body = r.get_json()
        check("POST /api/search", r.status_code == 200 and len(body["results"]) >= 1)
        check("search payload has answer + stages",
              "answer" in body and "stages" in body and "understanding" in body)

        r = client.get("/api/search?q=probiotics+depression&top_k=2&with_answer=false")
        check("GET /api/search alias", r.status_code == 200
              and len(r.get_json()["results"]) >= 1)

        r = client.post("/api/search", json={"query": "   "})
        check("empty query -> 400", r.status_code == 400)

        r = client.post("/api/ask", json={"query": "does probiotic help depression"})
        check("POST /api/ask returns an answer",
              r.status_code == 200 and r.get_json()["answer"]["answer"])

        r = client.get("/api/article/29276734")
        check("GET /api/article/<pmid>", r.status_code == 200
              and r.get_json()["pmid"] == "29276734")

        r = client.get("/api/similar/31456127")
        check("GET /api/similar/<pmid>", r.status_code == 200
              and "results" in r.get_json())

        r = client.get("/api/stats")
        check("GET /api/stats", r.status_code == 200 and "store" in r.get_json())

        r = client.get("/api/nope")
        check("unknown route -> 404 json", r.status_code == 404)

    # ---------------- 10. Static assets ---------------------------------
    print("\n[10] Static assets")
    for rel in ("static/css/style.css", "static/js/app.js", "templates/index.html"):
        check(f"{rel} exists", (ROOT / rel).exists())

    # ---------------- summary -------------------------------------------
    print("\n" + "=" * 74)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  Failing checks:")
        for f in FAIL:
            print(f"    - {f}")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

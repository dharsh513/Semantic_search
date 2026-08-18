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


def _article(pmid, title, abstract, journal, year, volume, issue, pages,
             mesh, authors):
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
          <JournalIssue>
            <Volume>{volume}</Volume>
            <Issue>{issue}</Issue>
            <PubDate><Year>{year}</Year><Month>Mar</Month></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>{title}</ArticleTitle>
        <Pagination><MedlinePgn>{pages}</MedlinePgn></Pagination>
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
        "Nat Microbiol", "2019", "4", "6", "623-32",
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
        "JAMA Psychiatry", "2023", "80", "1", "44-52",
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
        "Proc Natl Acad Sci U S A", "2018", "115", "12", "3047-52",
        ["Vagus Nerve", "Gastrointestinal Microbiome", "Anxiety", "Mice"],
        [("Bravo", "J"), ("Cryan", "J")],
    ),
    # Record with no abstract — must still be indexed via title + MeSH.
    _article(
        "34567890",
        "Faecal microbiota transplantation in treatment-resistant depression: a case series",
        "", "Transl Psychiatry", "2021", "11", "", "e451",
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


UID = 0          # primary test account, filled in by main()
UID2 = 0         # second account, used to prove data isolation
TEST_PASSWORD = "Correct-Horse-Battery-7"
TEST_PASSWORD2 = "Second-Account-Pass-9"


def signed_in_client(app, email, password):
    """A Flask test client carrying a real session cookie."""
    client = app.test_client()
    client.post("/api/auth/login", json={"email": email, "password": password})
    return client


def main() -> int:
    global UID, UID2

    print("=" * 74)
    print("PubMed RAG — offline self-test")
    print(f"Embedding backend : {embedder.kind}  ({embedder.name}, dim={embedder.dim})")
    print(f"Scratch DB        : {config.DB_PATH}")
    print("=" * 74)

    # Accounts have to exist before anything writes history or chat.
    from rag import auth as _auth

    UID = _auth.signup("tester@example.edu", "Test Runner", TEST_PASSWORD)["user"]["id"]
    UID2 = _auth.signup("other@example.edu", "Other Person", TEST_PASSWORD2)["user"]["id"]

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
    check("volume/issue/pages parsed",
          (a0["volume"], a0["issue"], a0["pages"]) == ("4", "6", "623-32"),
          f"{a0['volume']}/{a0['issue']}/{a0['pages']}")

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
    res = pipeline.search("how does gut microbiota influence depression?", top_k=5,
                          user_id=UID)
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
    res2 = pipeline.search("how does gut microbiota influence depression?", top_k=5,
                           user_id=UID)
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
                              top_k=4, with_answer=False, user_id=UID)
    top_pmid = r_vagus["results"][0]["pmid"] if r_vagus["results"] else None
    check("vocabulary-mismatch query ranks the vagus paper first",
          top_pmid == "29276734",
          f"top was {top_pmid} ({(r_vagus['results'][0]['title'][:60] if r_vagus['results'] else '-')})")
    r_trial = pipeline.search("randomised trial of live bacteria supplements for mood",
                              top_k=4, with_answer=False, user_id=UID)
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
    with signed_in_client(app, "tester@example.edu", TEST_PASSWORD) as client:
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

    # ---------------- 10. Citation formatting ----------------------------
    print("\n[10] Citation formatting (Vancouver / NLM)")
    from rag import citations as cite

    ref = cite.vancouver(store.get_articles(["31456127"])["31456127"])
    check("authors then title", ref.startswith("Valles-Colomer M, Falony G, Raes J."), ref)
    check("journal abbreviation present", "Nat Microbiol." in ref, ref)
    check("year;volume(issue):pages locator", "2019;4(6):623-32." in ref, ref)
    check("doi included", "doi:10.1000/test.31456127." in ref, ref)
    check("pmid included", ref.rstrip().endswith("PMID: 31456127."), ref)

    many = {"authors": [f"Author{i} A" for i in range(9)], "title": "T", "year": "2020"}
    check("7+ authors truncate to six + et al",
          cite.author_string(many["authors"]).endswith("et al")
          and cite.author_string(many["authors"]).count(",") == 6,
          cite.author_string(many["authors"]))

    arts = [store.get_articles([p])[p] for p in ("31456127", "36527918")]
    bib = cite.bibtex(arts)
    check("bibtex has one entry per article", bib.count("@article{") == 2)
    check("bibtex escapes & in fields", "\\&" in cite.bibtex(
        [{**arts[0], "title": "Gut & brain"}]))
    check("bibtex page range uses --", "623--32" in bib, bib[:200])

    ris_out = cite.ris(arts)
    check("ris starts each record with TY", ris_out.count("TY  - JOUR") == 2)
    check("ris terminates each record", ris_out.count("ER  - ") == 2)
    check("ris splits start/end pages", "SP  - 623" in ris_out and "EP  - 32" in ris_out)

    # ---------------- 11. Export ------------------------------------------
    print("\n[11] Export formats")
    from rag import export as exporter

    pdf_refs = exporter.build("pdf", arts, mode="references", query="gut microbiota")
    check("references PDF has a PDF header", pdf_refs[:5] == b"%PDF-")
    check("references PDF is non-trivial", len(pdf_refs) > 2000, f"{len(pdf_refs)} bytes")

    full = [{**a, "matched_passages": [{"section": "Results", "text": a["abstract"][:200],
                                        "score": 0.8}]} for a in arts]
    pdf_report = exporter.build(
        "pdf", full, mode="report", query="gut microbiota depression",
        answer=res["answer"], stages=res["stages"],
    )
    check("report PDF is a PDF", pdf_report[:5] == b"%PDF-")
    check("report PDF is larger than the reference list",
          len(pdf_report) > len(pdf_refs),
          f"report={len(pdf_report)} refs={len(pdf_refs)}")

    try:
        from pypdf import PdfReader as _PdfReader
        import io as _io
        text = "\n".join(
            p.extract_text() or "" for p in _PdfReader(_io.BytesIO(pdf_report)).pages
        )
        check("report PDF contains the query", "gut microbiota depression" in text, "")
        check("report PDF contains the reference list", "References" in text, "")
        check("report PDF contains an abstract body",
              "Accumulating evidence" in text or "Abstracts" in text, "")
        check("report PDF is paginated", "Page 1" in text, "")
    except ImportError:
        print("  SKIP  PDF text extraction (pypdf not installed)")

    check("bib export encodes utf-8", exporter.build("bib", arts).startswith(b"@article{"))
    check("ris export encodes utf-8", exporter.build("ris", arts).startswith(b"TY  - JOUR"))
    check("txt export is numbered", exporter.build("txt", arts).startswith(b"1. "))
    name = exporter.filename("pdf", "gut microbiota & depression!", "report")
    check("filename is filesystem-safe",
          name.endswith(".pdf") and " " not in name and "&" not in name, name)

    try:
        exporter.build("docx", arts)
        check("unknown format rejected", False, "no exception raised")
    except ValueError:
        check("unknown format rejected", True)

    # ---------------- 12. Search history ----------------------------------
    print("\n[12] Search history")
    hist = store.history(UID, limit=50)
    check("searches were recorded", len(hist) >= 3, str(len(hist)))
    check("newest first", hist[0]["created_at"] >= hist[-1]["created_at"])
    latest = hist[0]
    check("history row carries pmids", len(latest["pmids"]) >= 1)

    full_record = store.get_search(latest["id"], user_id=UID)
    check("snapshot stores results", len(full_record["snapshot"].get("results", [])) >= 1)
    check("snapshot stores the answer",
          "answer" in full_record["snapshot"])
    check("snapshot restores without NCBI",
          {d["pmid"] for d in full_record["snapshot"]["results"]}.issubset(
              set(full_record["pmids"])))

    store.set_pinned(latest["id"], True, UID)
    check("pin persists", store.get_search(latest["id"], user_id=UID)["pinned"] == 1)
    check("pinned rows sort first", store.history(UID, limit=50)[0]["id"] == latest["id"])

    before = len(store.history(UID, limit=200))
    kept = store.clear_history(UID, keep_pinned=True)
    after = store.history(UID, limit=200)
    check("clear keeps pinned rows",
          len(after) == 1 and after[0]["id"] == latest["id"],
          f"before={before} deleted={kept} left={len(after)}")

    filtered = store.history(UID, limit=50, search="microbiota")
    check("history is searchable", all("microbiota" in h["query"] for h in filtered))
    check("delete removes a row", store.delete_search(latest["id"], UID) is True)
    check("deleting a missing row returns False", store.delete_search(999999, UID) is False)

    # ---------------- 13. Per-paper chat ----------------------------------
    print("\n[13] Per-paper chat")
    from rag import paper_chat

    check("methods intent detected",
          paper_chat.detect_intent("what sample size did they use?") == "Methods")
    check("results intent detected",
          paper_chat.detect_intent("what were the main findings?") == "Results")
    check("conclusion intent detected (verb form)",
          paper_chat.detect_intent("what do the authors conclude?") == "Conclusions",
          str(paper_chat.detect_intent("what do the authors conclude?")))
    check("conclusion intent detected (noun form)",
          paper_chat.detect_intent("what is the conclusion of this study?") == "Conclusions")
    check("background intent detected",
          paper_chat.detect_intent("why did they run this study?") == "Background")
    check("no intent on a bare question",
          paper_chat.detect_intent("tell me about it") is None,
          str(paper_chat.detect_intent("tell me about it")))

    m = paper_chat.answer_question("31456127", "how many patients were studied?")
    check("methods question hits the methods sentence",
          "212" in m["answer"], m["answer"][:120])
    check("evidence is returned", len(m["evidence"]) >= 1)
    check("evidence is verbatim from the abstract",
          all(e["text"] in store.get_articles(["31456127"])["31456127"]["abstract"]
              for e in m["evidence"]))
    check("evidence carries section labels",
          any(e["section"] in {"Background", "Methods", "Results", "Conclusions"}
              for e in m["evidence"]), str([e["section"] for e in m["evidence"]]))

    r = paper_chat.answer_question("31456127", "what were the main results?")
    check("results question hits the results sentence",
          "Faecalibacterium" in r["answer"] or "short-chain" in r["answer"],
          r["answer"][:120])

    s = paper_chat.answer_question("31456127", "summarise this paper")
    check("summary spans multiple sections",
          len({e["section"] for e in s["evidence"]}) >= 2,
          str([e["section"] for e in s["evidence"]]))
    check("summary is prefixed", s["answer"].startswith("In short:"))

    off = paper_chat.answer_question(
        "31456127", "what is the boiling point of liquid helium at 3 atmospheres?")
    check("off-topic question yields low confidence",
          off["confidence"] < paper_chat.LOW_CONFIDENCE, str(off["confidence"]))
    check("off-topic question declines instead of asserting",
          "does not directly address" in off["answer"], off["answer"][:120])
    check("off-topic reply is flagged ungrounded", off["grounded"] is False)
    check("off-topic has zero lexical overlap",
          off["signals"]["lexical_cover"] == 0.0, str(off["signals"]))
    check("on-topic scores higher than off-topic",
          m["confidence"] > off["confidence"],
          f"on={m['confidence']} off={off['confidence']}")
    check("confidence is model-scale independent (not raw cosine)",
          m["confidence"] != m["signals"]["top_score"])

    mesh_q = paper_chat.answer_question("31456127", "what MeSH terms is this indexed under?")
    check("mesh question answered from metadata",
          "Brain-Gut Axis" in mesh_q["answer"], mesh_q["answer"][:120])

    no_abs = paper_chat.answer_question("34567890", "what were the results?")
    check("abstract-less record says so",
          "no abstract" in no_abs["answer"].lower(), no_abs["answer"][:120])
    check("abstract-less record is ungrounded", no_abs["grounded"] is False)

    sugg = paper_chat.suggested_questions(store.get_articles(["36527918"])["36527918"])
    check("suggestions generated", 2 <= len(sugg) <= 5, str(len(sugg)))

    store.clear_chat(UID, "31456127")
    paper_chat.chat(UID, "31456127", "what methods were used?")
    transcript = store.chat_history(UID, "31456127")
    check("chat persists both turns", len(transcript) == 2, str(len(transcript)))
    check("chat roles are ordered user then assistant",
          [t["role"] for t in transcript] == ["user", "assistant"])
    check("stored assistant turn keeps its evidence", len(transcript[1]["evidence"]) >= 1)
    check("clear_chat empties the transcript",
          store.clear_chat(UID, "31456127") >= 2 and store.chat_history(UID, "31456127") == [])

    # ---------------- 14. New Flask routes --------------------------------
    print("\n[14] Flask routes — export, history, chat")
    with signed_in_client(app, "tester@example.edu", TEST_PASSWORD) as client:
        client.post("/api/search", json={"query": "gut microbiota depression", "top_k": 3})

        r = client.get("/api/history")
        body = r.get_json()
        check("GET /api/history", r.status_code == 200 and len(body["items"]) >= 1)
        hid = body["items"][0]["id"]

        r = client.get(f"/api/history/{hid}")
        restored = r.get_json()
        check("GET /api/history/<id> restores results",
              r.status_code == 200 and len(restored["results"]) >= 1)
        check("restored payload is flagged", restored.get("from_history") is True)

        r = client.post(f"/api/history/{hid}/pin", json={"pinned": True})
        check("POST /api/history/<id>/pin", r.status_code == 200
              and r.get_json()["pinned"] is True)

        r = client.get("/api/history/999999")
        check("missing history id -> 404", r.status_code == 404)

        pmids = [d["pmid"] for d in restored["results"]][:2]

        r = client.post("/api/export", json={"pmids": pmids, "format": "pdf",
                                             "mode": "references", "query": "test"})
        check("POST /api/export pdf", r.status_code == 200
              and r.data[:5] == b"%PDF-")
        check("export sets a download filename",
              "attachment" in r.headers.get("Content-Disposition", "")
              and r.headers.get("X-Export-Filename", "").endswith(".pdf"))
        check("export content-type is pdf",
              r.headers["Content-Type"].startswith("application/pdf"))

        r = client.post("/api/export", json={"search_id": hid, "format": "pdf",
                                             "mode": "report"})
        check("export by search_id builds a report",
              r.status_code == 200 and r.data[:5] == b"%PDF-")

        r = client.post("/api/export", json={"pmids": pmids, "format": "bib"})
        check("POST /api/export bibtex",
              r.status_code == 200 and r.data.startswith(b"@article{"))

        r = client.post("/api/export", json={"pmids": pmids, "format": "ris"})
        check("POST /api/export ris", r.status_code == 200
              and r.data.startswith(b"TY  - JOUR"))

        r = client.post("/api/export", json={"pmids": [], "format": "pdf"})
        check("export with no selection -> 400", r.status_code == 400)

        r = client.post("/api/export", json={"pmids": pmids, "format": "docx"})
        check("export with bad format -> 400", r.status_code == 400)

        r = client.post("/api/citations", json={"pmids": pmids})
        cites = r.get_json()["citations"]
        check("POST /api/citations", r.status_code == 200 and len(cites) == len(pmids))
        check("citation preview is numbered from 1", cites[0]["n"] == 1)

        r = client.get("/api/chat/31456127")
        check("GET /api/chat/<pmid>", r.status_code == 200
              and "suggestions" in r.get_json())

        r = client.post("/api/chat/31456127", json={"question": "what methods were used?"})
        chat_body = r.get_json()
        check("POST /api/chat/<pmid>", r.status_code == 200 and chat_body["answer"])
        check("chat response carries confidence + evidence",
              "confidence" in chat_body and len(chat_body["evidence"]) >= 1)

        r = client.post("/api/chat/31456127", json={"question": "  "})
        check("empty question -> 400", r.status_code == 400)

        r = client.get("/api/chat/31456127")
        check("transcript is returned on reload",
              len(r.get_json()["messages"]) >= 2)

        r = client.delete("/api/chat/31456127")
        check("DELETE /api/chat/<pmid>", r.status_code == 200
              and r.get_json()["deleted"] >= 2)

        r = client.delete(f"/api/history/{hid}")
        check("DELETE /api/history/<id>", r.status_code == 200)

    # ---------------- 15. Authentication ----------------------------------
    print("\n[15] Authentication")
    from rag import auth

    # -- hashing --
    h = auth.hash_password("Correct-Horse-Battery-7")
    check("hash is not the password", "Correct-Horse-Battery-7" not in h)
    check("hash records its algorithm and cost", h.startswith("pbkdf2_sha256$260000$"),
          h[:40])
    check("same password hashes differently each time",
          auth.hash_password("same-password-x") != auth.hash_password("same-password-x"))
    check("correct password verifies",
          auth.verify_password("Correct-Horse-Battery-7", h)[0] is True)
    check("wrong password rejected",
          auth.verify_password("Correct-Horse-Battery-8", h)[0] is False)
    check("garbage hash rejected without raising",
          auth.verify_password("anything", "not-a-hash")[0] is False)

    weak = auth.hash_password("Correct-Horse-Battery-7", iterations=1000)
    ok_weak, needs = auth.verify_password("Correct-Horse-Battery-7", weak)
    check("low-iteration hash still verifies", ok_weak is True)
    check("low-iteration hash is flagged for upgrade", needs is True)
    check("current-cost hash is not flagged",
          auth.verify_password("Correct-Horse-Battery-7", h)[1] is False)

    # -- validation --
    for bad in ("", "nope", "a@b", "a b@c.com", "no-at-sign.com"):
        try:
            auth.validate_email(bad)
            check(f"rejects invalid email {bad!r}", False, "accepted")
        except auth.AuthError:
            check(f"rejects invalid email {bad!r}", True)
    check("accepts a normal address",
          auth.validate_email("  Arul@Uni.EDU ") == "arul@uni.edu")

    check("rejects short passwords", auth.password_problems("Ab1!x") is not None)
    check("rejects common passwords", auth.password_problems("password123") is not None)
    check("rejects all-numeric passwords", auth.password_problems("9182736450") is not None)
    check("rejects low-variety passwords", auth.password_problems("aaaabbbb") is not None)
    check("rejects password containing the email",
          auth.password_problems("arulrocks99A", email="arul@uni.edu") is not None)
    check("accepts a reasonable password",
          auth.password_problems("Ribosome-42-Coffee") is None)

    check("strength rises with quality",
          auth.password_strength("Ribosome-42-Coffee")["score"]
          > auth.password_strength("abcd1234")["score"])
    check("common password scores zero",
          auth.password_strength("password123")["score"] == 0)

    # -- accounts --
    try:
        auth.signup("tester@example.edu", "Duplicate", "Another-Good-Pass-3")
        check("duplicate email rejected", False, "signup succeeded")
    except auth.AuthError as exc:
        check("duplicate email rejected", exc.status == 409)

    try:
        auth.signup("weakling@example.edu", "Weak Person", "password")
        check("weak password rejected at signup", False, "signup succeeded")
    except auth.AuthError as exc:
        check("weak password rejected at signup", exc.field == "password")

    check("password digest never leaves the store layer",
          "password_hash" not in store.user_by_id(UID))
    check("user lookup by email works",
          store.user_by_email("tester@example.edu")["id"] == UID)
    check("account count is accurate", store.count_users() == 2)

    # ---------------- 16. Auth routes and gating --------------------------
    print("\n[16] Auth routes and gating")
    PROTECTED = [
        ("GET", "/api/health"), ("GET", "/api/stats"), ("GET", "/api/history"),
        ("POST", "/api/search"), ("POST", "/api/export"), ("POST", "/api/citations"),
        ("GET", "/api/chat/31456127"), ("GET", "/api/article/31456127"),
        ("GET", "/api/similar/31456127"),
    ]
    with app.test_client() as anon:
        blocked = []
        for method, path in PROTECTED:
            r = anon.open(path, method=method, json={})
            if r.status_code != 401 or not (r.get_json() or {}).get("auth_required"):
                blocked.append(f"{path}={r.status_code}")
        check("every API route rejects anonymous callers with 401",
              not blocked, ", ".join(blocked))

        r = anon.get("/")
        check("anonymous page request redirects to login",
              r.status_code == 302 and "/login" in r.headers.get("Location", ""),
              r.headers.get("Location", ""))

        r = anon.get("/login")
        check("GET /login renders the auth page",
              r.status_code == 200 and b"Create account" in r.data)
        check("auth page ships the helix animation", b"helix" in r.data)

        r = anon.post("/api/auth/login",
                      json={"email": "tester@example.edu", "password": "wrong-password"})
        check("wrong password -> 401", r.status_code == 401)
        check("error does not reveal whether the account exists",
              "incorrect" in r.get_json()["error"].lower())

        r = anon.post("/api/auth/login",
                      json={"email": "ghost@example.edu", "password": "wrong-password"})
        check("unknown email gives the same message as a wrong password",
              r.status_code == 401 and "incorrect" in r.get_json()["error"].lower())

    # a real signed-in session
    with app.test_client() as client:
        r = client.post("/api/auth/login",
                        json={"email": "tester@example.edu", "password": TEST_PASSWORD})
        check("POST /api/auth/login succeeds", r.status_code == 200)
        check("login returns the user without a password hash",
              "password_hash" not in r.get_json()["user"])

        cookie_header = "; ".join(h for h in r.headers.getlist("Set-Cookie"))
        check("session cookie is HttpOnly", "HttpOnly" in cookie_header, cookie_header)
        check("session cookie is SameSite=Lax", "SameSite=Lax" in cookie_header,
              cookie_header)

        r = client.get("/api/auth/me")
        check("GET /api/auth/me identifies the session",
              r.get_json()["user"]["email"] == "tester@example.edu")

        r = client.get("/")
        check("signed-in page request renders the app", r.status_code == 200)

        r = client.get("/login")
        check("signed-in user visiting /login is sent to the app", r.status_code == 302)

        r = client.post("/api/search", json={"query": "gut microbiota", "top_k": 2})
        check("signed-in search works", r.status_code == 200)

        r = client.post("/api/auth/logout")
        check("POST /api/auth/logout succeeds", r.status_code == 200)

        r = client.get("/api/health")
        check("session is dead after logout", r.status_code == 401)

    # -- redirect target safety --
    r = app.test_client().post("/api/auth/login", json={
        "email": "tester@example.edu", "password": TEST_PASSWORD,
        "next": "https://evil.example.com/steal"})
    check("off-site redirect target is discarded",
          r.get_json()["next"] == "/", str(r.get_json().get("next")))

    r = app.test_client().post("/api/auth/login", json={
        "email": "tester@example.edu", "password": TEST_PASSWORD,
        "next": "//evil.example.com"})
    check("protocol-relative redirect target is discarded",
          r.get_json()["next"] == "/", str(r.get_json().get("next")))

    r = app.test_client().post("/api/auth/login", json={
        "email": "tester@example.edu", "password": TEST_PASSWORD,
        "next": "/?q=aspirin"})
    check("same-site redirect target is kept",
          r.get_json()["next"] == "/?q=aspirin", str(r.get_json().get("next")))

    # -- throttling --
    auth.throttle.reset("tester@example.edu|127.0.0.1")
    with app.test_client() as client:
        codes = [
            client.post("/api/auth/login",
                        json={"email": "tester@example.edu", "password": f"bad{i}"}
                        ).status_code
            for i in range(6)
        ]
        check("repeated bad logins eventually return 429", 429 in codes, str(codes))
        r = client.post("/api/auth/login",
                        json={"email": "tester@example.edu", "password": TEST_PASSWORD})
        check("throttle blocks even the correct password while locked",
              r.status_code == 429, str(r.status_code))
    auth.throttle.reset("tester@example.edu|127.0.0.1")
    r = app.test_client().post("/api/auth/login",
                               json={"email": "tester@example.edu", "password": TEST_PASSWORD})
    check("login works again once the lockout is cleared", r.status_code == 200)

    # ---------------- 17. Per-user data isolation -------------------------
    print("\n[17] Per-user data isolation")
    alice = signed_in_client(app, "tester@example.edu", TEST_PASSWORD)
    bob = signed_in_client(app, "other@example.edu", TEST_PASSWORD2)

    r = alice.post("/api/search", json={"query": "alice private query", "top_k": 2})
    alice_sid = r.get_json()["search_id"]
    check("search records a history id", isinstance(alice_sid, int))

    bob_hist = bob.get("/api/history").get_json()["items"]
    check("bob cannot see alice's searches",
          all(h["query"] != "alice private query" for h in bob_hist),
          str([h["query"] for h in bob_hist]))

    check("bob cannot open alice's search by id",
          bob.get(f"/api/history/{alice_sid}").status_code == 404)
    check("bob cannot delete alice's search",
          bob.delete(f"/api/history/{alice_sid}").status_code == 404)
    check("bob cannot pin alice's search",
          bob.post(f"/api/history/{alice_sid}/pin", json={"pinned": True}).status_code == 404)
    check("bob cannot export alice's search",
          bob.post("/api/export", json={"search_id": alice_sid, "format": "pdf"}
                   ).status_code == 404)
    check("alice can still open her own search",
          alice.get(f"/api/history/{alice_sid}").status_code == 200)

    alice.post("/api/chat/31456127", json={"question": "what methods were used?"})
    bob_chat = bob.get("/api/chat/31456127").get_json()["messages"]
    check("chat transcripts are per user", len(bob_chat) == 0, str(len(bob_chat)))
    alice_chat = alice.get("/api/chat/31456127").get_json()["messages"]
    check("alice keeps her own transcript", len(alice_chat) >= 2)

    bob.post("/api/chat/31456127", json={"question": "what were the findings?"})
    bob.delete("/api/chat/31456127")
    check("clearing bob's chat leaves alice's intact",
          len(alice.get("/api/chat/31456127").get_json()["messages"]) >= 2)

    alice_before = len(alice.get("/api/history").get_json()["items"])
    bob.delete("/api/history?keep_pinned=false")
    alice_after = len(alice.get("/api/history").get_json()["items"])
    check("bob clearing his history leaves alice's intact",
          alice_after == alice_before, f"{alice_before} -> {alice_after}")

    check("the article cache is shared, not duplicated per user",
          store.stats()["articles"] == 4, str(store.stats()["articles"]))

    # ---------------- 18. Password change ---------------------------------
    print("\n[18] Password change")
    auth.signup("charlie@example.edu", "Charlie", "Ribosome-42-Coffee")
    charlie = signed_in_client(app, "charlie@example.edu", "Ribosome-42-Coffee")
    check("charlie starts signed in", charlie.get("/api/health").status_code == 200)

    r = charlie.post("/api/auth/password", json={
        "current_password": "wrong", "new_password": "Mitochondria-99-Blue"})
    check("wrong current password is rejected", r.status_code == 401)

    r = charlie.post("/api/auth/password", json={
        "current_password": "Ribosome-42-Coffee", "new_password": "short"})
    check("weak new password is rejected", r.status_code == 400)

    r = charlie.post("/api/auth/password", json={
        "current_password": "Ribosome-42-Coffee", "new_password": "Mitochondria-99-Blue"})
    check("password change succeeds", r.status_code == 200)
    check("changing the password revokes every session",
          charlie.get("/api/health").status_code == 401)
    check("the old password no longer works",
          app.test_client().post("/api/auth/login", json={
              "email": "charlie@example.edu", "password": "Ribosome-42-Coffee"
          }).status_code == 401)
    auth.throttle.reset("charlie@example.edu|127.0.0.1")
    check("the new password works",
          app.test_client().post("/api/auth/login", json={
              "email": "charlie@example.edu", "password": "Mitochondria-99-Blue"
          }).status_code == 200)

    # ---------------- 19. Upgrading a v1.0.0 database ---------------------
    # A user who ran the first release already has a data/pubmed_cache.sqlite3
    # without volume/issue/pages and without the searches/chat tables. Opening
    # it with the current Store must migrate it in place, not crash.
    print("\n[19] Migrating a v1.0.0 database")
    import sqlite3 as _sq
    from rag.store import Store as _Store

    old_path = os.path.join(_TMP, "legacy_v1.sqlite3")
    legacy = _sq.connect(old_path)
    legacy.executescript(
        """
        CREATE TABLE articles (
            pmid TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '',
            abstract TEXT NOT NULL DEFAULT '', authors TEXT NOT NULL DEFAULT '[]',
            journal TEXT NOT NULL DEFAULT '', pub_date TEXT NOT NULL DEFAULT '',
            year TEXT NOT NULL DEFAULT '', doi TEXT NOT NULL DEFAULT '',
            mesh_terms TEXT NOT NULL DEFAULT '[]', keywords TEXT NOT NULL DEFAULT '[]',
            publication_types TEXT NOT NULL DEFAULT '[]', url TEXT NOT NULL DEFAULT '',
            fetched_at REAL NOT NULL DEFAULT 0);
        CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, pmid TEXT NOT NULL,
            ordinal INTEGER NOT NULL, section TEXT NOT NULL DEFAULT '', text TEXT NOT NULL);
        CREATE TABLE embeddings (chunk_id TEXT PRIMARY KEY, model TEXT NOT NULL,
            dim INTEGER NOT NULL, vector BLOB NOT NULL);
        CREATE TABLE query_log (id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL,
            translated TEXT NOT NULL DEFAULT '', n_results INTEGER NOT NULL DEFAULT 0,
            took_ms INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
        """
    )
    legacy.execute(
        "INSERT INTO articles (pmid, title, abstract, fetched_at) VALUES (?,?,?,?)",
        ("11111111", "A legacy record", "Some cached abstract text.", 9e9),
    )
    legacy.execute(
        "INSERT INTO query_log (query, created_at) VALUES (?,?)", ("old query", 1.0)
    )
    legacy.commit()
    legacy.close()

    try:
        upgraded = _Store(old_path)
        check("v1 database opens without error", True)
    except Exception as exc:  # noqa: BLE001
        check("v1 database opens without error", False, str(exc))
        upgraded = None

    if upgraded:
        cols = {
            r["name"] for r in upgraded._conn().execute("PRAGMA table_info(articles)")
        }
        check("migration adds volume/issue/pages",
              {"volume", "issue", "pages"}.issubset(cols), str(sorted(cols)))
        check("existing cached rows survive",
              upgraded.get_articles(["11111111"])["11111111"]["title"] == "A legacy record")
        check("new tables are created",
              upgraded.stats()["searches"] == 0 and upgraded.stats()["chat_messages"] == 0)

        sid = upgraded.save_search({"user_id": 1, "query": "after upgrade",
                                    "pmids": ["11111111"], "n_results": 1,
                                    "snapshot": {"results": []}})
        check("history works on the upgraded database",
              upgraded.get_search(sid, user_id=1)["query"] == "after upgrade")
        upgraded.add_chat_message(1, "11111111", "user", "hello")
        check("chat works on the upgraded database",
              len(upgraded.chat_history(1, "11111111")) == 1)
        check("re-opening an already-migrated database is a no-op",
              _Store(old_path).stats()["articles"] == 1)

        # Rows written before accounts existed carry user_id 0. The first
        # account created on such a database must inherit them, so upgrading
        # does not look like the history was wiped.
        upgraded.save_search({"query": "pre-accounts search", "pmids": ["11111111"],
                              "n_results": 1, "snapshot": {"results": []}})
        upgraded.add_chat_message(0, "11111111", "user", "asked before logins existed")
        check("pre-accounts rows default to user_id 0",
              upgraded.history(0, limit=10)[0]["query"] == "pre-accounts search")

        new_owner = upgraded.create_user("owner@example.edu", "Owner", "x")
        adopted = upgraded.adopt_orphan_data(new_owner)
        check("adoption moves orphaned searches",
              adopted["searches"] >= 1, str(adopted))
        check("adoption moves orphaned chat messages",
              adopted["chat_messages"] >= 1, str(adopted))
        check("the first account now sees the old history",
              any(h["query"] == "pre-accounts search"
                  for h in upgraded.history(new_owner, limit=10)))
        check("nothing is left orphaned", upgraded.history(0, limit=10) == [])

    # ---------------- 20. Dev-server stability ----------------------------
    # Regression guard. Flask's reloader watches every module in sys.modules;
    # sentence-transformers imports most of `transformers` lazily, so those
    # files appear only after the server is serving and the stat reloader
    # restarts the process mid-request. The visible symptom was the engine
    # badge flipping to "offline" and searches failing at random.
    print("\n[20] Dev-server stability")
    import fnmatch
    import inspect as _inspect
    from werkzeug.serving import run_simple

    import app as app_module

    check("the auto-reloader is off by default", config.USE_RELOADER is False)
    check("werkzeug accepts exclude_patterns",
          "exclude_patterns" in _inspect.signature(run_simple).parameters)

    patterns = app_module.RELOADER_EXCLUDES
    watched_traps = [
        r"C:\Users\arula\anaconda3\envs\dl_gpu\Lib\site-packages\transformers\models\vit\modeling_vit.py",
        "/home/me/project/.venv/lib/python3.11/site-packages/transformers/x.py",
        "/usr/local/lib/python3.11/dist-packages/torch/_torch_docs.py",
    ]
    missed = [
        p for p in watched_traps
        if not any(fnmatch.fnmatch(p, pat) for pat in patterns)
    ]
    check("installed-package paths are excluded from the reloader",
          not missed, "; ".join(missed))

    project_files = [
        str(ROOT / "app.py"), str(ROOT / "rag" / "pipeline.py"),
        str(ROOT / "templates" / "index.html"),
    ]
    over = [
        p for p in project_files
        if any(fnmatch.fnmatch(p, pat) for pat in patterns)
    ]
    check("project files are still watched when the reloader is on",
          not over, "; ".join(over))

    check("app.run is called with the exclusion list",
          "exclude_patterns=RELOADER_EXCLUDES" in (ROOT / "app.py").read_text("utf-8"))
    check("the embedder is preloaded at startup",
          config.PRELOAD_EMBEDDER is True and hasattr(app_module, "_warm_embedder"))

    js = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
    check("the health probe retries before reporting offline",
          "engine: starting" in js and "loadHealth(attempt + 1)" in js)
    check("a stale tab re-probes health on focus", "visibilitychange" in js)

    # ---------------- 21. Static assets -----------------------------------
    print("\n[21] Static assets")
    for rel in ("static/css/style.css", "static/js/app.js", "templates/index.html"):
        check(f"{rel} exists", (ROOT / rel).exists())

    html = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    for node in ("paper-view", "history-drawer", "export-dialog", "chat-form",
                 "results-toolbar", "floating-drawer", "compare-table", "cluster-grid"):
        check(f"index.html wires #{node}", f'id="{node}"' in html)

    js = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
    for route in ("/api/export", "/api/history", "/api/chat/", "/api/citations",
                  "/api/landscape", "/api/compare", "/api/research-gaps"):
        check(f"app.js calls {route}", route in js)

    # ---------------- 22. Research Landscape ------------------------------
    print("\n[22] Research Landscape Engine")
    from rag.landscape import landscape_engine, classify_methodology
    arts_list = [store.get_articles([p])[p] for p in ("31456127", "36527918", "29276734", "34567890")]
    land = landscape_engine.build_landscape(arts_list, query="gut microbiota depression", total_matches=18452)
    check("landscape total papers accurate", land["total_papers"] == 4)
    check("landscape total matches surfaced", land["total_matches"] == 18452)
    check("publication trend calculated", len(land["publication_trend"]) >= 1)
    check("topic clusters generated", len(land["topic_clusters"]) >= 1)
    check("topic clusters have titles and counts",
          all("title" in c and c["paper_count"] >= 1 for c in land["topic_clusters"]))
    check("topic clusters have representative papers",
          all(len(c["representative_papers"]) >= 1 for c in land["topic_clusters"]))
    check("methodology classification works", len(land["methodologies"]) >= 1)
    check("methodology names are structured",
          all("method" in m and "percentage" in m for m in land["methodologies"]))
    check("MeSH distribution calculated", len(land["top_mesh_terms"]) >= 1)
    check("top journals extracted", len(land["top_journals"]) >= 1)
    check("landscape summary generated", len(land["summary"]) > 20)

    # ---------------- 23. Paper Comparison --------------------------------
    print("\n[23] Paper Comparison Engine")
    from rag.comparator import comparator, extract_paper_dimensions
    p_dims = extract_paper_dimensions(arts_list[0])
    check("extracts objective dimension", p_dims["objective"] != "")
    check("extracts dataset dimension", p_dims["dataset"] != "")
    check("extracts methodology dimension", p_dims["methodology"] != "")
    check("extracts findings dimension", p_dims["findings"] != "")
    check("extracts strengths dimension", p_dims["strengths"] != "")
    check("extracts limitations dimension", p_dims["limitations"] != "")

    comp = comparator.compare(["31456127", "36527918"])
    check("comparison paper count accurate", comp["paper_count"] == 2)
    check("comparison side-by-side dimensions present", len(comp["dimensions"]) >= 8)
    check("comparison summary generated", len(comp["summary"]) > 50)
    check("comparative synthesis grounds papers", "31456127" in comp["summary"])

    try:
        comparator.compare(["31456127"])
        check("rejects < 2 papers", False)
    except ValueError:
        check("rejects < 2 papers", True)

    try:
        comparator.compare(["31456127"] * 4)
        check("rejects > 3 papers", False)
    except ValueError:
        check("rejects > 3 papers", True)

    # ---------------- 24. Research Gap Identification --------------------
    print("\n[24] Research Gap & Directions Engine")
    from rag.gaps import gap_engine
    gaps_res = gap_engine.analyze_gaps(["31456127", "36527918", "29276734"])
    check("gaps identified", len(gaps_res["gaps"]) >= 1)
    check("gaps carry category and title",
          all(g.get("category") and g.get("title") for g in gaps_res["gaps"]))
    check("gaps carry confidence rating",
          all(g.get("confidence") in {"high", "medium", "low"} for g in gaps_res["gaps"]))
    check("gaps cite supporting PMIDs",
          all(len(g.get("evidence_pmids", [])) >= 1 for g in gaps_res["gaps"]))
    check("gaps provide supporting observation points",
          all(len(g.get("supporting_points", [])) >= 1 for g in gaps_res["gaps"]))
    check("research directions generated", len(gaps_res["research_directions"]) >= 1)
    check("directions link back to gap title",
          all(d.get("linked_gap_title") for d in gaps_res["research_directions"]))
    check("directions specify suggested methodology",
          all(d.get("suggested_methodology") for d in gaps_res["research_directions"]))
    check("safety disclaimer present", "safety_disclaimer" in gaps_res)

    # ---------------- 25. Pagination & Explainability ---------------------
    print("\n[25] Pagination & Explainable Relevance ('Why this paper?')")
    p_res = pipeline.search("gut microbiota depression", page=1, page_size=2, with_answer=True)
    check("search returns pagination block", "pagination" in p_res)
    check("pagination page is 1", p_res["pagination"]["page"] == 1)
    check("pagination page_size is 2", p_res["pagination"]["page_size"] == 2)
    check("pagination total_matches surfaced", p_res["pagination"]["total_matches"] == 18452)
    check("results limited to page size", len(p_res["results"]) <= 2)

    top_doc = p_res["results"][0]
    check("doc carries semantic_score", "semantic_score" in top_doc)
    check("doc carries bm25_score", "bm25_score" in top_doc)
    check("doc carries hybrid_score", "hybrid_score" in top_doc)
    check("doc carries why_this_paper explainability", "why_this_paper" in top_doc)
    check("why_this_paper has semantic_pct", "semantic_pct" in top_doc["why_this_paper"])
    check("why_this_paper has keyword_pct", "keyword_pct" in top_doc["why_this_paper"])
    check("why_this_paper has overall_pct", "overall_pct" in top_doc["why_this_paper"])
    check("why_this_paper has evidence reasons", len(top_doc["why_this_paper"]["reasons"]) >= 1)

    # ---------------- 26. New API Endpoints -------------------------------
    print("\n[26] New API Endpoints (/api/landscape, /api/compare, /api/research-gaps, /api/papers)")
    with signed_in_client(app, "tester@example.edu", TEST_PASSWORD) as client:
        r_land = client.post("/api/landscape", json={"query": "gut microbiota depression"})
        check("POST /api/landscape succeeds", r_land.status_code == 200)
        check("landscape returns topic clusters", "topic_clusters" in r_land.get_json())
        check("landscape returns publication trend", "publication_trend" in r_land.get_json())

        r_comp = client.post("/api/compare", json={"pmids": ["31456127", "36527918"]})
        check("POST /api/compare succeeds", r_comp.status_code == 200)
        check("compare returns dimensions", "dimensions" in r_comp.get_json())
        check("compare returns summary", "summary" in r_comp.get_json())

        r_comp_bad = client.post("/api/compare", json={"pmids": ["31456127"]})
        check("POST /api/compare with 1 paper -> 400", r_comp_bad.status_code == 400)

        r_gaps = client.post("/api/research-gaps", json={"pmids": ["31456127", "36527918"]})
        check("POST /api/research-gaps succeeds", r_gaps.status_code == 200)
        check("gaps returns identified gaps", "gaps" in r_gaps.get_json())
        check("gaps returns research directions", "research_directions" in r_gaps.get_json())

        r_papers = client.get("/api/papers?pmids=31456127,36527918")
        check("GET /api/papers succeeds", r_papers.status_code == 200 and len(r_papers.get_json()["papers"]) >= 2)

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




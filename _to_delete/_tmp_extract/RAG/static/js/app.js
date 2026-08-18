/* =============================================================================
   PubMed Semantic Search (RAG) — frontend controller
   ========================================================================== */
(function () {
  "use strict";

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const el = {
    form:        $("#search-form"),
    query:       $("#query"),
    field:       $("#field"),
    topK:        $("#top_k"),
    sort:        $("#sort"),
    minDate:     $("#mindate"),
    withAnswer:  $("#with_answer"),
    searchBtn:   $("#search-btn"),
    status:      $("#status"),
    empty:       $("#empty"),
    pipeline:    $("#pipeline"),
    pipeToggle:  $("#pipeline-toggle"),
    pipeBody:    $("#pipeline-body"),
    answerCard:  $("#answer-card"),
    answerText:  $("#answer-text"),
    answerMode:  $("#answer-mode"),
    citations:   $("#answer-citations"),
    resultsSec:  $("#results-section"),
    results:     $("#results"),
    resultsMeta: $("#results-meta"),
    enginePill:  $("#engine-pill"),
    cachePill:   $("#cache-pill"),
    themeBtn:    $("#theme-btn"),
    modal:       $("#modal"),
    modalBody:   $("#modal-content")
  };

  let lastResults = [];
  let inFlight = null;

  const SECTION_LABEL_RE = new RegExp(
    "^(background|objectives?|methods?|materials? and methods?|results?|findings?|" +
    "conclusions?|introduction|aims?|purpose|discussion|design|significance|importance)" +
    "\\s*:\\s*", "i"
  );

  /* ------------------------------------------------------------ helpers */
  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function show(node, on) {
    if (node) node.hidden = !on;
  }

  function setStatus(kind, message) {
    if (!kind) { show(el.status, false); return; }
    el.status.className = "status " + kind;
    el.status.innerHTML =
      (kind === "loading" ? '<span class="spinner"></span>' : "") +
      "<span>" + escapeHtml(message) + "</span>";
    show(el.status, true);
  }

  function authorLine(authors) {
    if (!authors || !authors.length) return "Unknown author";
    if (authors.length === 1) return authors[0];
    if (authors.length <= 3) return authors.join(", ");
    return authors[0] + ", " + authors[1] + " … +" + (authors.length - 2) + " more";
  }

  /* Highlight the query's content words inside a snippet. */
  function highlight(text, terms) {
    let out = escapeHtml(text);
    (terms || []).filter(t => t.length > 3).slice(0, 8).forEach(term => {
      const safe = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      out = out.replace(new RegExp("\\b(" + safe + "\\w{0,3})\\b", "gi"),
                        "<mark>$1</mark>");
    });
    return out;
  }

  /* ------------------------------------------------------------- health */
  async function loadHealth() {
    try {
      const res  = await fetch("/api/health");
      const data = await res.json();
      const emb  = data.embedding || {};
      el.enginePill.textContent = "engine: " + (emb.kind || "?") + " · " + (emb.dim || "?") + "d";
      el.enginePill.title = "Embedding model: " + (emb.model || "unknown");
      const st = data.store || {};
      el.cachePill.textContent = "cache: " + (st.articles || 0) + " papers";
    } catch (_) {
      el.enginePill.textContent = "engine: offline";
    }
  }

  /* ---------------------------------------------------------- pipeline */
  function renderPipeline(data) {
    const u = data.understanding || {};
    const s = data.stages || {};
    const rows = [];

    const row = (label, html) =>
      rows.push('<div class="trace-row"><dt>' + label + "</dt><dd>" + html + "</dd></div>");

    row("Original query", "<code>" + escapeHtml(u.original || data.query) + "</code>");
    if (u.corrected) row("Spelling (ESpell)", "<code>" + escapeHtml(u.corrected) + "</code>");
    row("Search mode", escapeHtml(u.field || "semantic (auto)"));
    row("Sent to ESearch", "<code>" + escapeHtml(s.pubmed_query || "—") + "</code>");

    if (s.query_translation)
      row("NCBI query translation", '<span class="code">' + escapeHtml(s.query_translation) + "</span>");

    if (s.mesh_terms && s.mesh_terms.length)
      row("MeSH terms matched",
          '<div class="tagrow">' + s.mesh_terms
            .map(m => '<span class="tag tag-mesh">' + escapeHtml(m) + "</span>").join("") +
          "</div>");

    if (u.terms && u.terms.length)
      row("Content terms",
          '<div class="tagrow">' + u.terms
            .map(t => '<span class="tag">' + escapeHtml(t) + "</span>").join("") +
          "</div>");

    row("PubMed matches",
        "<strong>" + (s.total_matches || 0).toLocaleString() + "</strong> records · " +
        "fetched top <strong>" + (s.candidates_fetched || 0) + "</strong> as candidates");

    row("Passages indexed",
        "<strong>" + (s.chunks_indexed || 0) + "</strong> chunks embedded · " +
        "<strong>" + (s.chunks_retrieved || 0) + "</strong> retrieved after re-ranking");

    const emb = s.embedding || {};
    row("Embedding model",
        "<code>" + escapeHtml(emb.model || "?") + "</code> (" + escapeHtml(emb.kind || "?") +
        ", " + (emb.dim || "?") + "-d)");

    const r = s.retrieval || {};
    if (r.dense_weight != null)
      row("Hybrid scoring",
          "dense × <strong>" + r.dense_weight + "</strong> + BM25 × <strong>" +
          r.lexical_weight + "</strong>, MMR λ = <strong>" + r.mmr_lambda + "</strong>");

    row("Latency", "<code>" + (data.took_ms || 0) + " ms</code>");

    const notes = (u.notes || []).map(n => '<p class="note">• ' + escapeHtml(n) + "</p>").join("");

    el.pipeBody.innerHTML = notes + '<div class="trace">' + rows.join("") + "</div>";
    show(el.pipeline, true);
  }

  el.pipeToggle.addEventListener("click", () => {
    const open = el.pipeToggle.getAttribute("aria-expanded") === "true";
    el.pipeToggle.setAttribute("aria-expanded", String(!open));
    show(el.pipeBody, !open);
  });

  /* ------------------------------------------------------------- answer */
  function renderAnswer(answer) {
    if (!answer || !answer.answer) { show(el.answerCard, false); return; }

    el.answerMode.textContent =
      answer.mode === "llm" ? "LLM-phrased · grounded" : "extractive · grounded";

    const html = escapeHtml(answer.answer).replace(
      /\[(\d+)\]/g,
      '<button class="cite" data-cite="$1" title="Jump to source $1">$1</button>'
    );
    el.answerText.innerHTML = html;

    el.citations.innerHTML = (answer.citations || []).map(c =>
      '<li id="cite-' + c.n + '">' +
        '<span class="cite-n">' + c.n + "</span>" +
        '<span class="cite-body">' +
          "<strong>" + escapeHtml(c.title || "(untitled)") + "</strong>" +
          escapeHtml(c.citation || "") +
          ' · <a href="' + escapeHtml(c.url) + '" target="_blank" rel="noopener">PMID ' +
          escapeHtml(c.pmid) + "</a>" +
        "</span>" +
      "</li>"
    ).join("");

    $$(".cite", el.answerText).forEach(btn => {
      btn.addEventListener("click", () => {
        const target = $("#cite-" + btn.dataset.cite);
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "center" });
          target.style.transition = "background .3s";
          target.style.background = "var(--accent-soft)";
          setTimeout(() => { target.style.background = ""; }, 1200);
        }
      });
    });

    show(el.answerCard, true);
  }

  /* ------------------------------------------------------------ results */
  function renderResults(data) {
    lastResults = data.results || [];
    const terms = (data.understanding && data.understanding.terms) || [];

    if (!lastResults.length) {
      el.results.innerHTML =
        '<div class="card"><p style="margin:0;color:var(--text-2)">' +
        "No records matched. Try fewer or broader terms, or switch the mode to Semantic (auto)." +
        "</p></div>";
      el.resultsMeta.textContent = "0 results";
      show(el.resultsSec, true);
      return;
    }

    el.resultsMeta.textContent =
      lastResults.length + " of " +
      ((data.stages && data.stages.total_matches) || 0).toLocaleString() +
      " PubMed matches · " + (data.took_ms || 0) + " ms";

    el.results.innerHTML = lastResults.map((d, i) => {
      // Prefer a real abstract passage; the title/MeSH chunk only repeats the
      // heading already shown above the snippet.
      const passages = d.matched_passages || [];
      const top =
        passages.find(p => (p.section || "").toLowerCase() !== "title") ||
        passages[0] || null;

      let snippet = "";
      if (top) {
        let body = String(top.text || "")
          .replace(/\s*MeSH:[^.]*/i, "")          // drop the MeSH keyword bag
          .replace(/^.*?\s—\s/, "")               // drop the repeated title prefix
          // drop the leading structured-abstract label (shown as a chip instead)
          .replace(SECTION_LABEL_RE, "")
          .trim();
        if (!body) body = String(top.text || "");
        if (body.length > 460) body = body.slice(0, 460).trim() + "…";
        const label = (top.section || "").toLowerCase() === "title" ? "" : top.section;
        snippet =
          '<div class="snippet">' +
            (label ? '<span class="sec">' + escapeHtml(label) + "</span>" : "") +
            highlight(body, terms) +
          "</div>";
      }

      const mesh = (d.mesh_terms || []).slice(0, 6).map(
        m => '<span class="tag tag-mesh">' + escapeHtml(m) + "</span>"
      ).join("");

      return (
        '<article class="card fade-in" style="animation-delay:' + (i * 35) + 'ms">' +
          '<div class="card-top">' +
            '<span class="rank">' + (d.rank || i + 1) + "</span>" +
            '<div class="card-main">' +
              '<h3 class="card-title"><a href="' + escapeHtml(d.url) +
                '" target="_blank" rel="noopener">' + escapeHtml(d.title || "(untitled)") + "</a></h3>" +
              '<div class="meta">' +
                "<span>" + escapeHtml(authorLine(d.authors)) + "</span>" +
                (d.journal ? '<span class="sep">|</span><span><em>' + escapeHtml(d.journal) + "</em></span>" : "") +
                (d.pub_date ? '<span class="sep">|</span><span>' + escapeHtml(d.pub_date) + "</span>" : "") +
                '<span class="sep">|</span><span>PMID ' + escapeHtml(d.pmid) + "</span>" +
              "</div>" +
            "</div>" +
            '<div class="relevance">' +
              '<span class="relevance-num">' + (d.relevance != null ? d.relevance : 0) + "</span>" +
              '<span class="relevance-lbl">match</span>' +
              '<span class="bar"><i style="width:' +
                 Math.max(4, Math.min(100, d.relevance || 0)) + '%"></i></span>' +
            "</div>" +
          "</div>" +
          snippet +
          (mesh ? '<div class="mesh-list">' + mesh + "</div>" : "") +
          '<div class="card-foot">' +
            '<button class="btn-ghost" data-abstract="' + escapeHtml(d.pmid) + '">Abstract</button>' +
            '<button class="btn-ghost" data-similar="' + escapeHtml(d.pmid) + '">Similar papers</button>' +
            (d.doi ? '<a class="btn-ghost" href="https://doi.org/' + escapeHtml(d.doi) +
                     '" target="_blank" rel="noopener">DOI</a>' : "") +
            '<span class="spacer"></span>' +
            '<a class="btn-ghost" href="' + escapeHtml(d.url) +
              '" target="_blank" rel="noopener">Open in PubMed ↗</a>' +
          "</div>" +
        "</article>"
      );
    }).join("");

    $$("[data-abstract]", el.results).forEach(b =>
      b.addEventListener("click", () => openAbstract(b.dataset.abstract)));
    $$("[data-similar]", el.results).forEach(b =>
      b.addEventListener("click", () => openSimilar(b.dataset.similar)));

    show(el.resultsSec, true);
  }

  /* -------------------------------------------------------------- modal */
  function openModal(html) {
    el.modalBody.innerHTML = html;
    show(el.modal, true);
    document.body.style.overflow = "hidden";
  }
  function closeModal() {
    show(el.modal, false);
    document.body.style.overflow = "";
  }
  $$("[data-close]", el.modal).forEach(n => n.addEventListener("click", closeModal));
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

  function openAbstract(pmid) {
    const doc = lastResults.find(d => d.pmid === pmid);
    openModal('<h3>Loading…</h3>');
    fetch("/api/article/" + encodeURIComponent(pmid))
      .then(r => r.json())
      .then(a => {
        if (a.error) { openModal("<h3>Not available</h3><p>" + escapeHtml(a.error) + "</p>"); return; }
        openModal(
          "<h3>" + escapeHtml(a.title || "(untitled)") + "</h3>" +
          '<div class="meta">' + escapeHtml(authorLine(a.authors)) +
            (a.journal ? " · <em>" + escapeHtml(a.journal) + "</em>" : "") +
            (a.pub_date ? " · " + escapeHtml(a.pub_date) : "") + "</div>" +
          '<div class="abstract">' +
            escapeHtml(a.abstract || "No abstract is available for this record.") + "</div>" +
          ((a.mesh_terms || []).length
            ? "<h4>MeSH headings</h4><div class='tagrow'>" +
              a.mesh_terms.map(m => "<span class='tag tag-mesh'>" + escapeHtml(m) + "</span>").join("") +
              "</div>"
            : "") +
          "<h4>Links</h4><p><a href='" + escapeHtml(a.url) +
            "' target='_blank' rel='noopener'>PubMed ↗</a>" +
            (a.doi ? " · <a href='https://doi.org/" + escapeHtml(a.doi) +
                     "' target='_blank' rel='noopener'>DOI ↗</a>" : "") + "</p>"
        );
      })
      .catch(() => openModal("<h3>Network error</h3><p>Could not load PMID " +
                             escapeHtml(pmid) + ".</p>" +
                             (doc ? "<div class='abstract'>" + escapeHtml(doc.abstract) + "</div>" : "")));
  }

  function openSimilar(pmid) {
    openModal("<h3>Finding semantically similar papers…</h3><div class='skeleton'></div>");
    fetch("/api/similar/" + encodeURIComponent(pmid))
      .then(r => r.json())
      .then(d => {
        if (d.error) { openModal("<h3>Unavailable</h3><p>" + escapeHtml(d.error) + "</p>"); return; }
        const list = (d.results || []);
        openModal(
          "<h3>Similar to PMID " + escapeHtml(pmid) + "</h3>" +
          (list.length
            ? "<ol style='padding-left:20px;display:grid;gap:12px;margin-top:14px'>" +
              list.map(r =>
                "<li><a href='" + escapeHtml(r.url) + "' target='_blank' rel='noopener'>" +
                escapeHtml(r.title) + "</a><br><span style='color:var(--text-3);font-size:12.5px'>" +
                escapeHtml(r.journal || "") + " " + escapeHtml(r.year || "") +
                " · match " + (r.relevance || 0) + "</span></li>").join("") +
              "</ol>"
            : "<p>No close neighbours found.</p>")
        );
      })
      .catch(() => openModal("<h3>Network error</h3><p>Could not load similar papers.</p>"));
  }

  /* ------------------------------------------------------------- search */
  async function runSearch(query) {
    if (inFlight) inFlight.abort();
    inFlight = new AbortController();

    el.searchBtn.disabled = true;
    $(".btn-label", el.searchBtn).textContent = "Searching…";
    show(el.empty, false);
    show(el.answerCard, false);
    show(el.pipeline, false);
    setStatus("loading", "Querying PubMed, embedding passages and re-ranking…");

    el.results.innerHTML =
      '<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>';
    show(el.resultsSec, true);
    el.resultsMeta.textContent = "";

    const body = {
      query: query,
      top_k: parseInt(el.topK.value, 10) || 10,
      field: el.field.value,
      sort: el.sort.value,
      with_answer: el.withAnswer.checked
    };
    if (el.minDate.value) body.mindate = el.minDate.value + "/01/01";

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: inFlight.signal
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        setStatus("error", data.error || ("Request failed (HTTP " + res.status + ")"));
        el.results.innerHTML = "";
        show(el.resultsSec, false);
        return;
      }

      setStatus(null);
      renderPipeline(data);
      if (body.with_answer) renderAnswer(data.answer); else show(el.answerCard, false);
      renderResults(data);
      loadHealth();

      const url = new URL(window.location);
      url.searchParams.set("q", query);
      history.replaceState(null, "", url);
    } catch (err) {
      if (err.name === "AbortError") return;
      setStatus("error", "Network error: " + err.message);
      el.results.innerHTML = "";
    } finally {
      el.searchBtn.disabled = false;
      $(".btn-label", el.searchBtn).textContent = "Search";
      inFlight = null;
    }
  }

  el.form.addEventListener("submit", e => {
    e.preventDefault();
    const q = el.query.value.trim();
    if (q) runSearch(q);
  });

  $$(".chip").forEach(chip =>
    chip.addEventListener("click", () => {
      el.query.value = chip.textContent.trim();
      runSearch(el.query.value);
    })
  );

  /* -------------------------------------------------------------- theme */
  const savedTheme = (() => { try { return window.__theme; } catch (_) { return null; } })();
  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    window.__theme = t;
    el.themeBtn.textContent = t === "dark" ? "☀" : "◐";
  }
  setTheme(savedTheme ||
    (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  el.themeBtn.addEventListener("click", () =>
    setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));

  /* --------------------------------------------------------------- boot */
  loadHealth();
  const initial = new URLSearchParams(window.location.search).get("q");
  if (initial) { el.query.value = initial; runSearch(initial); }
  el.query.focus();
})();

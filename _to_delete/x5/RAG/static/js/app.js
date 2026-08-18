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
    toolbar:     $("#results-toolbar"),
    selectAll:   $("#select-all"),
    selectCount: $("#select-count"),
    exportBtn:   $("#export-btn"),
    enginePill:  $("#engine-pill"),
    cachePill:   $("#cache-pill"),
    themeBtn:    $("#theme-btn"),
    modal:       $("#modal"),
    modalBody:   $("#modal-content"),
    toast:       $("#toast"),

    paperView:   $("#paper-view"),
    paperTitle:  $("#paper-title"),
    paperMeta:   $("#paper-meta"),
    paperAbs:    $("#paper-abstract"),
    paperMeshWrap: $("#paper-mesh-wrap"),
    paperMesh:   $("#paper-mesh"),
    paperKwWrap: $("#paper-keywords-wrap"),
    paperKw:     $("#paper-keywords"),
    paperLinks:  $("#paper-links"),
    chatThread:  $("#chat-thread"),
    chatSugg:    $("#chat-suggestions"),
    chatForm:    $("#chat-form"),
    chatInput:   $("#chat-input"),
    chatSend:    $("#chat-send"),
    chatClear:   $("#chat-clear"),

    historyBtn:  $("#history-btn"),
    historyDraw: $("#history-drawer"),
    historyList: $("#history-list"),
    historySearch: $("#history-search"),
    historyClear: $("#history-clear"),

    exportDlg:   $("#export-dialog"),
    exportCount: $("#export-count"),
    exportPrev:  $("#export-preview"),
    exportHint:  $("#export-hint"),
    exportGo:    $("#export-go"),

    accountBtn:  $("#account-btn"),
    accountMenu: $("#account-menu"),
    accountStats: $("#account-stats"),
    accountPassword: $("#account-password")
  };

  /* state */
  let lastResults  = [];
  let lastPayload  = null;      // the whole /api/search response
  let selected     = new Set(); // pmids ticked for export
  let currentPaper = null;      // article open in the detail view
  let inFlight     = null;
  let chatBusy     = false;

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

  function show(node, on) { if (node) node.hidden = !on; }

  function setStatus(kind, message) {
    if (!kind) { show(el.status, false); return; }
    el.status.className = "status " + kind;
    el.status.innerHTML =
      (kind === "loading" ? '<span class="spinner"></span>' : "") +
      "<span>" + escapeHtml(message) + "</span>";
    show(el.status, true);
  }

  let toastTimer = null;
  function toast(message, isError) {
    el.toast.className = "toast" + (isError ? " error" : "");
    el.toast.textContent = message;
    show(el.toast, true);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => show(el.toast, false), isError ? 5200 : 3200);
  }

  function authorLine(authors) {
    if (!authors || !authors.length) return "Unknown author";
    if (authors.length === 1) return authors[0];
    if (authors.length <= 3) return authors.join(", ");
    return authors[0] + ", " + authors[1] + " … +" + (authors.length - 2) + " more";
  }

  function timeAgo(seconds) {
    const diff = Math.max(0, Date.now() / 1000 - (seconds || 0));
    if (diff < 60)    return "just now";
    if (diff < 3600)  return Math.floor(diff / 60) + "m ago";
    if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
    if (diff < 604800) return Math.floor(diff / 86400) + "d ago";
    return new Date((seconds || 0) * 1000).toLocaleDateString();
  }

  function highlight(text, terms) {
    let out = escapeHtml(text);
    (terms || []).filter(t => t.length > 3).slice(0, 8).forEach(term => {
      const safe = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      out = out.replace(new RegExp("\\b(" + safe + "\\w{0,3})\\b", "gi"), "<mark>$1</mark>");
    });
    return out;
  }

  function cleanPassage(text) {
    return String(text || "")
      .replace(/\s*MeSH:[^.]*/i, "")
      .replace(/^.*?\s—\s/, "")
      .replace(SECTION_LABEL_RE, "")
      .trim();
  }

  async function api(url, options) {
    const res  = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    // A dropped or revoked session should send the user to sign in again
    // rather than surfacing a confusing "Authentication required" error.
    if (res.status === 401 && data.auth_required) {
      window.location.href = "/login?next=" +
        encodeURIComponent(window.location.pathname + window.location.search);
      throw new Error("Session expired — redirecting to sign in.");
    }
    if (!res.ok || data.error) throw new Error(data.error || ("HTTP " + res.status));
    return data;
  }

  /* ------------------------------------------------------------- health */
  // The embedding model takes a few seconds to load on first boot, and a dev
  // server restart briefly refuses connections. Neither means the engine is
  // gone, so a single failed probe must not be reported as "offline" — retry
  // with backoff first, and keep retrying quietly in the background after.
  let healthRetry = null;

  async function loadHealth(attempt) {
    attempt = attempt || 0;
    clearTimeout(healthRetry);

    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (res.status === 401) return;           // signed out; other code redirects
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();

      const emb = data.embedding || {};
      el.enginePill.textContent =
        "engine: " + (emb.kind || "?") + " · " + (emb.dim || "?") + "d";
      el.enginePill.title = "Embedding model: " + (emb.model || "unknown");
      el.enginePill.classList.remove("pill-muted");
      el.cachePill.textContent = "cache: " + ((data.store || {}).articles || 0) + " papers";
    } catch (_) {
      if (attempt < 4) {
        el.enginePill.textContent = "engine: starting…";
        el.enginePill.title = "Waiting for the embedding model to finish loading.";
        el.enginePill.classList.add("pill-muted");
        healthRetry = setTimeout(() => loadHealth(attempt + 1), 1200 * (attempt + 1));
        return;
      }
      el.enginePill.textContent = "engine: offline";
      el.enginePill.title =
        "The server is not responding. It may be restarting — this will clear itself.";
      el.enginePill.classList.add("pill-muted");
      healthRetry = setTimeout(() => loadHealth(0), 15000);
    }
  }

  // A tab left open across a server restart should recover on its own.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) loadHealth(0);
  });

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
      row("MeSH terms matched", '<div class="tagrow">' + s.mesh_terms
        .map(m => '<span class="tag tag-mesh">' + escapeHtml(m) + "</span>").join("") + "</div>");

    if (u.terms && u.terms.length)
      row("Content terms", '<div class="tagrow">' + u.terms
        .map(t => '<span class="tag">' + escapeHtml(t) + "</span>").join("") + "</div>");

    row("PubMed matches",
        "<strong>" + (s.total_matches || 0).toLocaleString() + "</strong> records · fetched top <strong>" +
        (s.candidates_fetched || 0) + "</strong> as candidates");
    row("Passages indexed",
        "<strong>" + (s.chunks_indexed || 0) + "</strong> chunks embedded · <strong>" +
        (s.chunks_retrieved || 0) + "</strong> retrieved after re-ranking");

    const emb = s.embedding || {};
    row("Embedding model", "<code>" + escapeHtml(emb.model || "?") + "</code> (" +
        escapeHtml(emb.kind || "?") + ", " + (emb.dim || "?") + "-d)");

    const r = s.retrieval || {};
    if (r.dense_weight != null)
      row("Hybrid scoring", "dense × <strong>" + r.dense_weight + "</strong> + BM25 × <strong>" +
          r.lexical_weight + "</strong>, MMR λ = <strong>" + r.mmr_lambda + "</strong>");

    row("Latency", "<code>" + (data.took_ms || 0) + " ms</code>" +
        (data.from_history ? ' <span class="tag">restored from history</span>' : ""));

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

    el.answerText.innerHTML = escapeHtml(answer.answer).replace(
      /\[(\d+)\]/g,
      '<button class="cite" data-cite="$1" title="Jump to source $1">$1</button>'
    );

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
        if (!target) return;
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.style.transition = "background .3s";
        target.style.background = "var(--accent-soft)";
        setTimeout(() => { target.style.background = ""; }, 1200);
      });
    });

    show(el.answerCard, true);
  }

  /* ------------------------------------------------------------ results */
  function renderResults(data) {
    lastResults = data.results || [];
    lastPayload = data;
    selected = new Set(lastResults.map(d => d.pmid));   // everything ticked by default
    const terms = (data.understanding && data.understanding.terms) || [];

    if (!lastResults.length) {
      el.results.innerHTML =
        '<div class="card"><p style="margin:0;color:var(--text-2)">' +
        "No records matched. Try fewer or broader terms, or switch the mode to Semantic (auto)." +
        "</p></div>";
      el.resultsMeta.textContent = "0 results";
      show(el.toolbar, false);
      show(el.resultsSec, true);
      return;
    }

    el.resultsMeta.textContent =
      lastResults.length + " of " +
      ((data.stages && data.stages.total_matches) || 0).toLocaleString() +
      " PubMed matches · " + (data.took_ms || 0) + " ms" +
      (data.from_history ? " · from history" : "");

    el.results.innerHTML = lastResults.map((d, i) => {
      const passages = d.matched_passages || [];
      const top = passages.find(p => (p.section || "").toLowerCase() !== "title") ||
                  passages[0] || null;

      let snippet = "";
      if (top) {
        let body = cleanPassage(top.text) || String(top.text || "");
        if (body.length > 460) body = body.slice(0, 460).trim() + "…";
        const label = (top.section || "").toLowerCase() === "title" ? "" : top.section;
        snippet = '<div class="snippet">' +
          (label ? '<span class="sec">' + escapeHtml(label) + "</span>" : "") +
          highlight(body, terms) + "</div>";
      }

      const mesh = (d.mesh_terms || []).slice(0, 6)
        .map(m => '<span class="tag tag-mesh">' + escapeHtml(m) + "</span>").join("");

      return (
        '<article class="card fade-in selected" data-pmid="' + escapeHtml(d.pmid) +
             '" style="animation-delay:' + (i * 35) + 'ms">' +
          '<div class="card-top">' +
            '<input type="checkbox" class="card-check" checked ' +
              'aria-label="Select for export" data-select="' + escapeHtml(d.pmid) + '" />' +
            '<span class="rank">' + (d.rank || i + 1) + "</span>" +
            '<div class="card-main">' +
              '<h3 class="card-title"><a href="#" data-open="' + escapeHtml(d.pmid) + '">' +
                escapeHtml(d.title || "(untitled)") + "</a></h3>" +
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
              '<span class="bar"><i style="width:' + Math.max(4, Math.min(100, d.relevance || 0)) + '%"></i></span>' +
            "</div>" +
          "</div>" +
          snippet +
          (mesh ? '<div class="mesh-list">' + mesh + "</div>" : "") +
          '<div class="card-foot">' +
            '<button class="btn-ghost" data-open="' + escapeHtml(d.pmid) + '">Read &amp; ask</button>' +
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

    $$("[data-open]", el.results).forEach(b =>
      b.addEventListener("click", e => { e.preventDefault(); openPaper(b.dataset.open); }));
    $$("[data-similar]", el.results).forEach(b =>
      b.addEventListener("click", () => openSimilar(b.dataset.similar)));
    $$("[data-select]", el.results).forEach(cb =>
      cb.addEventListener("change", () => toggleSelect(cb.dataset.select, cb.checked)));

    show(el.toolbar, true);
    syncSelection();
    show(el.resultsSec, true);
  }

  /* ---------------------------------------------------------- selection */
  function toggleSelect(pmid, on) {
    if (on) selected.add(pmid); else selected.delete(pmid);
    const card = el.results.querySelector('[data-pmid="' + CSS.escape(pmid) + '"]');
    if (card) card.classList.toggle("selected", on);
    syncSelection();
  }

  function syncSelection() {
    const n = selected.size, total = lastResults.length;
    el.selectCount.textContent =
      n === 0 ? "Select all"
              : n + " of " + total + " selected";
    el.selectAll.checked = n > 0 && n === total;
    el.selectAll.indeterminate = n > 0 && n < total;
    el.exportBtn.disabled = n === 0;
  }

  el.selectAll.addEventListener("change", () => {
    const on = el.selectAll.checked;
    selected = on ? new Set(lastResults.map(d => d.pmid)) : new Set();
    $$("[data-select]", el.results).forEach(cb => { cb.checked = on; });
    $$(".card", el.results).forEach(c => c.classList.toggle("selected", on));
    syncSelection();
  });

  /* ------------------------------------------------------------- export */
  function selectedPmids() {
    return lastResults.map(d => d.pmid).filter(p => selected.has(p));
  }

  function exportOptions() {
    return {
      mode: ($('input[name="expmode"]:checked') || {}).value || "references",
      format: ($('input[name="expfmt"]:checked') || {}).value || "pdf"
    };
  }

  function updateExportHint() {
    const o = exportOptions();
    const hints = {
      pdf: o.mode === "report"
        ? "A formatted PDF: search parameters, the grounded answer, references, then every abstract."
        : "A formatted PDF containing just the numbered Vancouver reference list.",
      bib: "BibTeX (.bib) for LaTeX, Overleaf, JabRef. Layout options do not apply.",
      ris: "RIS (.ris) for EndNote, Zotero, Mendeley. Layout options do not apply.",
      txt: "Plain-text numbered Vancouver citations, ready to paste."
    };
    el.exportHint.textContent = hints[o.format] || "";
    $$("#export-mode input").forEach(i => { i.disabled = o.format !== "pdf"; });
    $("#export-mode").style.opacity = o.format === "pdf" ? "1" : ".5";
  }

  async function openExport() {
    const pmids = selectedPmids();
    if (!pmids.length) return;
    el.exportCount.textContent =
      pmids.length + " record" + (pmids.length === 1 ? "" : "s") + " selected for export.";
    el.exportPrev.textContent = "Loading preview…";
    updateExportHint();
    show(el.exportDlg, true);

    try {
      const data = await api("/api/citations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pmids: pmids })
      });
      el.exportPrev.innerHTML = "<ol>" + (data.citations || [])
        .map(c => "<li>" + escapeHtml(c.text) + "</li>").join("") + "</ol>";
    } catch (err) {
      el.exportPrev.textContent = "Preview unavailable: " + err.message;
    }
  }

  async function runExport() {
    const pmids = selectedPmids();
    if (!pmids.length) return;
    const o = exportOptions();

    el.exportGo.disabled = true;
    el.exportGo.textContent = "Preparing…";

    try {
      const body = {
        pmids: pmids,
        format: o.format,
        mode: o.format === "pdf" ? o.mode : "references",
        query: (lastPayload && lastPayload.query) || el.query.value || "",
        answer: (lastPayload && lastPayload.answer) || null,
        stages: (lastPayload && lastPayload.stages) || null
      };
      if (lastPayload && lastPayload.search_id) body.search_id = lastPayload.search_id;

      const res = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || ("HTTP " + res.status));
      }

      const blob = await res.blob();
      const name = res.headers.get("X-Export-Filename") ||
                   ("pubmed-export." + o.format);
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);

      show(el.exportDlg, false);
      toast("Downloaded " + name);
    } catch (err) {
      toast("Export failed: " + err.message, true);
    } finally {
      el.exportGo.disabled = false;
      el.exportGo.textContent = "Download";
    }
  }

  el.exportBtn.addEventListener("click", openExport);
  el.exportGo.addEventListener("click", runExport);
  $$('#export-format input, #export-mode input').forEach(i =>
    i.addEventListener("change", updateExportHint));

  /* ------------------------------------------------------------ history */
  async function loadHistory() {
    el.historyList.innerHTML = '<p class="chat-empty">Loading…</p>';
    try {
      const q = (el.historySearch.value || "").trim();
      const data = await api("/api/history?limit=100" +
                             (q ? "&q=" + encodeURIComponent(q) : ""));
      const items = data.items || [];

      if (!items.length) {
        el.historyList.innerHTML =
          '<p class="chat-empty">' +
          (q ? "No past searches match that filter."
             : "No searches yet. Every search you run is saved here automatically.") +
          "</p>";
        return;
      }

      el.historyList.innerHTML = items.map(h =>
        '<div class="hist' + (h.pinned ? " pinned" : "") + '" data-hist="' + h.id + '">' +
          '<div class="hist-main">' +
            '<div class="hist-q">' + escapeHtml(h.query) + "</div>" +
            '<div class="hist-meta">' +
              h.n_results + " result" + (h.n_results === 1 ? "" : "s") +
              (h.total_hits ? " · " + h.total_hits.toLocaleString() + " matches" : "") +
              " · " + h.took_ms + "ms · " + timeAgo(h.created_at) +
            "</div>" +
            ((h.mesh_terms || []).length
              ? '<div class="hist-mesh">' + h.mesh_terms.slice(0, 4)
                  .map(m => '<span class="tag tag-mesh">' + escapeHtml(m) + "</span>").join("") + "</div>"
              : "") +
          "</div>" +
          '<div class="hist-actions">' +
            '<button class="hist-act' + (h.pinned ? " on" : "") + '" data-pin="' + h.id +
              '" title="' + (h.pinned ? "Unpin" : "Pin") + '">★</button>' +
            '<button class="hist-act" data-del="' + h.id + '" title="Delete">🗑</button>' +
          "</div>" +
        "</div>"
      ).join("");

      $$("[data-hist]", el.historyList).forEach(node =>
        node.addEventListener("click", e => {
          if (e.target.closest("[data-pin],[data-del]")) return;
          restoreSearch(node.dataset.hist);
        }));

      $$("[data-pin]", el.historyList).forEach(btn =>
        btn.addEventListener("click", async () => {
          const on = !btn.classList.contains("on");
          try {
            await api("/api/history/" + btn.dataset.pin + "/pin", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ pinned: on })
            });
            loadHistory();
          } catch (err) { toast(err.message, true); }
        }));

      $$("[data-del]", el.historyList).forEach(btn =>
        btn.addEventListener("click", async () => {
          try {
            await api("/api/history/" + btn.dataset.del, { method: "DELETE" });
            loadHistory();
          } catch (err) { toast(err.message, true); }
        }));

    } catch (err) {
      el.historyList.innerHTML =
        '<p class="chat-empty">Could not load history: ' + escapeHtml(err.message) + "</p>";
    }
  }

  async function restoreSearch(id) {
    try {
      const data = await api("/api/history/" + id);
      show(el.historyDraw, false);
      document.body.style.overflow = "";
      el.query.value = data.query || "";
      show(el.empty, false);
      setStatus(null);
      renderPipeline(data);
      renderAnswer(data.answer);
      renderResults(data);
      window.scrollTo({ top: 0, behavior: "smooth" });
      toast("Restored “" + (data.query || "").slice(0, 48) + "” from history");
    } catch (err) {
      toast("Could not restore: " + err.message, true);
    }
  }

  el.historyBtn.addEventListener("click", () => {
    show(el.historyDraw, true);
    document.body.style.overflow = "hidden";
    loadHistory();
  });

  el.historyClear.addEventListener("click", async () => {
    try {
      const data = await api("/api/history?keep_pinned=true", { method: "DELETE" });
      toast("Cleared " + (data.deleted || 0) + " search(es). Pinned ones were kept.");
      loadHistory();
    } catch (err) { toast(err.message, true); }
  });

  let histTimer = null;
  el.historySearch.addEventListener("input", () => {
    clearTimeout(histTimer);
    histTimer = setTimeout(loadHistory, 220);
  });

  /* ------------------------------------------------- paper detail + chat */
  async function openPaper(pmid) {
    const known = lastResults.find(d => d.pmid === pmid) || {};
    show(el.paperView, true);
    document.body.style.overflow = "hidden";

    el.paperTitle.textContent = known.title || "Loading…";
    el.paperMeta.innerHTML = "";
    el.paperAbs.textContent = "Loading abstract…";
    show(el.paperMeshWrap, false);
    show(el.paperKwWrap, false);
    el.paperLinks.innerHTML = "";
    el.chatThread.innerHTML = '<p class="chat-empty">Loading…</p>';
    el.chatSugg.innerHTML = "";
    el.chatInput.value = "";

    try {
      const article = await api("/api/article/" + encodeURIComponent(pmid));
      currentPaper = article;
      renderPaper(article, known);
      await loadChat(pmid);
      el.chatInput.focus();
    } catch (err) {
      el.paperAbs.textContent = "Could not load this record: " + err.message;
      el.chatThread.innerHTML = "";
    }
  }

  function renderPaper(a, known) {
    const terms = (lastPayload && lastPayload.understanding &&
                   lastPayload.understanding.terms) || [];

    el.paperTitle.textContent = a.title || "(untitled)";
    el.paperMeta.innerHTML =
      "<span>" + escapeHtml(authorLine(a.authors)) + "</span>" +
      (a.journal ? '<span class="sep">|</span><span><em>' + escapeHtml(a.journal) + "</em></span>" : "") +
      (a.pub_date ? '<span class="sep">|</span><span>' + escapeHtml(a.pub_date) + "</span>" : "") +
      '<span class="sep">|</span><span>PMID ' + escapeHtml(a.pmid) + "</span>" +
      (known && known.relevance != null
        ? '<span class="sep">|</span><span>match ' + known.relevance + "</span>" : "");

    el.paperAbs.innerHTML = a.abstract
      ? highlight(a.abstract, terms)
      : "<em>No abstract is available for this record in PubMed. Only its title and " +
        "MeSH headings can be searched or asked about.</em>";

    const mesh = a.mesh_terms || [];
    el.paperMesh.innerHTML = mesh
      .map(m => '<span class="tag tag-mesh">' + escapeHtml(m) + "</span>").join("");
    show(el.paperMeshWrap, mesh.length > 0);

    const kws = a.keywords || [];
    el.paperKw.innerHTML = kws
      .map(k => '<span class="tag">' + escapeHtml(k) + "</span>").join("");
    show(el.paperKwWrap, kws.length > 0);

    el.paperLinks.innerHTML =
      '<a class="btn-ghost" href="' + escapeHtml(a.url) +
        '" target="_blank" rel="noopener">Open in PubMed ↗</a>' +
      (a.doi ? '<a class="btn-ghost" href="https://doi.org/' + escapeHtml(a.doi) +
               '" target="_blank" rel="noopener">DOI ↗</a>' : "") +
      '<button class="btn-ghost" data-similar-inline="' + escapeHtml(a.pmid) +
        '">Similar papers</button>';

    const simBtn = $("[data-similar-inline]", el.paperLinks);
    if (simBtn) simBtn.addEventListener("click", () => openSimilar(a.pmid));
  }

  async function loadChat(pmid) {
    try {
      const data = await api("/api/chat/" + encodeURIComponent(pmid));
      el.chatThread.innerHTML = "";

      (data.messages || []).forEach(m => {
        if (m.role === "user") addBubble("user", m.text);
        else addBubble("bot", m.text, { evidence: m.evidence, confidence: m.confidence });
      });

      if (!(data.messages || []).length) {
        el.chatThread.innerHTML =
          '<p class="chat-empty">Ask anything about this paper.<br>' +
          "Answers are pulled verbatim from its abstract, with the exact sentences shown as evidence." +
          "</p>";
      }

      el.chatSugg.innerHTML = (data.suggestions || [])
        .map(q => '<button class="chip" type="button" data-suggest>' + escapeHtml(q) + "</button>")
        .join("");
      $$("[data-suggest]", el.chatSugg).forEach(b =>
        b.addEventListener("click", () => sendChat(b.textContent.trim())));

      scrollChat();
    } catch (err) {
      el.chatThread.innerHTML =
        '<p class="chat-empty">Chat unavailable: ' + escapeHtml(err.message) + "</p>";
    }
  }

  function addBubble(role, text, extra) {
    const empty = $(".chat-empty", el.chatThread);
    if (empty) empty.remove();

    const node = document.createElement("div");
    const low  = extra && extra.confidence != null && extra.confidence < 0.3;
    node.className = "bubble " + role + (low ? " low" : "");

    if (role === "user") {
      node.textContent = text;
    } else {
      let html = "<div>" + escapeHtml(text) + "</div>";
      const ev = (extra && extra.evidence) || [];
      const conf = extra && extra.confidence != null ? extra.confidence : null;

      if (conf != null || ev.length) {
        html += '<div class="bubble-foot">';
        if (conf != null)
          html += '<span class="conf' + (low ? " low" : "") + '">confidence ' +
                  Math.round(conf * 100) + "%</span>";
        if (ev.length)
          html += '<button class="ev-btn" type="button">Show evidence (' + ev.length + ")</button>";
        html += "</div>";
        if (ev.length) {
          html += '<div class="evidence" hidden>' + ev.map(e =>
            '<div class="evidence-item">' +
              (e.section ? '<span class="sec">' + escapeHtml(e.section) + "</span>" : "") +
              escapeHtml(e.text) +
            "</div>").join("") + "</div>";
        }
      }
      node.innerHTML = html;

      const btn = $(".ev-btn", node), box = $(".evidence", node);
      if (btn && box) {
        btn.addEventListener("click", () => {
          box.hidden = !box.hidden;
          btn.textContent = (box.hidden ? "Show" : "Hide") + " evidence (" + ev.length + ")";
          scrollChat();
        });
      }
    }

    el.chatThread.appendChild(node);
    return node;
  }

  function scrollChat() {
    el.chatThread.scrollTop = el.chatThread.scrollHeight;
  }

  async function sendChat(question) {
    if (chatBusy || !currentPaper) return;
    question = (question || el.chatInput.value || "").trim();
    if (!question) return;

    chatBusy = true;
    el.chatInput.value = "";
    el.chatSend.disabled = true;
    addBubble("user", question);

    const thinking = addBubble("bot", "Reading the abstract…");
    thinking.classList.add("thinking");
    scrollChat();

    try {
      const data = await api("/api/chat/" + encodeURIComponent(currentPaper.pmid), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })
      });
      thinking.remove();
      addBubble("bot", data.answer, {
        evidence: data.evidence, confidence: data.confidence
      });
    } catch (err) {
      thinking.remove();
      addBubble("bot", "Sorry — " + err.message);
    } finally {
      chatBusy = false;
      el.chatSend.disabled = false;
      scrollChat();
      el.chatInput.focus();
    }
  }

  el.chatForm.addEventListener("submit", e => { e.preventDefault(); sendChat(); });

  el.chatClear.addEventListener("click", async () => {
    if (!currentPaper) return;
    try {
      await api("/api/chat/" + encodeURIComponent(currentPaper.pmid), { method: "DELETE" });
      await loadChat(currentPaper.pmid);
      toast("Conversation cleared");
    } catch (err) { toast(err.message, true); }
  });

  /* -------------------------------------------------------------- modal */
  function openModal(html) {
    el.modalBody.innerHTML = html;
    show(el.modal, true);
    document.body.style.overflow = "hidden";
  }

  function closeAllOverlays() {
    [el.modal, el.paperView, el.historyDraw, el.exportDlg].forEach(n => show(n, false));
    document.body.style.overflow = "";
    currentPaper = null;
  }

  $$("[data-close]", el.modal).forEach(n => n.addEventListener("click", () => {
    show(el.modal, false);
    document.body.style.overflow = el.paperView.hidden ? "" : "hidden";
  }));
  $$("[data-close-paper]", el.paperView).forEach(n =>
    n.addEventListener("click", closeAllOverlays));
  $$("[data-close-history]", el.historyDraw).forEach(n =>
    n.addEventListener("click", () => { show(el.historyDraw, false); document.body.style.overflow = ""; }));
  $$("[data-close-export]", el.exportDlg).forEach(n =>
    n.addEventListener("click", () => { show(el.exportDlg, false); }));

  document.addEventListener("keydown", e => {
    if (e.key === "Escape") {
      if (!el.exportDlg.hidden) { show(el.exportDlg, false); return; }
      if (!el.modal.hidden) {
        show(el.modal, false);
        document.body.style.overflow = el.paperView.hidden ? "" : "hidden";
        return;
      }
      closeAllOverlays();
      return;
    }
    // "h" toggles history when not typing
    if ((e.key === "h" || e.key === "H") && !/^(INPUT|TEXTAREA|SELECT)$/.test(
          (document.activeElement || {}).tagName || "")) {
      el.historyBtn.click();
    }
  });

  function openSimilar(pmid) {
    openModal("<h3>Finding semantically similar papers…</h3><div class='skeleton'></div>");
    fetch("/api/similar/" + encodeURIComponent(pmid))
      .then(r => r.json())
      .then(d => {
        if (d.error) { openModal("<h3>Unavailable</h3><p>" + escapeHtml(d.error) + "</p>"); return; }
        const list = d.results || [];
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
    show(el.toolbar, false);
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

  $$(".examples .chip").forEach(chip =>
    chip.addEventListener("click", () => {
      el.query.value = chip.textContent.trim();
      runSearch(el.query.value);
    })
  );

  /* ------------------------------------------------------------ account */
  if (el.accountBtn) {
    el.accountBtn.addEventListener("click", async e => {
      e.stopPropagation();
      const open = el.accountMenu.hidden;
      show(el.accountMenu, open);
      el.accountBtn.setAttribute("aria-expanded", String(open));
      if (!open) return;

      try {
        const data = await api("/api/auth/me");
        const s = data.stats || {};
        el.accountStats.innerHTML =
          '<div class="astat"><b>' + (s.searches || 0) + "</b><span>searches</span></div>" +
          '<div class="astat"><b>' + (s.papers_discussed || 0) + "</b><span>papers asked</span></div>";
      } catch (_) {
        el.accountStats.innerHTML = "";
      }
    });

    document.addEventListener("click", e => {
      if (!el.accountMenu.hidden && !e.target.closest(".account")) {
        show(el.accountMenu, false);
        el.accountBtn.setAttribute("aria-expanded", "false");
      }
    });
  }

  if (el.accountPassword) {
    el.accountPassword.addEventListener("click", () => {
      show(el.accountMenu, false);
      openModal(
        "<h3>Change password</h3>" +
        "<p class='note'>All your other sessions are signed out when the password changes, " +
        "including this one — you will be asked to sign in again.</p>" +
        "<div class='auth-form' style='display:grid;gap:14px;margin-top:16px'>" +
          "<label class='field'><span>Current password</span>" +
            "<input id='cp-current' type='password' autocomplete='current-password' /></label>" +
          "<label class='field'><span>New password</span>" +
            "<input id='cp-new' type='password' autocomplete='new-password' /></label>" +
          "<label class='field'><span>Confirm new password</span>" +
            "<input id='cp-confirm' type='password' autocomplete='new-password' /></label>" +
          "<p class='note' id='cp-msg' style='color:var(--danger)'></p>" +
          "<div class='dialog-actions'>" +
            "<button class='btn-ghost' id='cp-cancel'>Cancel</button>" +
            "<button class='btn-primary' id='cp-go'>Change password</button>" +
          "</div>" +
        "</div>"
      );

      const msg = $("#cp-msg");
      $("#cp-cancel").addEventListener("click", () => {
        show(el.modal, false);
        document.body.style.overflow = "";
      });

      $("#cp-go").addEventListener("click", async () => {
        const current = $("#cp-current").value;
        const next    = $("#cp-new").value;
        if (next !== $("#cp-confirm").value) {
          msg.textContent = "The new passwords do not match.";
          return;
        }
        const btn = $("#cp-go");
        btn.disabled = true;
        btn.textContent = "Changing…";
        try {
          const res = await fetch("/api/auth/password", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ current_password: current, new_password: next })
          });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
          openModal("<h3>Password changed</h3><p>Signing you out…</p>");
          setTimeout(() => { window.location.href = "/login"; }, 1200);
        } catch (err) {
          msg.textContent = err.message;
          btn.disabled = false;
          btn.textContent = "Change password";
        }
      });

      $("#cp-current").focus();
    });
  }

  /* -------------------------------------------------------------- theme */
  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    window.__theme = t;
    el.themeBtn.textContent = t === "dark" ? "☀" : "◐";
  }
  setTheme(window.__theme ||
    (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  el.themeBtn.addEventListener("click", () =>
    setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));

  /* --------------------------------------------------------------- boot */
  loadHealth();
  updateExportHint();
  const initial = new URLSearchParams(window.location.search).get("q");
  if (initial) { el.query.value = initial; runSearch(initial); }
  el.query.focus();
})();

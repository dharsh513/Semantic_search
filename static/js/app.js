/* =============================================================================
   PubMed AI Research Assistant — Frontend Controller
   ========================================================================== */
(function () {
  "use strict";

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const el = {
    // Navigation
    navTabs:       $$(".nav-tab"),
    tabViews:      $$(".tab-view"),
    tabBadge:      $("#compare-tab-badge"),

    // Search & Controls
    form:          $("#search-form"),
    query:         $("#query"),
    field:         $("#field"),
    pageSize:      $("#page_size") || $("#top_k"),
    sort:          $("#sort"),
    minDate:       $("#mindate"),
    withAnswer:    $("#with_answer"),
    searchBtn:     $("#search-btn"),
    status:        $("#status"),
    empty:         $("#empty"),
    pipeline:      $("#pipeline"),
    pipeToggle:    $("#pipeline-toggle"),
    pipeBody:      $("#pipeline-body"),
    answerCard:    $("#answer-card"),
    answerText:    $("#answer-text"),
    answerMode:    $("#answer-mode"),
    citations:     $("#answer-citations"),

    // Results & Toolbar
    resultsSec:    $("#results-section"),
    results:       $("#results"),
    resultsMeta:   $("#results-meta"),
    toolbar:       $("#results-toolbar"),
    selectAll:     $("#select-all"),
    selectCount:   $("#select-count"),
    exportBtn:     $("#export-btn"),
    jumpLandscape: $("#jump-landscape-btn"),
    jumpCompare:   $("#jump-compare-btn"),
    jumpGaps:      $("#jump-gaps-btn"),

    // Pagination
    paginationBar: $("#pagination-bar"),
    paginationInfo: $("#pagination-info"),
    pageFirst:     $("#page-first"),
    pagePrev:      $("#page-prev"),
    pageCurrentLabel: $("#page-current-label"),
    pageNext:      $("#page-next"),
    pageLast:      $("#page-last"),

    // Landscape
    landscapeContent: $("#landscape-content"),
    landscapeScope:   $("#landscape-scope"),
    landscapeRefresh: $("#landscape-refresh-btn"),
    metricTotalMatches: $("#metric-total-matches"),
    metricTotalSample:  $("#metric-total-sample"),
    metricYearSpan:     $("#metric-year-span"),
    metricTopCluster:   $("#metric-top-cluster"),
    metricClusterCount: $("#metric-cluster-count"),
    metricTopMethod:    $("#metric-top-method"),
    metricMethodPct:    $("#metric-method-pct"),
    landscapeSummaryText: $("#landscape-summary-text"),
    timelineChart:      $("#timeline-chart"),
    methodologyList:    $("#methodology-list"),
    clusterGrid:        $("#cluster-grid"),
    landscapeMeshCloud: $("#landscape-mesh-cloud"),
    landscapeJournalsList: $("#landscape-journals-list"),
    landscapeAuthorsList:  $("#landscape-authors-list"),

    // Compare
    compareSelectionBar: $("#compare-selection-bar"),
    compareSelectedChips: $("#compare-selected-chips"),
    runCompareBtn:       $("#run-compare-btn"),
    compareOutput:       $("#compare-output"),
    compareSummaryBody:  $("#compare-summary-body"),
    compareTableHead:    $("#compare-table-head"),
    compareTableBody:    $("#compare-table-body"),
    compareEmptyState:   $("#compare-empty-state"),

    // Research Gaps
    runGapsBtn:          $("#run-gaps-btn"),
    gapsOutput:          $("#gaps-output"),
    gapsEmptyState:      $("#gaps-empty-state"),
    gapsEmptyAnalyzeBtn: $("#gaps-empty-analyze-btn"),
    gapsCountBadge:      $("#gaps-count-badge"),
    gapsCardsGrid:       $("#gaps-cards-grid"),
    directionsCardsGrid: $("#directions-cards-grid"),

    // Floating Drawer
    floatingDrawer:      $("#floating-drawer"),
    floatingCount:       $("#floating-count"),
    floatingCompareBtn:  $("#floating-compare-btn"),
    floatingGapsBtn:     $("#floating-gaps-btn"),
    floatingExportBtn:   $("#floating-export-btn"),
    floatingClearBtn:    $("#floating-clear-btn"),

    // Header & Settings
    enginePill:    $("#engine-pill"),
    cachePill:     $("#cache-pill"),
    themeBtn:      $("#theme-btn"),
    modal:         $("#modal"),
    modalBody:     $("#modal-content"),
    toast:         $("#toast"),

    // Paper Detail Modal & Chat
    paperView:     $("#paper-view"),
    paperTitle:    $("#paper-title"),
    paperMeta:     $("#paper-meta"),
    paperAbs:      $("#paper-abstract"),
    paperMeshWrap: $("#paper-mesh-wrap"),
    paperMesh:     $("#paper-mesh"),
    paperKwWrap:   $("#paper-keywords-wrap"),
    paperKw:       $("#paper-keywords"),
    paperLinks:    $("#paper-links"),
    chatThread:    $("#chat-thread"),
    chatSugg:      $("#chat-suggestions"),
    chatForm:      $("#chat-form"),
    chatInput:     $("#chat-input"),
    chatSend:      $("#chat-send"),
    chatClear:     $("#chat-clear"),

    // History Drawer
    historyBtn:    $("#history-btn"),
    historyDraw:   $("#history-drawer"),
    historyList:   $("#history-list"),
    historySearch: $("#history-search"),
    historyClear:  $("#history-clear"),

    // Export Dialog
    exportDlg:     $("#export-dialog"),
    exportCount:   $("#export-count"),
    exportPrev:    $("#export-preview"),
    exportHint:    $("#export-hint"),
    exportGo:      $("#export-go"),

    // Account
    accountBtn:    $("#account-btn"),
    accountMenu:   $("#account-menu"),
    accountStats:  $("#account-stats"),
    accountPassword: $("#account-password")
  };

  /* Application State */
  let activeTab    = "search";
  let lastResults  = [];
  let lastPayload  = null;       // Full /api/search response
  let currentPage  = 1;
  let totalPages   = 1;
  let selected     = new Set();  // PMIDs selected for comparison/export (max 3)
  let currentPaper = null;       // Article open in detail view
  let inFlight     = null;
  let chatBusy     = false;

  const MAX_SELECTION = 3;
  const SECTION_LABEL_RE = new RegExp(
    "^(background|objectives?|methods?|materials? and methods?|results?|findings?|" +
    "conclusions?|introduction|aims?|purpose|discussion|design|significance|importance)" +
    "\\s*:\\s*", "i"
  );

  /* ------------------------------------------------------------ Helper Utilities */
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
    if (res.status === 401 && data.auth_required) {
      window.location.href = "/login?next=" +
        encodeURIComponent(window.location.pathname + window.location.search);
      throw new Error("Session expired — redirecting to sign in.");
    }
    if (!res.ok || data.error) throw new Error(data.error || ("HTTP " + res.status));
    return data;
  }

  /* ------------------------------------------------------------ Tab Navigation */
  function switchTab(tabName) {
    activeTab = tabName;
    el.navTabs.forEach(btn => {
      const isActive = btn.dataset.tab === tabName;
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-selected", String(isActive));
    });
    el.tabViews.forEach(view => {
      const isTarget = view.id === "view-" + tabName;
      view.classList.toggle("active", isTarget);
      show(view, isTarget);
    });

    if (tabName === "landscape") {
      if (lastPayload && lastPayload.landscape) {
        renderLandscape(lastPayload.landscape);
      } else if (lastPayload && lastPayload.query) {
        fetchLandscape(lastPayload.query);
      }
    } else if (tabName === "compare") {
      syncCompareTab();
    } else if (tabName === "gaps") {
      syncGapsTab();
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  el.navTabs.forEach(tabBtn => {
    tabBtn.addEventListener("click", () => switchTab(tabBtn.dataset.tab));
  });

  if (el.jumpLandscape) el.jumpLandscape.addEventListener("click", () => switchTab("landscape"));
  if (el.jumpCompare) el.jumpCompare.addEventListener("click", () => switchTab("compare"));
  if (el.jumpGaps) el.jumpGaps.addEventListener("click", () => switchTab("gaps"));

  /* ------------------------------------------------------------ Health Check */
  let healthRetry = null;
  async function loadHealth(attempt) {
    attempt = attempt || 0;
    clearTimeout(healthRetry);
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (res.status === 401) return;
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      const emb = data.embedding || {};
      el.enginePill.textContent = "engine: " + (emb.kind || "?") + " · " + (emb.dim || "?") + "d";
      el.enginePill.title = "Embedding model: " + (emb.model || "unknown");
      el.enginePill.classList.remove("pill-muted");
      el.cachePill.textContent = "cache: " + ((data.store || {}).articles || 0) + " papers";
    } catch (_) {
      if (attempt < 4) {
        el.enginePill.textContent = "engine: starting…";
        el.enginePill.classList.add("pill-muted");
        healthRetry = setTimeout(() => loadHealth(attempt + 1), 1200 * (attempt + 1));
        return;
      }
      el.enginePill.textContent = "engine: offline";
      el.enginePill.classList.add("pill-muted");
      healthRetry = setTimeout(() => loadHealth(0), 15000);
    }
  }

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) loadHealth(0);
  });

  /* ------------------------------------------------------------ Pipeline Trace */
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
        "<strong>" + (s.total_matches || 0).toLocaleString() + "</strong> matching records in NCBI");
    row("Candidate pool",
        "Retrieved and embedded <strong>" + (s.candidates_fetched || 0) + "</strong> candidate papers · <strong>" +
        (s.chunks_indexed || 0) + "</strong> passages");

    const emb = s.embedding || {};
    row("Embedding backend", "<code>" + escapeHtml(emb.model || "?") + "</code> (" +
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

  if (el.pipeToggle) {
    el.pipeToggle.addEventListener("click", () => {
      const open = el.pipeToggle.getAttribute("aria-expanded") === "true";
      el.pipeToggle.setAttribute("aria-expanded", String(!open));
      show(el.pipeBody, !open);
    });
  }

  /* ------------------------------------------------------------ Grounded RAG Answer */
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

  /* ------------------------------------------------------------ Search Results */
  function renderResults(data) {
    lastResults = data.results || [];
    lastPayload = data;
    const terms = (data.understanding && data.understanding.terms) || [];
    const p = data.pagination || {};
    currentPage = p.page || 1;
    totalPages  = p.total_pages || 1;

    if (!lastResults.length) {
      el.results.innerHTML =
        '<div class="card"><p style="margin:0;color:var(--text-2)">' +
        "No matching records found. Try adjusting keywords or switching to Semantic (Auto) mode." +
        "</p></div>";
      el.resultsMeta.textContent = "0 results";
      show(el.toolbar, false);
      show(el.paginationBar, false);
      show(el.resultsSec, true);
      return;
    }

    const totalMatches = (p.total_matches || (data.stages && data.stages.total_matches) || lastResults.length).toLocaleString();
    const startIndex   = p.start_index || 1;
    const endIndex     = p.end_index || lastResults.length;

    el.resultsMeta.textContent =
      totalMatches + " papers found · Showing " + startIndex + "–" + endIndex +
      " · " + (data.took_ms || 0) + " ms" + (data.from_history ? " · from history" : "");

    // Render Cards
    el.results.innerHTML = lastResults.map((d, i) => {
      const pmid = String(d.pmid);
      const isChecked = selected.has(pmid);
      const passages = d.matched_passages || [];
      const top = passages.find(p => (p.section || "").toLowerCase() !== "title") || passages[0] || null;

      let snippet = "";
      if (top) {
        let body = cleanPassage(top.text) || String(top.text || "");
        if (body.length > 400) body = body.slice(0, 400).trim() + "…";
        const label = (top.section || "").toLowerCase() === "title" ? "" : top.section;
        snippet = '<div class="snippet">' +
          (label ? '<span class="sec">' + escapeHtml(label) + "</span>" : "") +
          highlight(body, terms) + "</div>";
      }

      const mesh = (d.mesh_terms || []).slice(0, 5)
        .map(m => '<span class="tag tag-mesh">' + escapeHtml(m) + "</span>").join("");

      const why = d.why_this_paper || {};
      const semPct = why.semantic_pct != null ? why.semantic_pct : Math.round((d.semantic_score || 0.8) * 100);
      const kwPct  = why.keyword_pct != null ? why.keyword_pct : Math.round((d.bm25_score || 0.7) * 100);
      const ovPct  = why.overall_pct != null ? why.overall_pct : Math.round((d.score || 0.85) * 100);
      const reasons = why.reasons || ["Direct semantic match with query intent"];

      return (
        '<article class="card fade-in' + (isChecked ? " selected" : "") + '" data-pmid="' + escapeHtml(pmid) +
             '" style="animation-delay:' + (i * 25) + 'ms">' +
          '<div class="card-top">' +
            '<input type="checkbox" class="card-check" ' + (isChecked ? "checked " : "") +
              'aria-label="Select paper" data-select="' + escapeHtml(pmid) + '" />' +
            '<span class="rank">' + (d.rank || i + 1) + "</span>" +
            '<div class="card-main">' +
              '<h3 class="card-title"><a href="#" data-open="' + escapeHtml(pmid) + '">' +
                escapeHtml(d.title || "(untitled)") + "</a></h3>" +
              '<div class="meta">' +
                "<span>" + escapeHtml(authorLine(d.authors)) + "</span>" +
                (d.journal ? '<span class="sep">|</span><span><em>' + escapeHtml(d.journal) + "</em></span>" : "") +
                (d.pub_date || d.year ? '<span class="sep">|</span><span>' + escapeHtml(d.pub_date || d.year) + "</span>" : "") +
                '<span class="sep">|</span><span>PMID ' + escapeHtml(pmid) + "</span>" +
              "</div>" +
            "</div>" +
            '<div class="relevance">' +
              '<span class="relevance-num">' + (d.relevance != null ? d.relevance : ovPct) + "%</span>" +
              '<span class="relevance-lbl">match</span>' +
              '<span class="bar"><i style="width:' + Math.max(6, Math.min(100, d.relevance || ovPct)) + '%"></i></span>' +
            "</div>" +
          "</div>" +
          snippet +
          (mesh ? '<div class="mesh-list">' + mesh + "</div>" : "") +

          /* Explainability Accordion */
          '<div class="why-panel" id="why-' + escapeHtml(pmid) + '" hidden>' +
            '<div class="why-head">' +
              '<span>🎯 Why this paper?</span>' +
              '<div class="why-scores">' +
                '<span class="why-score-item">Semantic: <b>' + semPct + '%</b></span>' +
                '<span class="why-score-item">Keyword: <b>' + kwPct + '%</b></span>' +
                '<span class="why-score-item">Overall: <b>' + ovPct + '%</b></span>' +
              '</div>' +
            '</div>' +
            '<ul class="why-reasons">' +
              reasons.map(r => '<li>✓ ' + escapeHtml(r) + '</li>').join("") +
            '</ul>' +
          '</div>' +

          '<div class="card-foot">' +
            '<button class="why-btn" data-why="' + escapeHtml(pmid) + '">Why this paper?</button>' +
            '<button class="btn-ghost" data-open="' + escapeHtml(pmid) + '">Read &amp; Ask</button>' +
            '<button class="btn-ghost" data-similar="' + escapeHtml(pmid) + '">Similar</button>' +
            (d.doi ? '<a class="btn-ghost" href="https://doi.org/' + escapeHtml(d.doi) +
                     '" target="_blank" rel="noopener">DOI ↗</a>' : "") +
            '<span class="spacer"></span>' +
            '<a class="btn-ghost" href="' + escapeHtml(d.url) +
              '" target="_blank" rel="noopener">Open in PubMed ↗</a>' +
          "</div>" +
        "</article>"
      );
    }).join("");

    // Wire Card Events
    $$("[data-open]", el.results).forEach(b =>
      b.addEventListener("click", e => { e.preventDefault(); openPaper(b.dataset.open); }));
    $$("[data-similar]", el.results).forEach(b =>
      b.addEventListener("click", () => openSimilar(b.dataset.similar)));
    $$("[data-select]", el.results).forEach(cb =>
      cb.addEventListener("change", () => toggleSelect(cb.dataset.select, cb.checked)));
    $$("[data-why]", el.results).forEach(b => {
      b.addEventListener("click", () => {
        const panel = $("#why-" + b.dataset.why);
        if (panel) {
          const isClosed = panel.hidden;
          panel.hidden = !isClosed;
          b.textContent = isClosed ? "Hide explanation" : "Why this paper?";
        }
      });
    });

    renderPaginationControls(p);
    show(el.toolbar, true);
    syncSelection();
    show(el.resultsSec, true);
  }

  /* ------------------------------------------------------------ Pagination Controls */
  function renderPaginationControls(p) {
    if (!p || p.total_pages <= 1) {
      show(el.paginationBar, false);
      return;
    }
    el.paginationInfo.textContent =
      "Showing " + (p.start_index || 1) + "–" + (p.end_index || lastResults.length) +
      " of " + (p.total_matches || p.total_results || 0).toLocaleString() + " papers found";
    el.pageCurrentLabel.textContent = "Page " + currentPage + " of " + totalPages;

    el.pageFirst.disabled = currentPage <= 1;
    el.pagePrev.disabled  = currentPage <= 1;
    el.pageNext.disabled  = currentPage >= totalPages;
    el.pageLast.disabled  = currentPage >= totalPages;
    show(el.paginationBar, true);
  }

  if (el.pageFirst) el.pageFirst.addEventListener("click", () => goToPage(1));
  if (el.pagePrev)  el.pagePrev.addEventListener("click", () => goToPage(currentPage - 1));
  if (el.pageNext)  el.pageNext.addEventListener("click", () => goToPage(currentPage + 1));
  if (el.pageLast)  el.pageLast.addEventListener("click", () => goToPage(totalPages));

  function goToPage(page) {
    if (page < 1 || page > totalPages || page === currentPage) return;
    const query = el.query.value.trim() || (lastPayload && lastPayload.query);
    if (query) runSearch(query, page);
  }

  /* ------------------------------------------------------------ Paper Selection */
  function toggleSelect(pmid, on) {
    if (on) {
      if (selected.size >= MAX_SELECTION) {
        toast("Maximum " + MAX_SELECTION + " papers can be selected for comparison.", true);
        const cb = el.results.querySelector('[data-select="' + CSS.escape(pmid) + '"]');
        if (cb) cb.checked = false;
        return;
      }
      selected.add(pmid);
    } else {
      selected.delete(pmid);
    }

    const card = el.results.querySelector('[data-pmid="' + CSS.escape(pmid) + '"]');
    if (card) card.classList.toggle("selected", on);
    syncSelection();
  }

  function syncSelection() {
    const n = selected.size;
    const totalVisible = lastResults.length;

    // Update check states in results
    $$("[data-select]", el.results).forEach(cb => {
      cb.checked = selected.has(cb.dataset.select);
    });
    $$(".card", el.results).forEach(c => {
      c.classList.toggle("selected", selected.has(c.dataset.pmid));
    });

    // Update Toolbar
    el.selectCount.textContent = n === 0 ? "Select (up to 3)" : n + " selected (max " + MAX_SELECTION + ")";
    el.selectAll.checked = n > 0 && n === Math.min(MAX_SELECTION, totalVisible);
    el.selectAll.indeterminate = n > 0 && n < Math.min(MAX_SELECTION, totalVisible);
    el.exportBtn.disabled = n === 0;

    // Update Navigation Badges & Buttons
    $$(".sel-count-num").forEach(span => { span.textContent = n; });
    if (el.jumpCompare) el.jumpCompare.disabled = n < 2;
    if (el.tabBadge) {
      el.tabBadge.textContent = n;
      show(el.tabBadge, n > 0);
    }

    // Update Floating Bottom Drawer
    if (n > 0) {
      el.floatingCount.textContent = n + " paper" + (n === 1 ? "" : "s") + " selected";
      el.floatingCompareBtn.disabled = n < 2;
      show(el.floatingDrawer, true);
    } else {
      show(el.floatingDrawer, false);
    }

    syncCompareTab();
  }

  if (el.selectAll) {
    el.selectAll.addEventListener("change", () => {
      const on = el.selectAll.checked;
      if (on) {
        selected = new Set(lastResults.slice(0, MAX_SELECTION).map(d => String(d.pmid)));
      } else {
        selected.clear();
      }
      syncSelection();
    });
  }

  if (el.floatingClearBtn) {
    el.floatingClearBtn.addEventListener("click", () => {
      selected.clear();
      syncSelection();
    });
  }
  if (el.floatingCompareBtn) el.floatingCompareBtn.addEventListener("click", () => {
    switchTab("compare");
    executeComparison();
  });
  if (el.floatingGapsBtn)    el.floatingGapsBtn.addEventListener("click", () => switchTab("gaps"));
  if (el.floatingExportBtn)  el.floatingExportBtn.addEventListener("click", openExport);

  /* ------------------------------------------------------------ Research Landscape */
  async function fetchLandscape(query) {
    setStatus("loading", "Generating research landscape and topic clusters…");
    try {
      const data = await api("/api/landscape", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
      });
      setStatus(null);
      renderLandscape(data);
    } catch (err) {
      setStatus("error", "Landscape analysis error: " + err.message);
    }
  }

  function renderLandscape(land) {
    if (!land) return;
    const totalMatches = (land.total_matches || land.total_papers || 0).toLocaleString();
    el.landscapeScope.textContent =
      "Macro-level analytics across " + land.total_papers + " candidate studies (from " + totalMatches + " PubMed records).";

    // Metrics Row
    el.metricTotalMatches.textContent = totalMatches;
    el.metricTotalSample.textContent  = land.total_papers + " candidate papers analyzed";
    el.metricYearSpan.textContent     = (land.year_range && land.year_range.min && land.year_range.max)
      ? land.year_range.min + " – " + land.year_range.max : "Recent years";

    const topCluster = (land.topic_clusters && land.topic_clusters[0]) || null;
    el.metricTopCluster.textContent   = topCluster ? topCluster.title : "General Biomedical";
    el.metricClusterCount.textContent = topCluster ? topCluster.paper_count + " papers (" + topCluster.percentage + "%)" : "—";

    const topMethod = (land.methodologies && land.methodologies[0]) || null;
    el.metricTopMethod.textContent    = topMethod ? topMethod.method : "Experimental";
    el.metricMethodPct.textContent   = topMethod ? topMethod.percentage + "% of literature" : "—";

    el.landscapeSummaryText.textContent = land.summary || "No landscape summary available.";

    // 1. Timeline Chart
    const trend = land.publication_trend || [];
    if (trend.length) {
      const maxCount = Math.max(...trend.map(t => t.count), 1);
      el.timelineChart.innerHTML = trend.map(t => {
        const heightPct = Math.max(8, Math.round((t.count / maxCount) * 100));
        return (
          '<div class="timeline-col">' +
            '<span class="timeline-tooltip">' + t.year + ': ' + t.count + ' papers</span>' +
            '<div class="timeline-bar" style="height:' + heightPct + '%"></div>' +
            '<span class="timeline-lbl">' + t.year.slice(-2) + '</span>' +
          '</div>'
        );
      }).join("");
    } else {
      el.timelineChart.innerHTML = '<p class="chart-empty">No trend data available.</p>';
    }

    // 2. Methodology List
    const methods = land.methodologies || [];
    if (methods.length) {
      el.methodologyList.innerHTML = methods.map(m => (
        '<div class="method-row">' +
          '<div class="method-meta">' +
            '<span>' + escapeHtml(m.method) + '</span>' +
            '<span>' + m.count + ' (' + m.percentage + '%)</span>' +
          '</div>' +
          '<div class="method-bar-bg">' +
            '<div class="method-bar-fill" style="width:' + Math.max(4, m.percentage) + '%"></div>' +
          '</div>' +
        '</div>'
      )).join("");
    } else {
      el.methodologyList.innerHTML = '<p class="chart-empty">No methodology data available.</p>';
    }

    // 3. Topic Clusters
    const clusters = land.topic_clusters || [];
    if (clusters.length) {
      el.clusterGrid.innerHTML = clusters.map(c => {
        const repList = (c.representative_papers || []).map(p =>
          '<div class="cluster-rep-link" data-open="' + escapeHtml(p.pmid) + '">📄 ' +
          escapeHtml(p.title || "PMID " + p.pmid) + ' <small>(' + escapeHtml(p.year || "N/A") + ')</small></div>'
        ).join("");

        return (
          '<div class="cluster-card">' +
            '<div class="card-top" style="margin:0;">' +
              '<div class="cluster-title">' + escapeHtml(c.title) + '</div>' +
              '<span class="cluster-count-badge">' + c.paper_count + ' papers (' + c.percentage + '%)</span>' +
            '</div>' +
            '<div class="method-bar-bg" style="height:5px;">' +
              '<div class="method-bar-fill" style="width:' + c.percentage + '%"></div>' +
            '</div>' +
            '<div class="cluster-rep-papers">' +
              '<strong style="font-size:11px;color:var(--text-3);text-transform:uppercase;">Representative Studies:</strong>' +
              repList +
            '</div>' +
          '</div>'
        );
      }).join("");

      $$(".cluster-rep-link", el.clusterGrid).forEach(b => {
        b.addEventListener("click", () => openPaper(b.dataset.open));
      });
    } else {
      el.clusterGrid.innerHTML = '<p class="chart-empty">No clusters available.</p>';
    }

    // 4. MeSH Cloud
    const meshTerms = land.top_mesh_terms || [];
    if (meshTerms.length) {
      el.landscapeMeshCloud.innerHTML = meshTerms.map(m =>
        '<span class="mesh-pill">' +
          escapeHtml(m.term) +
          '<span class="mesh-pill-cnt">' + m.count + '</span>' +
        '</span>'
      ).join("");
    } else {
      el.landscapeMeshCloud.innerHTML = '<p class="chart-empty">No MeSH descriptors available.</p>';
    }

    // 5. Journals & Authors
    const journals = land.top_journals || [];
    el.landscapeJournalsList.innerHTML = journals.length
      ? journals.map(j =>
          '<li class="ranked-item"><strong title="' + escapeHtml(j.journal) + '">' +
          escapeHtml(j.journal) + '</strong><span>' + j.count + '</span></li>'
        ).join("")
      : '<li class="chart-empty">None available</li>';

    const authors = land.top_authors || [];
    el.landscapeAuthorsList.innerHTML = authors.length
      ? authors.map(a =>
          '<li class="ranked-item"><strong>' + escapeHtml(a.author) + '</strong><span>' + a.count + '</span></li>'
        ).join("")
      : '<li class="chart-empty">None available</li>';
  }

  if (el.landscapeRefresh) {
    el.landscapeRefresh.addEventListener("click", () => {
      const q = el.query.value.trim() || (lastPayload && lastPayload.query);
      if (q) fetchLandscape(q);
    });
  }

  /* ------------------------------------------------------------ Compare Papers */
  function syncCompareTab() {
    const pmids = Array.from(selected);
    el.runCompareBtn.disabled = pmids.length < 2;

    if (pmids.length === 0) {
      el.compareSelectedChips.innerHTML = '<span class="empty-chip-hint">No papers selected. Select 2 or 3 papers from Search results using the checkboxes.</span>';
      show(el.compareEmptyState, true);
      show(el.compareOutput, false);
      return;
    }

    show(el.compareEmptyState, false);
    el.compareSelectedChips.innerHTML = pmids.map(p => {
      const doc = (lastResults || []).find(d => String(d.pmid) === p) || {};
      const title = doc.title ? (doc.title.slice(0, 32) + "…") : "PMID " + p;
      return (
        '<span class="paper-sel-chip">' +
          '<span>' + escapeHtml(title) + '</span>' +
          '<button class="chip-del-btn" data-unsel="' + escapeHtml(p) + '" title="Remove">✕</button>' +
        '</span>'
      );
    }).join("");

    $$("[data-unsel]", el.compareSelectedChips).forEach(b => {
      b.addEventListener("click", () => toggleSelect(b.dataset.unsel, false));
    });
  }

  async function executeComparison(pmids) {
    pmids = pmids || Array.from(selected);
    if (pmids.length < 2) {
      toast("Please select 2 or 3 papers to compare.", true);
      return;
    }

    el.runCompareBtn.disabled = true;
    el.runCompareBtn.textContent = "Comparing…";
    setStatus("loading", "Extracting structured dimensions and synthesizing comparison…");

    try {
      const data = await api("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pmids: pmids })
      });
      setStatus(null);
      renderComparison(data);
      show(el.compareEmptyState, false);
      show(el.compareOutput, true);
    } catch (err) {
      setStatus("error", "Comparison failed: " + err.message);
      toast(err.message, true);
    } finally {
      el.runCompareBtn.disabled = false;
      el.runCompareBtn.textContent = "Run Comparison";
    }
  }

  function renderComparison(data) {
    const papers = data.papers || [];
    const dims   = data.dimensions || [];

    // Comparative summary
    el.compareSummaryBody.textContent = data.summary || "No comparative summary generated.";

    // Table Header
    el.compareTableHead.innerHTML =
      '<tr>' +
        '<th class="compare-dim-th">Dimension</th>' +
        papers.map(p =>
          '<th class="compare-paper-col">' +
            '<strong><a href="#" data-open="' + escapeHtml(p.pmid) + '">' + escapeHtml(p.title || "PMID " + p.pmid) + '</a></strong><br>' +
            '<small style="color:var(--text-3);font-family:var(--mono);">' +
              escapeHtml(p.first_author) + ' et al. (' + escapeHtml(p.year || "N/A") + ') · PMID ' + escapeHtml(p.pmid) +
            '</small>' +
          '</th>'
        ).join("") +
      '</tr>';

    // Table Body Rows
    el.compareTableBody.innerHTML = dims.map(d =>
      '<tr>' +
        '<td class="compare-dim-th"><strong>' + escapeHtml(d.label) + '</strong></td>' +
        papers.map(p => {
          const val = d.values[p.pmid] || "Not specified in the available paper content.";
          const isNotSpec = val.includes("Not specified");
          return (
            '<td>' +
              (isNotSpec
                ? '<span style="color:var(--text-3);font-style:italic;">' + escapeHtml(val) + '</span>'
                : escapeHtml(val)) +
            '</td>'
          );
        }).join("") +
      '</tr>'
    ).join("");

    $$("[data-open]", el.compareTableHead).forEach(b => {
      b.addEventListener("click", e => { e.preventDefault(); openPaper(b.dataset.open); });
    });
  }

  if (el.runCompareBtn) el.runCompareBtn.addEventListener("click", () => executeComparison());

  /* ------------------------------------------------------------ Research Gaps */
  function syncGapsTab() {
    if (!lastResults.length && selected.size === 0) {
      show(el.gapsEmptyState, true);
      show(el.gapsOutput, false);
    }
  }

  async function executeGapAnalysis(pmids) {
    pmids = pmids || (selected.size > 0 ? Array.from(selected) : (lastResults || []).slice(0, 6).map(d => String(d.pmid)));
    if (!pmids.length) {
      toast("Run a search or select papers to identify research gaps.", true);
      return;
    }

    el.runGapsBtn.disabled = true;
    el.runGapsBtn.textContent = "Analyzing Gaps…";
    setStatus("loading", "Evaluating 14 research gap dimensions across literature…");

    try {
      const data = await api("/api/research-gaps", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pmids: pmids })
      });
      setStatus(null);
      renderResearchGaps(data);
      show(el.gapsEmptyState, false);
      show(el.gapsOutput, true);
    } catch (err) {
      setStatus("error", "Gap analysis failed: " + err.message);
      toast(err.message, true);
    } finally {
      el.runGapsBtn.disabled = false;
      el.runGapsBtn.textContent = "Analyze Literature Gaps";
    }
  }

  function renderResearchGaps(data) {
    const gaps = data.gaps || [];
    const dirs = data.research_directions || [];

    el.gapsCountBadge.textContent = gaps.length + " Potential Gap" + (gaps.length === 1 ? "" : "s") + " Detected";

    // 1. Gap Cards
    el.gapsCardsGrid.innerHTML = gaps.length ? gaps.map(g => {
      const confClass = "conf-" + (g.confidence || "medium").toLowerCase();
      const points = (g.supporting_points || []).map(pt => '<li>' + escapeHtml(pt) + '</li>').join("");

      return (
        '<div class="gap-card">' +
          '<div class="gap-card-head">' +
            '<span class="gap-cat-pill">' + escapeHtml(g.category || "Research Gap") + '</span>' +
            '<span class="conf-pill ' + confClass + '">Confidence: ' + escapeHtml(g.confidence || "Medium") + '</span>' +
          '</div>' +
          '<h4 class="gap-title">' + escapeHtml(g.title) + '</h4>' +
          '<p class="gap-desc">' + escapeHtml(g.description) + '</p>' +
          (points ? '<div class="gap-evidence-box"><div class="gap-evidence-title">Supporting Published Evidence:</div><ul class="gap-evidence-list">' + points + '</ul></div>' : "") +
        '</div>'
      );
    }).join("") : '<p class="chart-empty">No distinct research gaps detected in this paper sample.</p>';

    // 2. Suggested Directions
    el.directionsCardsGrid.innerHTML = dirs.length ? dirs.map(d => (
      '<div class="direction-card">' +
        '<span class="dir-linked">🔗 Linked to: ' + escapeHtml(d.linked_gap_title) + '</span>' +
        '<h4 class="dir-title">💡 ' + escapeHtml(d.title) + '</h4>' +
        '<p class="dir-desc">' + escapeHtml(d.description) + '</p>' +
        (d.suggested_methodology ? '<div class="dir-method-box"><strong>Suggested Methodology:</strong> ' + escapeHtml(d.suggested_methodology) + '</div>' : "") +
      '</div>'
    )).join("") : '<p class="chart-empty">No proposed research directions generated.</p>';
  }

  if (el.runGapsBtn)          el.runGapsBtn.addEventListener("click", () => executeGapAnalysis());
  if (el.gapsEmptyAnalyzeBtn) el.gapsEmptyAnalyzeBtn.addEventListener("click", () => executeGapAnalysis());

  /* ------------------------------------------------------------ Search Execution */
  async function runSearch(query, page) {
    page = page || 1;
    if (inFlight) inFlight.abort();
    inFlight = new AbortController();

    el.searchBtn.disabled = true;
    $(".btn-label", el.searchBtn).textContent = "Searching…";
    show(el.empty, false);
    show(el.answerCard, false);
    show(el.pipeline, false);
    show(el.toolbar, false);
    show(el.paginationBar, false);
    setStatus("loading", "Querying PubMed, embedding passages, and building semantic landscape…");

    el.results.innerHTML =
      '<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>';
    show(el.resultsSec, true);
    el.resultsMeta.textContent = "";

    const pageSizeVal = parseInt(el.pageSize.value, 10) || 20;
    const body = {
      query: query,
      page: page,
      page_size: pageSizeVal,
      top_k: pageSizeVal,
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

      if (data.landscape) {
        renderLandscape(data.landscape);
      }

      loadHealth();

      const url = new URL(window.location);
      url.searchParams.set("q", query);
      if (page > 1) url.searchParams.set("p", page); else url.searchParams.delete("p");
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
    if (q) {
      switchTab("search");
      runSearch(q, 1);
    }
  });

  $$(".examples .chip").forEach(chip =>
    chip.addEventListener("click", () => {
      el.query.value = chip.textContent.trim();
      switchTab("search");
      runSearch(el.query.value, 1);
    })
  );

  /* ------------------------------------------------------------ Paper Detail & Chat */
  async function openPaper(pmid) {
    const known = (lastResults || []).find(d => String(d.pmid) === String(pmid)) || {};
    show(el.paperView, true);
    document.body.style.overflow = "hidden";

    el.paperTitle.textContent = known.title || "Loading PubMed article…";
    el.paperMeta.innerHTML = "";
    el.paperAbs.textContent = "Loading structured abstract…";
    show(el.paperMeshWrap, false);
    show(el.paperKwWrap, false);
    el.paperLinks.innerHTML = "";
    el.chatThread.innerHTML = '<p class="chat-empty">Loading QA assistant…</p>';
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
        ? '<span class="sep">|</span><span>match ' + known.relevance + "%</span>" : "");

    el.paperAbs.innerHTML = a.abstract
      ? highlight(a.abstract, terms)
      : "<em>No abstract is available for this record in PubMed. Only title and MeSH terms are available.</em>";

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
        '">Similar Papers</button>';

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
          "Answers are pulled verbatim from its abstract, with verified sentences shown as evidence." +
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

    const thinking = addBubble("bot", "Reading abstract evidence…");
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

  /* ------------------------------------------------------------ History Drawer */
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
             : "No searches yet. Every search is automatically saved here.") +
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
        '<p class="chat-empty">Could not load history: " + escapeHtml(err.message) + "</p>';
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
      switchTab("search");
      renderPipeline(data);
      renderAnswer(data.answer);
      renderResults(data);
      if (data.snapshot && data.snapshot.landscape) {
        renderLandscape(data.snapshot.landscape);
      }
      toast("Restored “" + (data.query || "").slice(0, 48) + "” from history");
    } catch (err) {
      toast("Could not restore: " + err.message, true);
    }
  }

  if (el.historyBtn) {
    el.historyBtn.addEventListener("click", () => {
      show(el.historyDraw, true);
      document.body.style.overflow = "hidden";
      loadHistory();
    });
  }

  if (el.historyClear) {
    el.historyClear.addEventListener("click", async () => {
      try {
        const data = await api("/api/history?keep_pinned=true", { method: "DELETE" });
        toast("Cleared " + (data.deleted || 0) + " unpinned search(es).");
        loadHistory();
      } catch (err) { toast(err.message, true); }
    });
  }

  let histTimer = null;
  if (el.historySearch) {
    el.historySearch.addEventListener("input", () => {
      clearTimeout(histTimer);
      histTimer = setTimeout(loadHistory, 220);
    });
  }

  /* ------------------------------------------------------------ Citation Export */
  function selectedPmids() {
    return Array.from(selected);
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
        ? "Formatted PDF: search query, grounded answer with citations, and complete structured abstracts."
        : "Formatted PDF containing the numbered Vancouver reference list.",
      bib: "BibTeX (.bib) for LaTeX, Overleaf, and reference managers.",
      ris: "RIS (.ris) for EndNote, Zotero, Mendeley, and JabRef.",
      txt: "Numbered plain-text Vancouver citations ready to copy."
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
    el.exportPrev.textContent = "Generating citation preview…";
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
    el.exportGo.textContent = "Generating…";

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
      const name = res.headers.get("X-Export-Filename") || ("pubmed-export." + o.format);
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

  if (el.exportBtn) el.exportBtn.addEventListener("click", openExport);
  if (el.exportGo)  el.exportGo.addEventListener("click", runExport);
  $$('#export-format input, #export-mode input').forEach(i =>
    i.addEventListener("change", updateExportHint));

  /* ------------------------------------------------------------ Modals & Overlays */
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
    if ((e.key === "h" || e.key === "H") && !/^(INPUT|TEXTAREA|SELECT)$/.test(
          (document.activeElement || {}).tagName || "")) {
      if (el.historyBtn) el.historyBtn.click();
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
                " · match " + (r.relevance || 0) + "%</span></li>").join("") +
              "</ol>"
            : "<p>No close neighbours found.</p>")
        );
      })
      .catch(() => openModal("<h3>Network error</h3><p>Could not load similar papers.</p>"));
  }

  /* ------------------------------------------------------------ Account Menu */
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
        "<p class='note'>All other active sessions are revoked when the password changes.</p>" +
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

  /* ------------------------------------------------------------ Theme Toggle */
  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    window.__theme = t;
    el.themeBtn.textContent = t === "dark" ? "☀" : "◐";
  }
  setTheme(window.__theme ||
    (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  el.themeBtn.addEventListener("click", () =>
    setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));

  /* ------------------------------------------------------------ App Boot */
  loadHealth();
  updateExportHint();
  const urlParams = new URLSearchParams(window.location.search);
  const initialQ = urlParams.get("q");
  const initialP = parseInt(urlParams.get("p"), 10) || 1;
  if (initialQ) {
    el.query.value = initialQ;
    runSearch(initialQ, initialP);
  }
  el.query.focus();
})();


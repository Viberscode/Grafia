/* ThemeGraph bidirectional graph component (plain JS, vis-network, Streamlit component protocol). */
(function () {
  "use strict";

  // ---------- Streamlit protocol ----------
  const Streamlit = {
    send(type, data) {
      window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data || {}), "*");
    },
    ready() { this.send("streamlit:componentReady", { apiVersion: 1 }); },
    height(h) { this.send("streamlit:setFrameHeight", { height: h }); },
    value(v) { this.send("streamlit:setComponentValue", { value: v, dataType: "json" }); },
  };

  const $ = (id) => document.getElementById(id);
  const S = {
    payload: null, sig: null, network: null, nodes: null, edges: null,
    byId: {}, edgeById: {}, adj: {}, hl: new Set(), hlEdges: new Set(), pinned: new Set(),
    view: "answer", depth: 0, selected: null, hiddenComms: new Set(), labels: "zoom",
    physics: false, replay: null, focusNonce: null, flash: null, height: 680, theme: null,
    visible: new Set(), pulseTimer: null, physicsTimer: null, lastExport: null,
    showDocs: false, hlDocs: new Set(), halo: [], trace: [], stepOf: {}, replayTimer: null,
  };
  const isDoc = (n) => n && n.kind === "document";

  // ---------- helpers ----------
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function rgba(hex, a) {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function hash(str) {
    let h = 5381;
    for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) | 0;
    return String(h);
  }
  function emit(evt) {
    evt.eid = Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 7);
    Streamlit.value(evt);
  }
  let statusTimer = null;
  function status(msg) {
    const el = $("status");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(statusTimer);
    statusTimer = setTimeout(() => el.classList.remove("show"), 1800);
  }
  const commColor = (c) => {
    const found = (S.payload.communities || []).find((x) => x.id === c);
    return found ? found.color : "#8f99a8";
  };
  const commTitle = (c) => {
    const found = (S.payload.communities || []).find((x) => x.id === c);
    return found ? found.title : "Community " + c;
  };

  // ---------- styling ----------
  function colors() {
    return { fg: cssVar("--fg") || "#1f2430", bg: cssVar("--bg") || "#ffffff",
             edge: cssVar("--edge") || "#8f99a8", hl: cssVar("--hl") || "#ff2d2d" };
  }
  function tooltip(n) {
    const el = document.createElement("div");
    el.className = "tip";
    const desc = (n.description || "").slice(0, 120);
    el.innerHTML = `<b>${esc(n.label)}</b><small>${esc(n.type || "entity")} &middot; ${esc(commTitle(n.community))}</small>` +
      `<div>${esc(desc)}${(n.description || "").length > 120 ? "&hellip;" : ""}</div>`;
    return el;
  }
  function docStyle(n) {
    const c = colors();
    const isH = S.hlDocs.has(n.id);
    const dim = (S.hl.size > 0 || S.hlDocs.size > 0) && !isH;
    const el = document.createElement("div");
    el.className = "tip";
    el.innerHTML = `<b>${esc(n.label)}</b><small>document &middot; ${esc(n.degree)} entities</small>`;
    return {
      id: n.id, label: n.label, shape: "square", size: isH ? 13 : 10, value: undefined, title: el,
      color: { background: rgba("#9aa4b2", dim ? 0.3 : 1), border: isH ? c.hl : rgba("#6b7280", dim ? 0.3 : 1),
               highlight: { background: "#9aa4b2", border: c.fg }, hover: { background: "#9aa4b2", border: c.fg } },
      borderWidth: isH ? 3 : 1, borderWidthSelected: 3,
      font: { color: rgba(c.fg.startsWith("#") ? c.fg : "#1f2430", dim ? 0.35 : 0.9), size: 11,
              strokeWidth: 3, strokeColor: rgba(c.bg.startsWith("#") ? c.bg : "#ffffff", 0.85) },
    };
  }
  function nodeStyle(n) {
    if (isDoc(n)) return docStyle(n);
    const c = colors();
    const base = commColor(n.community);
    const isH = S.hl.has(n.id);
    const dim = S.hl.size > 0 && !isH;
    const bg = isH ? c.hl : rgba(base, dim ? 0.25 : 1);
    return {
      id: n.id, label: n.label, value: Math.max(1, n.degree || 1) * (isH ? 1.6 : 1), title: tooltip(n),
      color: { background: bg, border: isH ? "#a30000" : rgba(base, dim ? 0.25 : 1),
               highlight: { background: isH ? c.hl : base, border: c.fg },
               hover: { background: isH ? c.hl : base, border: c.fg } },
      borderWidth: isH ? 3 : 1, borderWidthSelected: 3,
      font: { color: rgba(c.fg.startsWith("#") ? c.fg : "#1f2430", dim ? 0.35 : 1),
              strokeWidth: 3, strokeColor: rgba(c.bg.startsWith("#") ? c.bg : "#ffffff", 0.85) },
    };
  }
  function edgeIsHl(e) {
    if (e.kind === "cites") return S.hlDocs.has(e.source) && S.hl.has(e.target);
    if (S.hlEdges.size) return S.hlEdges.has(e.id);
    return S.hl.has(e.source) && S.hl.has(e.target);
  }
  function edgeStyle(e) {
    const c = colors();
    const isH = edgeIsHl(e);
    if (e.kind === "cites") {
      return { id: e.id, from: e.source, to: e.target, dashes: [4, 4], width: isH ? 2 : 1, title: e.description,
               color: { color: isH ? rgba(c.hl.startsWith("#") ? c.hl : "#ff2d2d", 0.7) : rgba(c.edge, 0.35),
                        highlight: c.fg, hover: c.fg } };
    }
    const dim = S.hl.size > 0 && !isH;
    const el = document.createElement("div");
    el.className = "tip";
    el.innerHTML = `<b>${esc(e.source)} &harr; ${esc(e.target)}</b><div>${esc((e.description || "").slice(0, 160))}</div>`;
    return {
      id: e.id, from: e.source, to: e.target, title: el, width: isH ? 4 : 1, hoverWidth: 1.5,
      color: { color: isH ? c.hl : rgba(c.edge, dim ? 0.15 : 0.55), highlight: isH ? c.hl : (cssVar("--accent") || "#6b7cff"), hover: c.fg },
    };
  }
  function labelOptions() {
    const threshold = { zoom: 7, on: 1, off: 1000 }[S.labels];
    return { nodes: { shape: "dot", scaling: { min: 8, max: 34, label: { enabled: true, min: 11, max: 22, drawThreshold: threshold, maxVisible: 40 } } } };
  }

  // ---------- build ----------
  function index(payload) {
    S.byId = {}; S.edgeById = {}; S.adj = {};
    payload.nodes.forEach((n) => { S.byId[n.id] = n; S.adj[n.id] = []; });
    const seen = new Set();
    payload.edges = payload.edges.filter((e) => S.byId[e.source] && S.byId[e.target]).map((e) => {
      let id = String(e.id);
      while (seen.has(id)) id += "'";
      seen.add(id);
      e.id = id;
      S.edgeById[id] = e;
      S.adj[e.source].push(e.target);
      S.adj[e.target].push(e.source);
      return e;
    });
    const h = payload.highlight || {};
    S.hl = new Set((h.nodes || []).filter((id) => S.byId[id]));
    S.hlEdges = new Set((h.edges || []).map(String).filter((id) => S.edgeById[id]));
    S.pinned = new Set(((payload.options || {}).pinned || []).filter((id) => S.byId[id]));
    S.hlDocs = new Set((h.documents || []).filter((id) => S.byId[id]));
    S.halo = (h.halo || []).map(String);
    S.trace = (payload.trace || []).map((s) => Object.assign({}, s, { nodes: (s.nodes || []).filter((id) => S.byId[id]) }));
    S.stepOf = {};
    S.trace.forEach((s) => {
      const badge = s.kind === "community" ? s.nodes.slice(0, 1) : s.nodes;
      badge.forEach((id) => { if (!(id in S.stepOf)) S.stepOf[id] = s.n; });
    });
  }

  function build(payload) {
    const prev = S.network ? S.network.getPositions() : {};
    const hadNetwork = !!S.network;
    if (S.network) { S.network.destroy(); S.network = null; }
    S.payload = payload;
    index(payload);
    const nodes = payload.nodes.map((n) => {
      const st = nodeStyle(n);
      if (prev[n.id]) { st.x = prev[n.id].x; st.y = prev[n.id].y; }
      return st;
    });
    const fresh = nodes.filter((n) => n.x === undefined).length;
    S.nodes = new vis.DataSet(nodes);
    S.edges = new vis.DataSet(payload.edges.map(edgeStyle));
    const opts = {
      autoResize: true,
      nodes: { shape: "dot" },
      edges: { smooth: false, selectionWidth: 2 },
      interaction: { hover: true, tooltipDelay: 120, hideEdgesOnDrag: payload.nodes.length > 200, multiselect: false },
      physics: { enabled: true, solver: "forceAtlas2Based",
                 forceAtlas2Based: { gravitationalConstant: -55, springLength: 110, springConstant: 0.06, avoidOverlap: 0.4 },
                 stabilization: { enabled: !hadNetwork || fresh > 0, iterations: 120, fit: true } },
    };
    S.network = new vis.Network($("net"), { nodes: S.nodes, edges: S.edges }, Object.assign(opts, labelOptions()));
    wireNetwork();
    if (S.selected && !S.byId[S.selected]) closeDrawer();
    renderLegend();
    applyVisibility(false);
    physicsBurst(!hadNetwork || fresh > payload.nodes.length * 0.2 ? 2000 : 300);
    startPulse();
  }

  function physicsBurst(ms) {
    clearTimeout(S.physicsTimer);
    S.network.setOptions({ physics: { enabled: true } });
    S.physicsTimer = setTimeout(() => {
      if (!S.physics && S.network) S.network.setOptions({ physics: { enabled: false } });
    }, ms);
  }

  function restyleAll() {
    if (!S.nodes) return;
    S.nodes.update(S.payload.nodes.map(nodeStyle));
    S.edges.update(S.payload.edges.map(edgeStyle));
    applyVisibility(false);
  }

  // ---------- visibility ----------
  function neighborhood(id, depth) {
    const seen = new Set([id]);
    let frontier = [id];
    for (let d = 0; d < depth; d++) {
      const next = [];
      frontier.forEach((n) => (S.adj[n] || []).forEach((m) => { if (!seen.has(m)) { seen.add(m); next.push(m); } }));
      frontier = next;
    }
    return seen;
  }
  function computeVisible() {
    let vis;
    const anchor = S.selected || (S.payload.highlight.order || [])[0];
    if (S.depth > 0 && anchor && S.byId[anchor]) {
      vis = neighborhood(anchor, S.depth);
    } else if (S.view === "answer" && (S.hl.size || S.pinned.size)) {
      vis = new Set([...S.hl, ...S.pinned]);
      S.hl.forEach((n) => (S.adj[n] || []).forEach((m) => { if (!isDoc(S.byId[m])) vis.add(m); }));
      S.hlDocs.forEach((d) => vis.add(d));
    } else {
      vis = new Set(Object.keys(S.byId));
    }
    return new Set([...vis].filter((id) => {
      const n = S.byId[id];
      if (isDoc(n)) return S.showDocs || id === S.selected;
      return !S.hiddenComms.has(n.community) || id === S.selected;
    }));
  }
  function applyVisibility(fit) {
    if (!S.nodes) return;
    S.visible = computeVisible();
    S.nodes.update(S.payload.nodes.map((n) => ({ id: n.id, hidden: !S.visible.has(n.id) })));
    S.edges.update(S.payload.edges.map((e) => ({ id: e.id, hidden: !(S.visible.has(e.source) && S.visible.has(e.target)) })));
    $("empty").hidden = S.visible.size > 0;
    $("viewAnswer").setAttribute("aria-pressed", String(S.view === "answer"));
    $("viewFull").setAttribute("aria-pressed", String(S.view === "full"));
    $("depthVal").textContent = S.depth ? String(S.depth) : "off";
    if (fit) fitVisible();
  }
  function fitVisible() {
    const ids = [...S.visible];
    if (ids.length) S.network.fit({ nodes: ids, animation: { duration: 450, easingFunction: "easeInOutQuad" } });
  }
  function ensureVisible(ids) {
    let changed = false;
    ids.forEach((id) => {
      const n = S.byId[id];
      if (isDoc(n) && !S.showDocs) { S.showDocs = true; syncDocsButton(); changed = true; }
      if (n && !isDoc(n) && S.hiddenComms.has(n.community)) { S.hiddenComms.delete(n.community); changed = true; }
    });
    if (ids.some((id) => !computeVisible().has(id))) { S.view = "full"; S.depth = 0; $("depth").value = "0"; changed = true; }
    if (changed) { renderLegend(); applyVisibility(false); }
  }

  // ---------- drawing overlays (pulse, replay numbers, focus flash) ----------
  function nodeRadius(id) {
    const node = S.network.body.nodes[id];
    const r = node && node.shape && node.shape.radius;
    return r || (node && node.options && node.options.size) || 10;
  }
  function drawBadge(ctx, id, label) {
    const p = S.network.getPositions([id])[id];
    if (!p) return;
    const r = nodeRadius(id);
    ctx.fillStyle = "#111827";
    ctx.beginPath(); ctx.arc(p.x + r, p.y - r, 10, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#ffffff"; ctx.font = "bold 11px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(String(label), p.x + r, p.y - r);
  }
  function haloNodes(c) {
    const step = S.trace.find((s) => s.kind === "community" && s.ids.includes(c));
    const ids = step ? step.nodes : S.payload.nodes.filter((n) => n.community === c && S.hl.has(n.id)).map((n) => n.id);
    return ids.filter((id) => S.visible.has(id));
  }
  function beforeDrawing(ctx) {
    S.halo.forEach((c) => {
      const ids = haloNodes(c);
      if (!ids.length) return;
      const pos = S.network.getPositions(ids);
      const pts = ids.map((id) => pos[id]).filter(Boolean);
      const cx = pts.reduce((a, p) => a + p.x, 0) / pts.length;
      const cy = pts.reduce((a, p) => a + p.y, 0) / pts.length;
      const r = Math.max(...pts.map((p) => Math.hypot(p.x - cx, p.y - cy))) + 45;
      const col = commColor(c) === "#8f99a8" ? (colors().hl || "#ff2d2d") : commColor(c);
      ctx.fillStyle = rgba(col.startsWith("#") ? col : "#ff2d2d", 0.08);
      ctx.strokeStyle = rgba(col.startsWith("#") ? col : "#ff2d2d", 0.45);
      ctx.lineWidth = 2; ctx.setLineDash([8, 6]);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.setLineDash([]);
    });
  }
  function drawTrace(ctx, steps) {
    const c = colors();
    steps.filter((s) => s.kind === "hop" && s.nodes.length > 1).forEach((s) => {
      const pos = S.network.getPositions(s.nodes);
      ctx.lineWidth = 5;
      ctx.strokeStyle = rgba(c.hl.startsWith("#") ? c.hl : "#ff2d2d", 0.55);
      ctx.beginPath();
      s.nodes.forEach((id, i) => { const p = pos[id]; if (p) { i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); } });
      ctx.stroke();
    });
    const seen = new Set();
    steps.forEach((s) => (s.kind === "community" ? s.nodes.slice(0, 1) : s.nodes).forEach((id) => {
      if (!seen.has(id) && S.visible.has(id)) { seen.add(id); drawBadge(ctx, id, s.n); }
    }));
  }
  function afterDrawing(ctx) {
    const c = colors();
    const now = performance.now();
    if (S.replay && S.replay.trace) {
      drawTrace(ctx, S.replay.trace.slice(0, S.replay.step));
    } else if (S.replay) {
      const reached = S.replay.order.slice(0, S.replay.step);
      const pos = S.network.getPositions(reached);
      ctx.lineWidth = 5;
      ctx.strokeStyle = rgba(c.hl.startsWith("#") ? c.hl : "#ff2d2d", 0.7);
      ctx.beginPath();
      reached.forEach((id, i) => { const p = pos[id]; if (p) { i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); } });
      ctx.stroke();
      reached.forEach((id, i) => {
        const p = pos[id];
        if (!p) return;
        const r = nodeRadius(id);
        ctx.fillStyle = "#111827";
        ctx.beginPath(); ctx.arc(p.x + r, p.y - r, 11, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = "#ffffff"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), p.x + r, p.y - r);
      });
    } else if (S.hl.size || S.trace.length) {
      if (S.trace.length) drawTrace(ctx, S.trace);
      const t = (Math.sin(now / 320) + 1) / 2;
      const ids = [...S.hl].filter((id) => S.visible.has(id));
      const pos = S.network.getPositions(ids);
      ctx.lineWidth = 3;
      ids.forEach((id) => {
        const p = pos[id];
        if (!p) return;
        ctx.strokeStyle = rgba(c.hl.startsWith("#") ? c.hl : "#ff2d2d", 0.55 * (1 - t));
        ctx.beginPath(); ctx.arc(p.x, p.y, nodeRadius(id) + 3 + t * 9, 0, 2 * Math.PI); ctx.stroke();
      });
    }
    if (S.flash && now < S.flash.until) {
      const pos = S.network.getPositions(S.flash.ids);
      ctx.lineWidth = 3;
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = cssVar("--focus") || "#2f7cf6";
      S.flash.ids.forEach((id) => { const p = pos[id]; if (p) { ctx.beginPath(); ctx.arc(p.x, p.y, nodeRadius(id) + 8, 0, 2 * Math.PI); ctx.stroke(); } });
      ctx.setLineDash([]);
    }
  }
  function startPulse() {
    clearInterval(S.pulseTimer);
    S.pulseTimer = setInterval(() => {
      const active = S.hl.size || S.replay || S.halo.length || (S.flash && performance.now() < S.flash.until);
      if (active && S.network && !document.hidden) S.network.redraw();
    }, 70);
  }

  // ---------- drawer ----------
  function closeDrawer() {
    $("drawer").hidden = true;
    S.selected = null;
    if (S.network) S.network.unselectAll();
  }
  function sourcesHtml(sources) {
    if (!sources || !sources.length) return '<p class="muted">No source excerpt available.</p>';
    return sources.map((s) => `<div class="src"><b>${esc(s.doc)}</b>&ldquo;${esc(s.excerpt)}&rdquo;</div>`).join("");
  }
  function openDocument(n, fromUser) {
    const ents = [...new Set(S.adj[n.id] || [])].sort((a, b) => (S.byId[b].degree || 0) - (S.byId[a].degree || 0));
    const cited = ents.filter((m) => S.hl.has(m));
    const steps = S.trace.filter((s) => s.nodes.includes(n.id)).map((s) => s.n);
    $("drawerBody").innerHTML =
      `<h3>${esc(n.label)}</h3>
       <div class="chips">
         <span class="chip">&#9635; document</span>
         ${S.hlDocs.has(n.id) ? '<span class="chip hl">&#9679; cited by this answer</span>' : ""}
         ${steps.length ? `<span class="chip">trace step ${steps.join(", ")}</span>` : ""}
         <span class="chip">${ents.length} entities</span>
       </div>
       <h4>Excerpts</h4>${sourcesHtml(n.sources)}
       ${cited.length ? `<h4>Evidence entities (${cited.length})</h4>` : ""}
       <div class="nbrs">${cited.map((m) => `<button type="button" data-nbr="${esc(m)}">${esc(m)}</button>`).join("")}</div>
       <h4>Mentions (${ents.length})</h4>
       <div class="nbrs">${ents.slice(0, 30).map((m) => `<button type="button" data-nbr="${esc(m)}" aria-label="Open ${esc(m)}"><i class="dot" style="background:${esc(commColor(S.byId[m].community))}"></i>${esc(m)}</button>`).join("")}</div>
       <div class="actions"><button type="button" class="primary" id="askBtn">&#128172; Ask what it says</button></div>`;
    $("drawer").hidden = false;
    $("drawerBody").querySelectorAll("[data-nbr]").forEach((b) => b.addEventListener("click", () => jumpTo(b.dataset.nbr, true)));
    $("askBtn").addEventListener("click", () => emit({ type: "ask", question: `What does ${n.label} say?` }));
    if (fromUser) emit({ type: "doc_click", id: n.id });
  }
  function openNode(id, fromUser) {
    const n = S.byId[id];
    if (!n) return;
    S.selected = id;
    S.network.selectNodes([id]);
    if (isDoc(n)) { openDocument(n, fromUser); return; }
    const nbrs = [...new Set(S.adj[id] || [])].sort((a, b) => (S.byId[b].degree || 0) - (S.byId[a].degree || 0)).slice(0, 30);
    $("drawerBody").innerHTML =
      `<h3>${esc(n.label)}</h3>
       <div class="chips">
         <span class="chip">${esc(n.type || "entity")}</span>
         <span class="chip"><i class="dot" style="background:${esc(commColor(n.community))}"></i>${esc(commTitle(n.community))}</span>
         ${S.hl.has(id) ? '<span class="chip hl">&#9679; evidence</span>' : ""}
         <span class="chip">degree ${esc(n.degree)}</span>
       </div>
       <p>${esc(n.description || "No description.")}</p>
       <h4>Sources</h4>${sourcesHtml(n.sources)}
       <h4>Neighbours (${nbrs.length})</h4>
       <div class="nbrs">${nbrs.map((m) => `<button type="button" data-nbr="${esc(m)}" aria-label="Open ${esc(m)}"><i class="dot" style="background:${esc(commColor(S.byId[m].community))}"></i>${esc(m)}</button>`).join("")}</div>
       <div class="actions">
         <button type="button" class="primary" id="askBtn" aria-label="Ask about ${esc(n.label)}">&#128172; Ask about this</button>
         <button type="button" id="expandBtn" aria-label="Expand neighbours of ${esc(n.label)}">&#10010; Expand neighbours</button>
       </div>`;
    $("drawer").hidden = false;
    $("drawerBody").querySelectorAll("[data-nbr]").forEach((b) => b.addEventListener("click", () => jumpTo(b.dataset.nbr, true)));
    $("askBtn").addEventListener("click", () => { emit({ type: "ask", question: "Tell me about " + n.label }); status("Asking about " + n.label); });
    $("expandBtn").addEventListener("click", () => { emit({ type: "expand", id: id }); status("Expanding " + n.label); });
    if (S.depth > 0) applyVisibility(false);
    if (fromUser) emit({ type: "node_click", id: id });
  }
  function openEdge(id, fromUser) {
    const e = S.edgeById[id];
    if (!e) return;
    S.selected = null;
    S.network.selectEdges([id]);
    $("drawerBody").innerHTML =
      `<h3>Relationship</h3>
       <div class="chips">${edgeIsHl(e) ? '<span class="chip hl">&#9679; evidence</span>' : ""}<span class="chip">id ${esc(e.id)}</span></div>
       <div class="nbrs"><button type="button" data-nbr="${esc(e.source)}">${esc(e.source)}</button>
         <span aria-hidden="true">&harr;</span><button type="button" data-nbr="${esc(e.target)}">${esc(e.target)}</button></div>
       <h4>Description</h4><p>${esc(e.description || "No description.")}</p>
       <h4>Source</h4>${sourcesHtml(e.sources)}
       <div class="actions"><button type="button" class="primary" id="askBtn">&#128172; Ask how they connect</button></div>`;
    $("drawer").hidden = false;
    $("drawerBody").querySelectorAll("[data-nbr]").forEach((b) => b.addEventListener("click", () => jumpTo(b.dataset.nbr, true)));
    $("askBtn").addEventListener("click", () => emit({ type: "ask", question: `How is ${e.source} connected to ${e.target}?` }));
    if (fromUser) emit({ type: "edge_click", id: id });
  }
  function jumpTo(id, fromUser) {
    if (!S.byId[id]) return;
    ensureVisible([id]);
    S.network.focus(id, { scale: Math.max(S.network.getScale(), 1.1), animation: { duration: 450 } });
    openNode(id, fromUser);
  }

  // ---------- focus from Python ----------
  function applyFocus(f) {
    const ids = new Set((f.nodes || []).filter((id) => S.byId[id]));
    (f.edges || []).map(String).forEach((eid) => { const e = S.edgeById[eid]; if (e) { ids.add(e.source); ids.add(e.target); } });
    const list = [...ids];
    if (!list.length) { status("Nothing to focus in the current graph"); return; }
    ensureVisible(list);
    const edgeIds = (f.edges || []).map(String).filter((eid) => S.edgeById[eid]);
    S.network.setSelection({ nodes: list, edges: edgeIds }, { unselectAll: true, highlightEdges: false });
    S.flash = { ids: list, until: performance.now() + 2600 };
    if (list.length === 1) {
      S.network.focus(list[0], { scale: 1.3, animation: { duration: 600 } });
      openNode(list[0], false);
    } else {
      S.network.fit({ nodes: list, animation: { duration: 600, easingFunction: "easeInOutQuad" } });
    }
    status(`Focused ${list.length} node${list.length > 1 ? "s" : ""}`);
  }

  // ---------- replay ----------
  const KIND = { seed: "Seed entities", hop: "Graph hop", community: "Community theme",
                 chunk: "Evidence chunks", document: "Source documents" };
  function showCaption(s) {
    const el = $("caption");
    el.innerHTML = `<b>Step ${s.n} of ${S.trace.length}</b>${esc(KIND[s.kind] || s.kind)}: ${esc(s.note || s.ids.join(", "))}`;
    el.hidden = false;
  }
  function replayTrace() {
    clearInterval(S.replayTimer);
    ensureVisible([...new Set(S.trace.flatMap((s) => s.nodes))]);
    S.replay = { trace: S.trace, step: 0 };
    $("replay").disabled = true;
    const advance = () => {
      S.replay.step += 1;
      const s = S.trace[S.replay.step - 1];
      if (s) {
        showCaption(s);
        const reached = [...new Set(S.trace.slice(0, S.replay.step).flatMap((x) => x.nodes))].filter((id) => S.visible.has(id));
        S.network.selectNodes(s.nodes.filter((id) => S.visible.has(id)));
        if (reached.length) S.network.fit({ nodes: reached, animation: { duration: 500, easingFunction: "easeInOutQuad" } });
      }
      if (S.replay.step >= S.trace.length) {
        clearInterval(S.replayTimer);
        setTimeout(() => {
          S.replay = null; $("replay").disabled = false; $("caption").hidden = true;
          S.network.unselectAll(); S.network.redraw();
        }, 2600);
      }
    };
    advance();
    S.replayTimer = setInterval(advance, 1700);
  }
  function replay() {
    if (S.trace.length) { replayTrace(); return; }
    const order = (S.payload.highlight.order || []).filter((id) => S.byId[id]);
    if (!order.length) { status("No evidence path to replay"); return; }
    ensureVisible(order);
    S.replay = { order: order, step: 0 };
    S.network.fit({ nodes: order, animation: { duration: 400 } });
    $("replay").disabled = true;
    const timer = setInterval(() => {
      S.replay.step += 1;
      const id = order[S.replay.step - 1];
      if (id) { S.network.selectNodes([id]); status(`Step ${S.replay.step} of ${order.length}: ${id}`); }
      if (S.replay.step >= order.length) {
        clearInterval(timer);
        setTimeout(() => { S.replay = null; $("replay").disabled = false; S.network.unselectAll(); S.network.redraw(); }, 1500);
      }
    }, 600);
  }

  // ---------- legend ----------
  function renderLegend() {
    const box = $("legendItems");
    box.innerHTML = (S.payload.communities || []).map((c) =>
      `<label><input type="checkbox" data-comm="${esc(c.id)}" ${S.hiddenComms.has(c.id) ? "" : "checked"} aria-label="Show ${esc(c.title)}">
       <i class="dot" style="background:${esc(c.color)}"></i><span class="t" title="${esc(c.title)}">${esc(c.title)}</span><small>${esc(c.size)}</small></label>`).join("");
    box.querySelectorAll("input[data-comm]").forEach((cb) => cb.addEventListener("change", () => {
      cb.checked ? S.hiddenComms.delete(cb.dataset.comm) : S.hiddenComms.add(cb.dataset.comm);
      applyVisibility(false);
    }));
  }

  // ---------- search ----------
  let suggestIdx = 0;
  function suggestions(q) {
    const s = q.trim().toLowerCase();
    if (!s) return [];
    const all = S.payload.nodes.filter((n) => n.label.toLowerCase().includes(s));
    all.sort((a, b) => (b.label.toLowerCase().startsWith(s) - a.label.toLowerCase().startsWith(s)) || (b.degree - a.degree));
    return all.slice(0, 8);
  }
  function renderSuggest() {
    const list = suggestions($("search").value);
    const ul = $("suggest");
    ul.hidden = !list.length;
    suggestIdx = Math.min(suggestIdx, Math.max(0, list.length - 1));
    ul.innerHTML = list.map((n, i) =>
      `<li role="option" data-id="${esc(n.id)}" aria-selected="${i === suggestIdx}"><i class="dot" style="background:${esc(commColor(n.community))}"></i>${esc(n.label)}<small>${esc(n.type)}</small></li>`).join("");
    ul.querySelectorAll("li").forEach((li) => li.addEventListener("mousedown", (ev) => { ev.preventDefault(); pick(li.dataset.id); }));
    return list;
  }
  function pick(id) {
    $("search").value = "";
    $("suggest").hidden = true;
    $("search").blur();
    jumpTo(id, true);
  }

  // ---------- toolbar + keyboard ----------
  function exportPng() {
    const src = document.querySelector("#net canvas");
    if (!src) return;
    const out = document.createElement("canvas");
    out.width = src.width; out.height = src.height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = colors().bg || "#ffffff";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(src, 0, 0);
    const url = out.toDataURL("image/png");
    S.lastExport = { width: out.width, height: out.height, bytes: url.length };
    const a = document.createElement("a");
    a.href = url; a.download = "themegraph.png";
    document.body.appendChild(a); a.click(); a.remove();
    status("Exported themegraph.png");
  }
  function reset() {
    S.view = (S.payload.options || {}).show_full ? "full" : "answer";
    S.depth = 0; $("depth").value = "0";
    S.hiddenComms.clear(); S.labels = "zoom"; $("labels").textContent = "Aa zoom";
    S.network.setOptions(labelOptions());
    closeDrawer(); renderLegend(); applyVisibility(false);
    physicsBurst(1500);
    setTimeout(fitVisible, 300);
    status("View reset");
  }
  function toggleFullscreen() {
    const app = $("app");
    if (document.fullscreenElement) { document.exitFullscreen(); return; }
    if (app.requestFullscreen) {
      app.requestFullscreen().catch(() => {
        const tall = S.height < 900;
        S.height = tall ? 1000 : ((S.payload && S.payloadHeight) || 680);
        app.style.height = S.height + "px"; Streamlit.height(S.height);
      });
    }
  }
  function syncDocsButton() {
    $("docs").setAttribute("aria-pressed", String(S.showDocs));
  }
  function wireToolbar() {
    $("docs").addEventListener("click", () => {
      S.showDocs = !S.showDocs;
      syncDocsButton();
      applyVisibility(false);
      status(S.showDocs ? "Documents shown" : "Documents hidden");
    });
    $("viewAnswer").addEventListener("click", () => { S.view = "answer"; applyVisibility(true); });
    $("viewFull").addEventListener("click", () => { S.view = "full"; applyVisibility(true); });
    $("depth").addEventListener("input", (e) => {
      S.depth = Number(e.target.value);
      if (S.depth && !S.selected && !(S.payload.highlight.order || []).length) status("Select a node to explore its local graph");
      applyVisibility(true);
    });
    $("fit").addEventListener("click", fitVisible);
    $("reset").addEventListener("click", reset);
    $("replay").addEventListener("click", replay);
    $("export").addEventListener("click", exportPng);
    $("fullscreen").addEventListener("click", toggleFullscreen);
    $("labels").addEventListener("click", () => {
      S.labels = { zoom: "on", on: "off", off: "zoom" }[S.labels];
      $("labels").textContent = "Aa " + S.labels;
      S.network.setOptions(labelOptions());
      status("Labels: " + { zoom: "shown when zoomed in", on: "always shown", off: "hidden" }[S.labels]);
    });
    $("physics").addEventListener("click", () => {
      S.physics = !S.physics;
      clearTimeout(S.physicsTimer);
      S.network.setOptions({ physics: { enabled: S.physics } });
      $("physics").setAttribute("aria-pressed", String(S.physics));
      status("Physics " + (S.physics ? "on" : "off"));
    });
    $("drawerClose").addEventListener("click", closeDrawer);
    $("legendToggle").addEventListener("click", () => {
      const col = $("legend").classList.toggle("collapsed");
      $("legendToggle").setAttribute("aria-expanded", String(!col));
      $("legendToggle").innerHTML = col ? "+" : "&#8211;";
    });
    const search = $("search");
    search.addEventListener("input", () => { suggestIdx = 0; renderSuggest(); });
    search.addEventListener("blur", () => setTimeout(() => { $("suggest").hidden = true; }, 120));
    search.addEventListener("keydown", (e) => {
      const list = suggestions(search.value);
      if (e.key === "ArrowDown") { suggestIdx = Math.min(suggestIdx + 1, list.length - 1); renderSuggest(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { suggestIdx = Math.max(suggestIdx - 1, 0); renderSuggest(); e.preventDefault(); }
      else if (e.key === "Enter" && list.length) { pick(list[suggestIdx].id); e.preventDefault(); }
      else if (e.key === "Escape") { search.value = ""; $("suggest").hidden = true; search.blur(); }
    });
    document.addEventListener("keydown", (e) => {
      const typing = e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA");
      if (e.key === "/" && !typing) { e.preventDefault(); search.focus(); }
      else if (e.key === "Escape" && !typing) {
        const had = S.selected || !$("drawer").hidden;
        closeDrawer();
        if (S.depth) applyVisibility(false);
        if (had) emit({ type: "clear" });
      } else if ((e.key === "f" || e.key === "F") && !typing) { fitVisible(); }
    });
    document.addEventListener("fullscreenchange", () => setTimeout(() => { S.network.redraw(); fitVisible(); }, 150));
  }
  function wireNetwork() {
    S.network.on("click", (p) => {
      if (p.nodes.length) openNode(p.nodes[0], true);
      else if (p.edges.length) openEdge(p.edges[0], true);
      else if (!$("drawer").hidden) closeDrawer();
    });
    S.network.on("beforeDrawing", beforeDrawing);
    S.network.on("afterDrawing", afterDrawing);
  }

  // ---------- theme ----------
  function applyTheme(theme) {
    let base = theme && theme.base;
    if (!base) base = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    if (base === S.theme) return;
    S.theme = base;
    document.documentElement.dataset.theme = base;
    restyleAll();
  }
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { S.theme = null; applyTheme(S.lastTheme); });
  }

  // ---------- render ----------
  function onRender(args, theme) {
    const height = Number(args.height) || 680;
    if (!document.fullscreenElement && height !== S.payloadHeight) {
      S.payloadHeight = height; S.height = height;
      $("app").style.height = height + "px";
    }
    Streamlit.height(S.height + 2);
    S.lastTheme = theme;
    const payload = args.payload || { nodes: [], edges: [], highlight: {}, communities: [], options: {} };
    payload.highlight = payload.highlight || {};
    const sig = hash(JSON.stringify(payload));
    if (sig !== S.sig) {
      S.sig = sig;
      const firstBuild = !S.network;
      const showFull = !!(payload.options || {}).show_full;
      if (firstBuild || showFull !== S.lastShowFull) S.view = showFull ? "full" : "answer";
      S.lastShowFull = showFull;
      if ((payload.highlight.documents || []).length) S.showDocs = true;
      syncDocsButton();
      clearInterval(S.replayTimer); S.replay = null; $("caption").hidden = true; $("replay").disabled = false;
      build(payload);
      if (!firstBuild) setTimeout(fitVisible, 350);
    }
    applyTheme(theme);
    const focus = args.focus || {};
    if (focus.nonce != null && focus.nonce !== S.focusNonce) {
      const first = S.focusNonce === null;
      S.focusNonce = focus.nonce;
      if (!first || (focus.nodes || []).length || (focus.edges || []).length) setTimeout(() => applyFocus(focus), 450);
    }
  }

  window.addEventListener("message", (event) => {
    const data = event.data || {};
    if (data.type === "streamlit:render") onRender(data.args || {}, data.theme);
  });
  wireToolbar();
  Streamlit.ready();
  window.TG = { S: S, openNode: openNode, openEdge: openEdge, applyFocus: applyFocus, replay: replay, exportPng: exportPng,
                jumpTo: jumpTo, toggleDocs: () => $("docs").click() };
})();

// SFC Runtime Verifier - front-end (vanilla JS + vis-network).

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const C = {
  ink: "#15171c", ink2: "#5a6170", line: "#9aa3b2", accent: "#2a78d6",
  ok: "#1d9a55", okDark: "#147a41", warn: "#eb6834", and: "#e34948", andDark: "#b73232",
  visited: "#c4c9d2", visitedBorder: "#a5acb8", node: "#ffffff",
};
const ACTIVE_R = 21;

const state = {
  meta: null, graphs: [], g: null, net: null, nodes: null, edges: null,
  trace: null, frame: 0, timer: null, visited: new Set(), visitedFrame: -1,
  styleKey: new Map(), firedNow: [], edgeShape: {}, scope: "involved",
  checkCache: new Map(), evalTimer: null, perGraphSort: { key: "graph", dir: 1 },
};

// ------------------------------------------------------------------ utilities
async function request(url, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function toast(msg, ms = 4000) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.classList.add("hidden"), ms);
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x, d = 1) => (x === null || x === undefined ? "–" : `${(100 * x).toFixed(d)} %`);
const fmt = (x) => Number(x).toLocaleString("en-US");
const delta = (i, j, k) => `&delta;<sub>${i},${j}</sub><sup>(${k})</sup>`;
const chips = (items, cls) => items.length
  ? items.map((x) => `<span class="chip ${cls}">${x}</span>`).join("")
  : `<span class="empty">&empty;</span>`;

// ------------------------------------------------------------------ graph view
function baseStyle(step) {
  if (step.divergence === "AND") {
    return { color: { background: C.and, border: C.andDark }, font: { color: "#fff" }, borderWidth: 1.5 };
  }
  if (step.divergence === "AND/OR") {
    return { color: { background: C.node, border: C.and }, font: { color: C.ink }, borderWidth: 2.5 };
  }
  return { color: { background: C.node, border: C.line }, font: { color: C.ink }, borderWidth: 1.5 };
}

function activeRenderer(violation) {
  return ({ ctx, x, y, label, state: st }) => ({
    drawNode() {
      ctx.save();
      if (violation) {
        ctx.beginPath(); ctx.arc(x, y, ACTIVE_R + 6, 0, 2 * Math.PI);
        ctx.strokeStyle = C.warn; ctx.lineWidth = 4; ctx.stroke();
      }
      ctx.beginPath(); ctx.arc(x, y, ACTIVE_R, 0, 2 * Math.PI);
      ctx.fillStyle = C.ok; ctx.fill();
      ctx.lineWidth = st.selected ? 3 : 2; ctx.strokeStyle = st.selected ? C.accent : C.okDark; ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, ACTIVE_R - 5, 0, 2 * Math.PI);
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = "#ffffff"; ctx.font = "700 13px system-ui, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(label, x, y + 0.5);
      ctx.restore();
    },
    nodeDimensions: { width: 2 * ACTIVE_R, height: 2 * ACTIVE_R },
  });
}

function edgeSmooth(edge, shape) {
  const s = shape || (edge.back ? "cw" : "auto");
  if (s === "straight") return { enabled: false };
  if (s === "cw") return { enabled: true, type: "curvedCW", roundness: edge.back ? 0.35 : 0.22 };
  if (s === "ccw") return { enabled: true, type: "curvedCCW", roundness: edge.back ? 0.35 : 0.22 };
  return { enabled: true, type: "cubicBezier", forceDirection: "vertical", roundness: 0.45 };
}

function computeLayout(g) {
  // Layered drawing: rows = longest-path level, order inside a row by barycentre sweeps.
  const fwd = g.edges.filter((e) => !e.back);
  const rows = [];
  g.steps.forEach((s) => (rows[g.levels[s.id]] ||= []).push(s.id));
  const pos = new Map();
  const place = () => rows.forEach((row) => row.forEach((id, k) => pos.set(id, k - (row.length - 1) / 2)));
  place();
  const preds = new Map(), succs = new Map();
  fwd.forEach((e) => {
    (preds.get(e.to) || preds.set(e.to, []).get(e.to)).push(e.from);
    (succs.get(e.from) || succs.set(e.from, []).get(e.from)).push(e.to);
  });
  const bary = (id, nb) => { const n = nb.get(id) || []; return n.length ? n.reduce((a, x) => a + pos.get(x), 0) / n.length : pos.get(id); };
  for (let it = 0; it < 6; it++) {
    const order = it % 2 === 0 ? rows.keys() : [...rows.keys()].reverse();
    for (const L of order) {
      const row = rows[L];
      if (!row) continue;
      const nb = it % 2 === 0 ? preds : succs;
      const key = new Map(row.map((id) => [id, bary(id, nb)]));
      row.sort((a, b) => key.get(a) - key.get(b));
      row.forEach((id, k) => pos.set(id, k - (row.length - 1) / 2));
    }
  }
  const out = {};
  rows.forEach((row, L) => row && row.forEach((id) => { out[id] = { x: pos.get(id) * 118, y: L * 72 }; }));
  return out;
}

function buildNetwork() {
  const g = state.g;
  if (state.net) state.net.destroy();
  state.styleKey.clear();
  const xy = computeLayout(g);
  state.nodes = new vis.DataSet(g.steps.map((s) => ({
    id: s.id, label: String(s.id), x: xy[s.id].x, y: xy[s.id].y, shape: "circle",
    title: `${s.name} · ${s.duration} s`, ...baseStyle(s),
  })));
  state.edges = new vis.DataSet(g.edges.map((e) => ({
    id: `${e.from}-${e.to}`, from: e.from, to: e.to, label: String(e.k), dashes: e.back,
    smooth: edgeSmooth(e, state.edgeShape[`${e.from}-${e.to}`]),
  })));
  const options = {
    layout: { randomSeed: 1, improvedLayout: false },
    physics: false,
    interaction: { hover: true, dragNodes: true, tooltipDelay: 120, zoomSpeed: 0.6 },
    nodes: { font: { size: 13, face: "system-ui, sans-serif" }, margin: 7,
      widthConstraint: { minimum: 30 }, chosen: false },
    edges: {
      arrows: { to: { enabled: true, scaleFactor: 0.55 } }, width: 1.3, selectionWidth: 1.5,
      color: { color: C.line, highlight: C.accent, hover: C.ink2, inherit: false },
      font: { size: 10, color: C.ink2, strokeWidth: 4, strokeColor: "#ffffff", align: "middle" },
      chosen: false,
    },
  };
  state.net = new vis.Network($("#network"), { nodes: state.nodes, edges: state.edges }, options);
  state.net.once("afterDrawing", () => state.net.fit({ animation: false }));
  state.net.on("click", onCanvasClick);
  toggleEdgeLabels();
  const st = g.stats;
  $("#graph-badge").innerHTML = `<b>${esc(g.name)}</b> &middot; ${st.steps} steps &middot; ${st.edges} edges
    &middot; ${st.and_divergences} AND / ${st.or_divergences} OR divergences &middot;
    ${st.and_convergences} AND / ${st.or_convergences} OR convergences`;
  $("#graph-badge").title = g.metadata.structure || g.description;
}

function relayout() {
  buildNetwork();
  state.styleKey.clear();
  state.firedNow = [];
  renderFrame();
}

function toggleEdgeLabels() {
  const on = $("#chk-labels").checked;
  state.edges.update(state.g.edges.map((e) => ({ id: `${e.from}-${e.to}`, label: on ? String(e.k) : " " })));
}

function setEdgeShape(id, shape) {
  state.edgeShape[id] = shape;
  const e = state.g.edges.find((x) => `${x.from}-${x.to}` === id);
  state.edges.update({ id, smooth: edgeSmooth(e, shape) });
}

// ------------------------------------------------------------------ playback
function framesInfo(t) {
  const tr = state.trace;
  const curr = tr.states[t];
  const prev = t > 0 ? tr.states[t - 1] : curr;
  const cs = new Set(curr), ps = new Set(prev);
  return {
    curr, prev,
    left: prev.filter((s) => !cs.has(s)),
    entered: curr.filter((s) => !ps.has(s)),
    fired: tr.fired[t] || [],
    alarm: tr.alarms[t],
    violations: tr.violations[t] || [],
  };
}

function updateVisited(t) {
  const tr = state.trace;
  const init = new Set(state.g.initial);
  const cycleStart = (u) => u > 0 && tr.states[u].some((s) => init.has(s)) && !tr.states[u - 1].some((s) => init.has(s));
  if (t === state.visitedFrame + 1 && t > 0) {
    if (cycleStart(t)) state.visited.clear();
    else tr.states[t - 1].forEach((s) => state.visited.add(s));
  } else if (t !== state.visitedFrame) {
    let start = 0;
    for (let u = t; u > 0; u--) if (cycleStart(u)) { start = u; break; }
    state.visited = new Set();
    for (let u = start; u < t; u++) tr.states[u].forEach((s) => state.visited.add(s));
  }
  state.visitedFrame = t;
}

function renderFrame() {
  if (!state.trace) return;
  const t = state.frame;
  updateVisited(t);
  const info = framesInfo(t);
  const active = new Set(info.curr);
  const violSteps = new Set(info.violations.map((v) => v.step).filter((s) => s !== null));

  const updates = [];
  for (const step of state.g.steps) {
    const id = step.id;
    const mode = active.has(id) ? "active" : state.visited.has(id) ? "visited" : "idle";
    const viol = violSteps.has(id);
    const key = `${mode}${viol}`;
    if (state.styleKey.get(id) === key) continue;
    state.styleKey.set(id, key);
    if (mode === "active") {
      updates.push({ id, shape: "custom", ctxRenderer: activeRenderer(viol) });
    } else {
      const base = mode === "visited"
        ? { color: { background: C.visited, border: C.visitedBorder }, font: { color: C.ink }, borderWidth: 1.5 }
        : baseStyle(step);
      if (viol) { base.color = { ...base.color, border: C.warn }; base.borderWidth = 4; }
      updates.push({ id, shape: "circle", ...base });
    }
  }
  if (updates.length) state.nodes.update(updates);

  const edgeUpdates = state.firedNow.map((id) => ({ id, color: { color: C.line }, width: 1.3 }));
  state.firedNow = info.fired.map(([i, j]) => `${i}-${j}`);
  state.firedNow.forEach((id) => edgeUpdates.push({ id, color: { color: C.ok }, width: 3.4 }));
  if (edgeUpdates.length) state.edges.update(edgeUpdates);

  renderVerdict(t, info);
  $("#time-label").textContent = `t = ${(t * state.trace.dt).toFixed(0)} s  (scan ${t} / ${state.trace.states.length - 1})`;
  $("#scrub").value = t;
  drawTimeline();
  if ($("#tab-matrices").classList.contains("active")) renderMatrices();
}

function renderVerdict(t, info) {
  const pill = $("#verdict-pill");
  const changed = info.left.length || info.entered.length;
  if (t === 0) { pill.className = "pill"; pill.textContent = "initial"; }
  else if (info.alarm) { pill.className = "pill bad"; pill.textContent = "VIOLATION"; }
  else { pill.className = "pill ok"; pill.textContent = changed ? "VALID" : "VALID · no change"; }
  $("#v-transition").textContent = t === 0 ? "S(0)" : `S(${t - 1}) → S(${t})`;
  $("#v-active").innerHTML = chips(info.curr, "active");
  $("#v-left").innerHTML = chips(info.left, "left");
  $("#v-entered").innerHTML = chips(info.entered, "entered");
  $("#v-fired").innerHTML = chips(info.fired.map(([i, j]) => `${i}→${j}`), "fired");
  $("#v-violations").innerHTML = info.violations.map((v) =>
    `<li><span class="rule">${v.rule}</span>${esc(v.message)}</li>`).join("");
}

function drawTimeline() {
  const cv = $("#timeline");
  const tr = state.trace;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#eef0f4"; ctx.fillRect(0, 8, w, h - 16);
  const n = tr.states.length - 1 || 1;
  const X = (t) => (t / n) * w;
  if (tr.attack) {
    ctx.fillStyle = "rgba(235,104,52,.16)";
    ctx.fillRect(X(tr.attack.start), 2, Math.max(2, X(tr.attack.horizon + 1) - X(tr.attack.start)), h - 4);
    ctx.fillStyle = "rgba(235,104,52,.45)";
    ctx.fillRect(X(tr.attack.start), 2, Math.max(2, X(tr.attack.end) - X(tr.attack.start)), h - 4);
  }
  ctx.fillStyle = "#d63f3f";
  tr.alarms.forEach((a, t) => { if (a) ctx.fillRect(X(t) - 1, 6, 2.5, h - 12); });
  ctx.fillStyle = C.accent;
  ctx.fillRect(X(state.frame) - 1.5, 0, 3, h);
}

function play() {
  if (state.timer) return stop();
  if (state.frame >= state.trace.states.length - 1) state.frame = 0;
  $("#btn-play").innerHTML = "&#10074;&#10074;";
  const tick = () => {
    if (state.frame >= state.trace.states.length - 1) return stop();
    state.frame += 1;
    renderFrame();
  };
  state.timer = setInterval(tick, 1000 / Number($("#speed").value));
}

function stop() {
  clearInterval(state.timer);
  state.timer = null;
  $("#btn-play").innerHTML = "&#9654;";
}

function seek(t) {
  if (!state.trace) return;
  state.frame = Math.max(0, Math.min(state.trace.states.length - 1, t));
  renderFrame();
}

// ------------------------------------------------------------------ scenarios
function loadTrace(trace, label) {
  stop();
  state.trace = trace;
  state.frame = 0;
  state.visitedFrame = -1;
  state.checkCache.clear();
  $("#scrub").max = trace.states.length - 1;
  const alarms = trace.alarms.filter(Boolean).length;
  let html = `<b>${esc(label)}</b> &middot; ${fmt(trace.states.length)} scans &middot;
    <b style="color:${alarms ? "#d63f3f" : "#1d9a55"}">${alarms} alarm${alarms === 1 ? "" : "s"}</b>`;
  if (trace.attack) {
    const a = trace.attack;
    const first = trace.alarms.findIndex((x, t) => x && t >= a.start && t <= a.horizon);
    html += `<br>${esc(a.title)} (${a.family}) on scans ${a.start}–${a.end - 1}: `;
    html += first >= 0 ? `<b style="color:#1d9a55">detected at scan ${first}</b> (latency ${first - a.start})`
      : `<b style="color:#d63f3f">not detected</b>${a.family === "stealthy" ? " – legal SFC execution" : ""}`;
    state.frame = Math.max(0, a.start - 3);
  }
  $("#trace-summary").innerHTML = html;
  renderFrame();
}

async function runScenario() {
  const btn = $("#btn-simulate");
  btn.disabled = true;
  try {
    const attack = $("#in-attack").value || null;
    const trace = await request("/api/scenario", {
      graph: state.g.name, scans: Number($("#in-scans").value), seed: Number($("#in-seed").value),
      attack, attack_seed: Number($("#in-seed").value) + 1,
    });
    loadTrace(trace, attack ? "Attack scenario" : "Normal operation");
  } catch (e) { toast(e.message); }
  finally { btn.disabled = false; }
}

async function importCsv(file) {
  try {
    const csv = await file.text();
    const trace = await request("/api/trace", { graph: state.g.name, csv });
    loadTrace(trace, file.name);
  } catch (e) { toast(`Import failed: ${e.message}`); }
}

async function selectGraph(name) {
  stop();
  state.g = await request(`/api/graphs/${encodeURIComponent(name)}`);
  state.edgeShape = {};
  state.trace = null;
  buildNetwork();
  $("#selection").innerHTML = "Click a step or an edge.";
  await runScenario();
}

// ------------------------------------------------------------------ selection panel
function onCanvasClick(params) {
  const g = state.g;
  if (params.nodes.length) {
    const s = g.steps[params.nodes[0]];
    const groups = Object.entries(s.divergence_groups).map(([k, v]) => `k=${k}: {${v.join(", ")}}`).join("<br>");
    const conv = s.convergence_groups.map((v) => `{${v.join(", ")}}`).join(", ");
    $("#selection").innerHTML = `<div class="sel-title">Step ${s.id} · ${esc(s.name)}</div>
      <div class="sel-grid">
        <span>Duration</span><b>${s.duration} s</b>
        <span>Divergence</span><b>${s.divergence}</b>
        <span>Groups G<sub>i</sub><sup>k</sup></span><span class="mono">${groups || "–"}</span>
        <span>Convergence</span><b>${s.convergence}</b>
        <span>Groups</span><span class="mono">${conv || "–"}</span>
      </div>`;
  } else if (params.edges.length) {
    const id = params.edges[0];
    const e = g.edges.find((x) => `${x.from}-${x.to}` === id);
    const cur = state.edgeShape[id] || "auto";
    const opts = [["auto", "auto"], ["straight", "straight"], ["cw", "curved ↶"], ["ccw", "curved ↷"]];
    $("#selection").innerHTML = `<div class="sel-title">Edge ${e.from} → ${e.to} &nbsp; <span class="mono">${delta(e.from, e.to, e.k)}</span></div>
      <div class="sel-grid"><span>Group k</span><b>${e.k}</b><span>Loop edge</span><b>${e.back ? "yes" : "no"}</b></div>
      <div class="segmented" style="margin-top:10px" id="edge-shape">${opts.map(([v, l]) =>
        `<button data-v="${v}" class="${v === cur ? "on" : ""}">${l}</button>`).join("")}</div>`;
    $$("#edge-shape button").forEach((b) => b.addEventListener("click", () => {
      setEdgeShape(id, b.dataset.v === "auto" && !e.back ? undefined : b.dataset.v);
      $$("#edge-shape button").forEach((x) => x.classList.toggle("on", x === b));
    }));
  }
}

// ------------------------------------------------------------------ matrices
async function checkAt(t) {
  if (state.checkCache.has(t)) return state.checkCache.get(t);
  const info = framesInfo(t);
  const res = await request("/api/check", { graph: state.g.name, prev: info.prev, curr: info.curr });
  state.checkCache.set(t, res);
  return res;
}

function matrixTable(rows, cols, cell, opts = {}) {
  let html = `<table class="mx"><tr><th class="rowh"></th>`;
  for (const j of cols) html += `<th class="${opts.colClass ? opts.colClass(j) : ""}">${j}</th>`;
  if (opts.rowSum) html += `<th>&Phi;&middot;1</th>`;
  html += "</tr>";
  for (const i of rows) {
    html += `<tr><th class="rowh ${opts.rowClass ? opts.rowClass(i) : ""}">${i}</th>`;
    for (const j of cols) {
      const c = cell(i, j);
      html += c ? `<td class="${c.cls}" title="${c.title || ""}">${c.html}</td>` : `<td class="zero">0</td>`;
    }
    if (opts.rowSum) html += `<td class="sum">${opts.rowSum(i)}</td>`;
    html += "</tr>";
  }
  if (opts.colSum) {
    html += `<tr class="sumrow"><th class="rowh">1&#7488;&Phi;</th>`;
    for (const j of cols) html += `<td>${opts.colSum(j)}</td>`;
    if (opts.rowSum) html += "<td></td>";
    html += "</tr>";
  }
  return html + "</table>";
}

async function renderMatrices() {
  if (!state.trace) return;
  const t = state.frame;
  const g = state.g;
  const info = framesInfo(t);
  $("#m-time").textContent = t === 0 ? "t = 0" : `t = ${t}  (S(${t - 1}) → S(${t}))`;
  const label = new Map(g.edges.map((e) => [`${e.from}-${e.to}`, e.k]));
  const curr = new Set(info.curr), left = new Set(info.left), entered = new Set(info.entered);
  const fired = new Set(info.fired.map(([i, j]) => `${i}-${j}`));

  let steps;
  if (state.scope === "all") steps = g.steps.map((s) => s.id);
  else {
    const set = new Set([...info.prev, ...info.curr]);
    g.edges.forEach((e) => { if (curr.has(e.to) || left.has(e.from)) { set.add(e.from); set.add(e.to); } });
    steps = [...set].sort((a, b) => a - b);
  }
  const colClass = (j) => (entered.has(j) ? "entered" : curr.has(j) ? "on" : "");
  const rowClass = (i) => (left.has(i) ? "left" : "");
  const sym = (i, j) => { const k = label.get(`${i}-${j}`); return k ? delta(i, j, k) : null; };
  const sumOf = (pairs) => pairs.length ? pairs.map(([i, j]) => delta(i, j, label.get(`${i}-${j}`))).join(" + ") : "0";

  $("#m-x").innerHTML = matrixTable(steps, steps, (i, j) => {
    const s = sym(i, j); return s && { html: s, cls: "nz" };
  });
  $("#m-e").innerHTML = matrixTable(steps, steps, (i, j) => {
    const s = curr.has(j) && sym(i, j); return s && { html: s, cls: "nz" };
  }, { colClass: (j) => (curr.has(j) ? "on" : "") });
  $("#m-phi").innerHTML = matrixTable(steps, steps, (i, j) =>
    fired.has(`${i}-${j}`) ? { html: sym(i, j), cls: "fired" } : null, {
    colClass, rowClass,
    rowSum: (i) => sumOf(info.fired.filter(([a]) => a === i)),
    colSum: (j) => sumOf(info.fired.filter(([, b]) => b === j)),
  });

  const res = t === 0 ? null : await checkAt(t);
  if (state.frame !== t) return;
  if (!res) { $("#m-rules").innerHTML = `<p class="hint">Initial configuration – no transition yet.</p>`; return; }
  const ok = (b) => (b ? `<span class="ok-t">✓</span>` : `<span class="bad-t">✗</span>`);
  const byStep = (rule, s) => res.violations.some((v) => v.rule === rule && v.step === s);
  let html = `<div class="big-verdict"><span class="pill ${res.valid ? "ok" : "bad"}">${res.valid ? "VALID" : "VIOLATION"}</span>
    <span class="hint">l = {${res.left.join(", ")}} &nbsp; n = {${res.entered.join(", ")}}</span></div>
    <table class="checks"><tr><th>Rule</th><th>Step</th><th>Value</th><th>Allowed</th><th></th></tr>
    <tr><td>R1</td><td>–</td><td>1&#7488;S(t) = ${info.curr.length}</td><td>&ge; 1</td><td>${ok(info.curr.length > 0)}</td></tr>`;
  for (const i of res.left) {
    const s = g.steps[i];
    const groups = Object.entries(s.divergence_groups).map(([k, v]) => `G<sup>${k}</sup> = {${v.join(",")}}`).join(" or ");
    html += `<tr><td>R3</td><td>${i} (left)</td><td>(&Phi;1)<sub>${i}</sub> = ${sumOf(info.fired.filter(([a]) => a === i))}</td>
      <td class="mono">${groups || "sink"}</td><td>${ok(!byStep("R3", i))}</td></tr>`;
  }
  for (const j of res.entered) {
    const s = g.steps[j];
    const conv = s.convergence_groups.map((v) => `{${v.join(",")}}`).join(" or ");
    const r2 = !byStep("R2", j);
    html += `<tr><td>R2/R4</td><td>${j} (entered)</td><td>(1&#7488;&Phi;)<sub>${j}</sub> = ${sumOf(info.fired.filter(([, b]) => b === j))}</td>
      <td class="mono">${conv || "–"}</td><td>${ok(r2 && !byStep("R4", j))}</td></tr>`;
  }
  html += "</table>";
  if (!res.left.length && !res.entered.length && t > 0) html += `<p class="hint">S(t) = S(t&minus;1): nothing fired, &Phi; = 0.</p>`;
  $("#m-rules").innerHTML = html;
}

// ------------------------------------------------------------------ evaluation
function barCell(x) {
  if (x === null || x === undefined) return "–";
  return `<div class="bar-cell"><span>${(100 * x).toFixed(1)}</span><span class="b" style="width:${Math.round(60 * x)}px"></span></div>`;
}

async function loadEvaluation() {
  const s = await request("/api/evaluation");
  const body = $("#eval-body");
  if (s.available === false) {
    body.innerHTML = `<div class="card"><p>No results yet. Run the evaluation (≈30 s for the full profile).</p></div>`;
    return;
  }
  const m = s.meta, b = s.benchmark, cfg = m.config;
  const wlN = cfg.train_scans[cfg.train_scans.length - 1];
  const full = b.detection.phi_full;
  const fa = b.false_alarms;
  const graphs = s.graphs.filter((r) => !r.graph.startsWith("CS_"));
  const timings = graphs.map((r) => r.timing_us).sort((x, y) => x - y);
  const v = Date.now();
  $("#eval-meta").textContent = `Last run ${m.generated.replace("T", " ")} · profile ${cfg.profile} · ${m.graphs} benchmark graphs + ${m.case_studies} case study · ${fmt(m.episodes)} attack episodes · ${fmt(m.transitions)} clean transitions`;

  const kpi = (val, lbl, sub = "") => `<div class="kpi"><div class="v">${val}</div><div class="l">${lbl}</div><div class="s">${sub}</div></div>`;
  let html = `<div class="kpis">
    ${kpi(fmt(fa.phi_full.alarms), "false alarms (&Phi; verifier)", `on ${fmt(fa.phi_full.transitions)} clean transitions · FPR &le; ${fa.phi_full.upper95.toExponential(1)}`)}
    ${kpi(pct(full["family:violating"].rate, 2), "structure-violating attacks detected", `95 % CI ${pct(full["family:violating"].ci[0], 2)} – ${pct(full["family:violating"].ci[1], 2)}`)}
    ${kpi(pct(full["family:violating"].onset_rate), "detected on the first manipulated scan", "zero-latency detection")}
    ${kpi(pct(full["family:stealthy"].rate), "stealthy attacks detected", "legal executions: expected limit")}
    ${kpi(fa[`wl_transition_${wlN}`].per_hour.toFixed(1), "false alarms / hour, learned whitelist", `${fmt(wlN)} training scans · &Phi;: 0, no training`)}
    ${kpi(`${timings[Math.floor(timings.length / 2)].toFixed(2)} µs`, "median verification time", `per transition · max ${timings[timings.length - 1].toFixed(2)} µs`)}
  </div>`;

  const figs = [
    ["fig_detection", "Detection rate per attack class. (a) within the attack horizon, against learned whitelists; (b) on the first manipulated scan, against ablations of the rules."],
    ["fig_false_alarms", "False alarms on normal operation: the training-free verifier versus a transition whitelist learned from increasingly long normal traces."],
    ["fig_rules", "Which rule raises the first alarm, per structure-violating attack class."],
    ["fig_scalability", "(a) Verification time per transition versus graph size. (b) Concurrency reached by the benchmark graphs."],
  ];
  html += `<div class="figs">${figs.map(([f, cap]) => `<div class="card fig">
      <img src="/results/figures/${f}.svg?v=${v}" alt="${f}">
      <div class="cap"><span class="dl"><a href="/results/figures/${f}.svg" download>SVG</a><a href="/results/figures/${f}.pdf" download>PDF</a></span>${cap}</div>
    </div>`).join("")}</div>`;

  // detection table
  const labels = m.detectors;
  html += `<div class="section-title">Detection per attack class (100 benchmark graphs)</div>
    <div class="card table-card"><div class="scroll"><table class="data"><tr>
      <th>Attack</th><th>Family</th><th class="num">Episodes</th><th class="num">&Phi; detected %</th><th class="num">95 % CI</th>
      <th class="num">Onset %</th><th class="num">Median latency</th><th class="num">No div. (onset)</th><th class="num">No conv. (onset)</th>
      <th class="num">Adjacency (onset)</th><th class="num">Trans. WL %</th><th class="num">State WL %</th></tr>`;
  for (const [a, spec] of Object.entries(m.attacks)) {
    const f = full[a];
    const lat = f.latency.median;
    html += `<tr><td title="${esc(spec.description)}">${esc(spec.title)}</td><td><span class="tag ${spec.family}">${spec.family}</span></td>
      <td class="num">${fmt(f.episodes)}</td><td class="num">${barCell(f.rate)}</td>
      <td class="num">${(100 * f.ci[0]).toFixed(1)}–${(100 * f.ci[1]).toFixed(1)}</td>
      <td class="num">${pct(f.onset_rate)}</td><td class="num">${lat === null ? "–" : `${lat} scans`}</td>
      <td class="num">${pct(b.detection.phi_no_divergence[a].onset_rate)}</td>
      <td class="num">${pct(b.detection.phi_no_convergence[a].onset_rate)}</td>
      <td class="num">${pct(b.detection.phi_adjacency[a].onset_rate)}</td>
      <td class="num">${pct(b.detection[`wl_transition_${wlN}`][a].rate)}</td>
      <td class="num">${pct(b.detection[`wl_state_${wlN}`][a].rate)}</td></tr>`;
  }
  html += `</table></div></div>`;

  // false alarms
  html += `<div class="section-title">False alarms on clean traces</div>
    <div class="card table-card"><table class="data"><tr><th>Detector</th><th class="num">Transitions</th><th class="num">Alarms</th>
    <th class="num">FPR</th><th class="num">FPR 95 % upper</th><th class="num">Alarms / hour</th><th class="num">Graphs with alarms</th></tr>
    ${Object.entries(fa).map(([d, x]) => `<tr><td>${esc(labels[d])}</td><td class="num">${fmt(x.transitions)}</td>
      <td class="num">${fmt(x.alarms)}</td><td class="num">${x.rate.toExponential(2)}</td><td class="num">${x.upper95.toExponential(1)}</td>
      <td class="num">${x.per_hour.toFixed(2)}</td><td class="num">${x.graphs_with_alarms} / ${m.graphs}</td></tr>`).join("")}
    </table></div>`;

  // benchmark
  const r = (o) => `${o.min}–${o.max} <span class="hint">(${o.median})</span>`;
  html += `<div class="section-title">Benchmark</div><div class="card table-card"><table class="data"><tr><th>Tier</th>
    <th class="num">Graphs</th><th class="num">Steps</th><th class="num">Edges</th><th class="num">AND div.</th><th class="num">OR div.</th>
    <th class="num">AND conv.</th><th class="num">OR conv.</th><th class="num">Max. active</th><th class="num">Violating detected</th></tr>
    ${["small", "medium", "large", "complex"].filter((t) => b.tiers[t]).map((t) => { const x = b.tiers[t]; return `<tr><td>${t}</td>
      <td class="num">${x.graphs}</td><td class="num">${r(x.steps)}</td><td class="num">${r(x.edges)}</td><td class="num">${r(x.and_divergences)}</td>
      <td class="num">${r(x.or_divergences)}</td><td class="num">${r(x.and_convergences)}</td><td class="num">${r(x.or_convergences)}</td>
      <td class="num">${r(x.max_concurrency)}</td><td class="num">${pct(x.violating_detection, 2)}</td></tr>`; }).join("")}
    </table></div>`;

  // case study
  for (const [name, cs] of Object.entries(s.case_studies)) {
    const d = cs.detection.phi_full;
    html += `<div class="section-title">Case study ${esc(name)} (${cs.stats.steps} steps)</div><div class="card table-card"><table class="data">
      <tr><th>Attack</th><th class="num">Episodes</th><th class="num">&Phi; detected</th><th class="num">Onset</th><th class="num">Trans. WL</th></tr>
      ${Object.entries(m.attacks).map(([a, spec]) => `<tr><td>${esc(spec.title)} <span class="tag ${spec.family}">${spec.family}</span></td>
        <td class="num">${d[a].episodes}</td><td class="num">${pct(d[a].rate)}</td><td class="num">${pct(d[a].onset_rate)}</td>
        <td class="num">${pct(cs.detection[`wl_transition_${wlN}`][a].rate)}</td></tr>`).join("")}
      <tr><td><b>False alarms</b> (clean trace)</td><td class="num">${fmt(cs.false_alarms.phi_full.transitions)}</td>
        <td class="num">${cs.false_alarms.phi_full.alarms}</td><td></td><td class="num">${cs.false_alarms[`wl_transition_${wlN}`].alarms}</td></tr>
    </table></div>`;
  }

  // per graph
  html += `<div class="section-title" style="display:flex;justify-content:space-between;align-items:center">Per graph
      <input class="search" id="graph-search" placeholder="filter graphs…" type="search"></div>
    <div class="card table-card"><div class="scroll" id="per-graph"></div></div>
    <div class="section-title">Downloads</div><div class="card downloads">
      LaTeX tables:<a href="/results/tables/tab_detection.tex" target="_blank">tab_detection.tex</a>
      <a href="/results/tables/tab_false_alarms.tex" target="_blank">tab_false_alarms.tex</a>
      <a href="/results/tables/tab_benchmark.tex" target="_blank">tab_benchmark.tex</a>
      &nbsp;·&nbsp; <a href="/results/REPORT.md" target="_blank">REPORT.md</a>
      <a href="/results/summary.json" target="_blank">summary.json</a></div>`;
  body.innerHTML = html;
  state.perGraph = s.graphs;
  renderPerGraph();
  $("#graph-search").addEventListener("input", renderPerGraph);
}

function renderPerGraph() {
  const q = ($("#graph-search")?.value || "").toLowerCase();
  const cols = [["graph", "Graph"], ["tier", "Tier"], ["steps", "Steps"], ["edges", "Edges"],
    ["and_divergences", "AND"], ["or_divergences", "OR"], ["max_concurrency", "Max active"],
    ["test_changes", "Transitions"], ["false_alarms_full", "&Phi; FA"], ["false_alarms_wl", "WL FA"],
    ["viol", "Violating det."], ["stealth", "Stealthy det."], ["timing_us", "µs"]];
  const rows = state.perGraph.filter((r) => r.graph.toLowerCase().includes(q) || String(r.tier).includes(q))
    .map((r) => ({ ...r, viol: r.violating_detected / r.violating_episodes, stealth: r.stealthy_detected / r.stealthy_episodes }));
  const { key, dir } = state.perGraphSort;
  rows.sort((a, b) => (a[key] > b[key] ? dir : a[key] < b[key] ? -dir : 0));
  const num = (k) => !["graph", "tier"].includes(k);
  $("#per-graph").innerHTML = `<table class="data"><tr>${cols.map(([k, l]) =>
    `<th class="sortable ${num(k) ? "num" : ""}" data-k="${k}">${l}${k === key ? (dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr>
    ${rows.map((r) => `<tr>${cols.map(([k]) => {
      let val = r[k];
      if (k === "viol" || k === "stealth") val = pct(val);
      else if (k === "timing_us") val = val.toFixed(2);
      else if (k === "graph") val = `<a href="#" data-graph="${r.graph}">${r.graph}</a>`;
      return `<td class="${num(k) ? "num" : ""}">${num(k) && typeof val === "number" ? fmt(val) : val}</td>`;
    }).join("")}</tr>`).join("")}</table>`;
  $$("#per-graph th.sortable").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.k;
    state.perGraphSort = { key: k, dir: state.perGraphSort.key === k ? -state.perGraphSort.dir : 1 };
    renderPerGraph();
  }));
  $$("#per-graph a[data-graph]").forEach((a) => a.addEventListener("click", (ev) => {
    ev.preventDefault();
    $("#graph-select").value = a.dataset.graph;
    switchTab("monitor");
    selectGraph(a.dataset.graph);
  }));
}

async function runEvaluation() {
  const profile = $("#eval-profile").value;
  try {
    await request("/api/evaluation/run", { profile });
    $("#btn-eval").disabled = true;
    $("#eval-progress").classList.remove("hidden");
    pollEvaluation();
  } catch (e) { toast(e.message); }
}

async function pollEvaluation() {
  const st = await request("/api/evaluation/status");
  $("#eval-bar").style.width = `${st.total ? (100 * st.done) / st.total : 0}%`;
  $("#eval-status").textContent = st.running
    ? `Evaluating ${st.done} / ${st.total} graphs ${st.current ? "· " + st.current : ""}`
    : st.error ? `Error: ${st.error}` : "Done";
  if (st.running) { state.evalTimer = setTimeout(pollEvaluation, 700); return; }
  $("#btn-eval").disabled = false;
  if (!st.error) {
    setTimeout(() => $("#eval-progress").classList.add("hidden"), 1500);
    loadEvaluation();
  }
}

// ------------------------------------------------------------------ tabs & init
function switchTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `tab-${name}`));
  if (name === "matrices") renderMatrices();
  if (name === "evaluation") loadEvaluation().catch((e) => toast(e.message));
  if (name === "monitor") { drawTimeline(); state.net?.redraw(); }
}

function fillAttackSelect() {
  const sel = $("#in-attack");
  const groups = { violating: "Structure-violating", stealthy: "Structure-preserving (stealthy)" };
  sel.innerHTML = `<option value="">None – normal operation</option>` + Object.entries(groups).map(([fam, label]) =>
    `<optgroup label="${label}">${Object.values(state.meta.attacks).filter((a) => a.family === fam)
      .map((a) => `<option value="${a.name}">${esc(a.title)}</option>`).join("")}</optgroup>`).join("");
  sel.addEventListener("change", () => {
    const a = state.meta.attacks[sel.value];
    $("#attack-desc").textContent = a ? a.description : "";
  });
}

function fillGraphSelect() {
  const sel = $("#graph-select");
  const tiers = {};
  state.graphs.forEach((g) => (tiers[g.tier] ||= []).push(g));
  sel.innerHTML = Object.entries(tiers).map(([tier, gs]) => `<optgroup label="${tier}">${gs.map((g) =>
    `<option value="${g.name}">${g.name} · ${g.steps} steps</option>`).join("")}</optgroup>`).join("");
}

async function init() {
  [state.meta, state.graphs] = await Promise.all([request("/api/meta"), request("/api/graphs")]);
  fillGraphSelect();
  fillAttackSelect();

  $$(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("#graph-select").addEventListener("change", (e) => selectGraph(e.target.value));
  $("#btn-simulate").addEventListener("click", runScenario);
  $("#in-csv").addEventListener("change", (e) => e.target.files[0] && importCsv(e.target.files[0]));
  $("#btn-play").addEventListener("click", play);
  $("#btn-back").addEventListener("click", () => { stop(); seek(state.frame - 1); });
  $("#btn-fwd").addEventListener("click", () => { stop(); seek(state.frame + 1); });
  $("#btn-next-alarm").addEventListener("click", () => {
    stop();
    const next = state.trace.alarms.findIndex((a, t) => a && t > state.frame);
    if (next >= 0) seek(next); else toast("No further alarm in this trace.");
  });
  $("#speed").addEventListener("change", () => { if (state.timer) { stop(); play(); } });
  $("#scrub").addEventListener("input", (e) => { stop(); seek(Number(e.target.value)); });
  $("#btn-relayout").addEventListener("click", relayout);
  $("#btn-fit").addEventListener("click", () => state.net.fit({ animation: true }));
  $("#chk-labels").addEventListener("change", toggleEdgeLabels);
  $("#all-edges").addEventListener("change", (e) => {
    if (!e.target.value) return;
    state.g.edges.forEach((x) => setEdgeShape(`${x.from}-${x.to}`, e.target.value === "auto" ? undefined : e.target.value));
  });
  $("#m-back").addEventListener("click", () => seek(state.frame - 1));
  $("#m-fwd").addEventListener("click", () => seek(state.frame + 1));
  $$("#m-scope button").forEach((b) => b.addEventListener("click", () => {
    state.scope = b.dataset.scope;
    $$("#m-scope button").forEach((x) => x.classList.toggle("on", x === b));
    renderMatrices();
  }));
  $("#btn-eval").addEventListener("click", runEvaluation);
  window.addEventListener("resize", () => state.trace && drawTimeline());
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === " ") { e.preventDefault(); play(); }
    if (e.key === "ArrowRight") { stop(); seek(state.frame + 1); }
    if (e.key === "ArrowLeft") { stop(); seek(state.frame - 1); }
  });

  const status = await request("/api/evaluation/status");
  if (status.running) { $("#btn-eval").disabled = true; $("#eval-progress").classList.remove("hidden"); pollEvaluation(); }

  // deep link: #graph=G090_complex&attack=or_violation&seed=3&tab=matrices&frame=alarm
  const params = new URLSearchParams(location.hash.slice(1));
  const first = params.get("graph") || state.graphs[0].name;
  if (params.get("attack")) $("#in-attack").value = params.get("attack");
  if (params.get("seed")) $("#in-seed").value = params.get("seed");
  if (params.get("scans")) $("#in-scans").value = params.get("scans");
  $("#graph-select").value = first;
  await selectGraph(first);
  const frame = params.get("frame");
  if (frame === "alarm") { const a = state.trace.alarms.findIndex(Boolean); if (a >= 0) seek(a); }
  else if (frame) seek(Number(frame));
  if (params.get("tab")) switchTab(params.get("tab"));
}

init().catch((e) => toast(`Initialisation failed: ${e.message}`, 10000));

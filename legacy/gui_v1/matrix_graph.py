"""Interactive graph view of the SFC matrix defined in control_logic.py.

Each non-zero cell M[i, j] holds a symbol delta_i_j^k: an edge i -> j of type k.
For a source node i with several outgoing edges:
  * all k equal      -> AND divergence (node coloured red)
  * different k      -> OR divergence  (default colour)

Usage:
    python matrix_graph.py            # builds the graph and serves it on http://127.0.0.1:8765
    python matrix_graph.py --no-serve # only write graph.html (no live verification)

The local server runs control_logic.control_logic_verifier_ones_M_ones on S(t) for each
time step shown in the page.
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import warnings
import webbrowser
from collections import defaultdict
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))  # pyvis installed with pip --target
sys.path.insert(0, os.path.dirname(HERE))         # src/, for control_logic

from pyvis.network import Network

SYMBOL_RE = re.compile(r"[A-Za-z]+_(\d+)_(\d+)\^(\d+)")

AND_COLOR = "#e53935"
OTHER_COLOR = "#97c2fc"


def matrix_to_tuples(M):
    """Extract (i, j, k) from every symbol delta_i_j^k found in the matrix."""
    tuples = set()
    for sym in M.free_symbols:
        m = SYMBOL_RE.fullmatch(str(sym))
        if m:
            tuples.add(tuple(int(g) for g in m.groups()))
    return sorted(tuples)


def classify_divergences(tuples):
    """Return {source: 'AND' | 'OR'} for sources with more than one outgoing edge."""
    out = defaultdict(list)
    for i, j, k in tuples:
        out[i].append((j, k))
    kinds = {}
    for i, edges in out.items():
        if len({j for j, _ in edges}) < 2:
            continue
        kinds[i] = "AND" if len({k for _, k in edges}) == 1 else "OR"
    return kinds


def build_graph(tuples, height="100vh"):
    kinds = classify_divergences(tuples)
    net = Network(height=height, width="100%", directed=True,
                  cdn_resources="in_line", bgcolor="#ffffff", font_color="#222222")
    nodes = sorted({n for i, j, _ in tuples for n in (i, j)})
    for n in nodes:
        kind = kinds.get(n)
        net.add_node(
            n, label=str(n), shape="circle", size=22,
            color=AND_COLOR if kind == "AND" else OTHER_COLOR,
            font={"color": "#ffffff" if kind == "AND" else "#222222"},
            title=f"Node {n}" + (f" - {kind} divergence" if kind else ""),
        )
    for i, j, k in tuples:
        net.add_edge(i, j, id=f"{i}-{j}-{k}", label=str(k), title=f"δ({i},{j})^{k}",
                     arrows="to", smooth=False, font={"size": 11, "align": "middle"})
    net.set_options("""{
      "physics": {"solver": "forceAtlas2Based",
                  "forceAtlas2Based": {"gravitationalConstant": -80, "springLength": 110}},
      "interaction": {"dragNodes": true, "dragView": true, "zoomView": true,
                      "hover": true, "navigationButtons": true}
    }""")
    return net


LEGEND = """
<div style="position:fixed;top:10px;left:10px;z-index:10;background:#fffd;padding:8px 12px;
            border:1px solid #ccc;border-radius:6px;font:13px sans-serif">
  <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%%;background:%s"></span>
       AND divergence</div>
  <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%%;background:%s"></span>
       other nodes (incl. OR divergence)</div>
  <div style="color:#666;margin-top:4px">Edge label = k. Drag a node: it stays where you drop it.<br>
       Double-click a node to unpin it.</div>
  <label><input type="checkbox" id="physics-toggle" checked> physics (auto layout)</label>
</div>
<div id="edge-panel" style="position:fixed;top:10px;right:10px;bottom:10px;width:230px;z-index:10;
     background:#fffe;border:1px solid #ccc;border-radius:6px;font:13px sans-serif;
     display:flex;flex-direction:column">
  <div style="padding:8px 10px;border-bottom:1px solid #ddd">
    <b>Edge shape</b><br>
    All: <select id="all-shape"><option value="">-</option></select>
    <div style="color:#666;margin-top:4px">Click an edge to find it in the list.</div>
  </div>
  <div id="edge-list" style="overflow-y:auto;padding:4px 10px"></div>
</div>
<script>
var SHAPES = {straight: false,
              "curved (left)": {enabled: true, type: "curvedCW", roundness: 0.35},
              "curved (right)": {enabled: true, type: "curvedCCW", roundness: 0.35}};
function shapeOptions(sel) {
  Object.keys(SHAPES).forEach(function (name) {
    var o = document.createElement("option"); o.value = name; o.textContent = name; sel.appendChild(o);
  });
}
function setShape(id, name) { edges.update({id: id, smooth: SHAPES[name]}); }
window.addEventListener("load", function () {
  var list = document.getElementById("edge-list"), allSel = document.getElementById("all-shape");
  shapeOptions(allSel); allSel.value = "";
  var selects = {};
  edges.get().sort(function (a, b) { return a.from - b.from || a.to - b.to; }).forEach(function (e) {
    var row = document.createElement("div");
    row.id = "row-" + e.id; row.style.cssText = "display:flex;justify-content:space-between;margin:3px 0;padding:1px 2px";
    row.innerHTML = "<span>" + e.from + " → " + e.to + " (k=" + e.label + ")</span>";
    var sel = document.createElement("select"); shapeOptions(sel); sel.value = "straight";
    sel.addEventListener("change", function () { setShape(e.id, sel.value); });
    selects[e.id] = sel; row.appendChild(sel); list.appendChild(row);
  });
  allSel.addEventListener("change", function () {
    if (!allSel.value) return;
    Object.keys(selects).forEach(function (id) { selects[id].value = allSel.value; setShape(id, allSel.value); });
  });
  network.on("selectEdge", function (p) {
    document.querySelectorAll("#edge-list div").forEach(function (r) { r.style.background = ""; });
    if (p.nodes.length === 0 && p.edges.length) {
      var row = document.getElementById("row-" + p.edges[0]);
      if (row) { row.style.background = "#ffe08a"; row.scrollIntoView({block: "nearest"}); }
    }
  });
  network.on("dragEnd", function (p) {
    p.nodes.forEach(function (id) {
      var pos = network.getPositions([id])[id];
      nodes.update({id: id, x: pos.x, y: pos.y, fixed: {x: true, y: true}});
    });
  });
  network.on("doubleClick", function (p) {
    p.nodes.forEach(function (id) { nodes.update({id: id, fixed: false}); });
  });
  document.getElementById("physics-toggle").addEventListener("change", function (e) {
    network.setOptions({physics: {enabled: e.target.checked}});
  });
});
</script>""" % (AND_COLOR, OTHER_COLOR)


PLAYER = """
<div id="player" style="position:fixed;bottom:12px;left:50%;transform:translateX(-50%);z-index:10;
     background:#fffe;border:1px solid #ccc;border-radius:6px;padding:8px 12px;font:13px sans-serif;
     width:560px;max-width:calc(100vw - 520px);min-width:360px">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <button id="run-btn">Run</button><button id="reset-btn">Reset</button>
    Speed <select id="speed"><option>1</option><option>2</option><option>5</option>
           <option>10</option><option>50</option></select>x
    <input type="file" id="csv-file" accept=".csv">
    <b id="time-label" style="margin-left:auto">t = 0 s</b>
  </div>
  <input type="range" id="scrub" min="0" max="0" value="0" style="width:100%;margin-top:6px">
  <div id="csv-info" style="color:#666">No CSV loaded: choose one (id, timestamp, 0, 1, ...).</div>
</div>
<script>
var PRELOAD_CSV = __CSV__;
var GREEN = "#2e7d32", GREY = "#b0b0b0", BASE = {}, rows = [], frame = -1, timer = null, visited = {};
function ringRenderer(args) {
  var ctx = args.ctx, x = args.x, y = args.y, r = 30;
  return {
    drawNode: function () {
      ctx.save();
      ctx.fillStyle = GREEN; ctx.strokeStyle = "#1b5e20"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, r - 6, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = "#ffffff"; ctx.font = "bold 16px sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(args.label), x, y);
      ctx.restore();
    },
    nodeDimensions: {width: 2 * r, height: 2 * r}
  };
}
function parseCSV(text) {
  var lines = text.trim().split(/\\r?\\n/), head = lines[0].split(",");
  var tcol = head.indexOf("timestamp"), out = [], t0 = null;
  for (var n = 1; n < lines.length; n++) {
    var c = lines[n].split(","), t = new Date(c[tcol].replace(" ", "T")).getTime() / 1000;
    if (t0 === null) t0 = t;
    var active = [];
    head.forEach(function (h, i) { if (/^\\d+$/.test(h) && c[i].trim().toLowerCase() === "true") active.push(Number(h)); });
    out.push({t: t - t0, active: active});
  }
  return out;
}
function loadCSV(text, name) {
  rows = parseCSV(text); frame = -1;
  var sc = document.getElementById("scrub"); sc.max = Math.max(0, rows.length - 1); sc.value = 0;
  document.getElementById("csv-info").textContent = name + ": " + rows.length + " rows, " +
      (rows.length ? rows[rows.length - 1].t : 0) + " s";
  stop(); show(0);
}
function paint(f) {
  var act = {}; rows[f].active.forEach(function (n) { act[n] = true; });
  if (act[0] && !(frame >= 0 && rows[frame].active.indexOf(0) >= 0)) visited = {};
  var upd = [];
  nodes.getIds().forEach(function (id) {
    var u = {id: id};
    if (act[id]) {
      visited[id] = true;
      u.shape = "custom"; u.ctxRenderer = ringRenderer; u.size = 34;
    } else {
      u.shape = "circle"; u.size = 22;
      var grey = visited[id];
      u.color = grey ? GREY : BASE[id].color; u.font = grey ? {color: "#222222"} : BASE[id].font;
    }
    upd.push(u);
  });
  nodes.update(upd);
}
function show(f) {
  if (!rows.length) return;
  f = Math.max(0, Math.min(rows.length - 1, f));
  if (f < frame) { visited = {}; frame = -1; }          // going back: replay from the start
  for (var i = frame + 1; i <= f; i++) { paint(i); frame = i; }
  document.getElementById("scrub").value = f;
  document.getElementById("time-label").textContent = "t = " + rows[f].t + " s";
  if (window.updateInfo) updateInfo();
}
function stop() { if (timer) clearInterval(timer); timer = null; document.getElementById("run-btn").textContent = "Run"; }
function run() {
  if (timer) { stop(); return; }
  if (!rows.length) return;
  if (frame >= rows.length - 1) show(0);
  document.getElementById("run-btn").textContent = "Pause";
  var started = performance.now(), base = rows[Math.max(frame, 0)].t,
      speed = Number(document.getElementById("speed").value);
  timer = setInterval(function () {
    var el = base + (performance.now() - started) / 1000 * speed, f = frame;
    while (f + 1 < rows.length && rows[f + 1].t <= el) f++;
    if (f > frame) show(f);
    if (f >= rows.length - 1) stop();
  }, 50);
}
window.addEventListener("load", function () {
  nodes.get().forEach(function (n) { BASE[n.id] = {color: n.color, font: n.font}; });
  document.getElementById("run-btn").onclick = run;
  document.getElementById("reset-btn").onclick = function () { stop(); show(0); };
  document.getElementById("speed").onchange = function () { if (timer) { stop(); run(); } };
  document.getElementById("scrub").oninput = function (e) { stop(); show(Number(e.target.value)); };
  document.getElementById("csv-file").onchange = function (e) {
    var f = e.target.files[0]; if (!f) return;
    var r = new FileReader(); r.onload = function () { loadCSV(r.result, f.name); }; r.readAsText(f);
  };
  if (PRELOAD_CSV) loadCSV(PRELOAD_CSV.text, PRELOAD_CSV.name);
});
</script>"""


TABS = """
<div id="tabbar" style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:30;
     display:flex;gap:4px;font:14px sans-serif">
  <button id="tab-graph" class="tab" style="font-weight:bold">Graph</button>
  <button id="tab-info" class="tab">Matrix &amp; predecessors</button>
</div>
<div id="info-view" style="display:none;position:fixed;inset:0;z-index:20;background:#fff;overflow:auto;
     padding:56px 20px 20px;font:14px sans-serif">
  <div id="verify-box" style="margin-bottom:14px;padding:10px 12px;border:1px solid #ccc;border-radius:6px;
       display:inline-block;min-width:420px"></div>
  <div id="pred-summary" style="margin-bottom:14px"></div>
  <div style="margin-bottom:6px;color:#555">Matrix M (row i = source, column j = destination; cell = &delta;<sup>k</sup>).
    Green header: enabled state. Yellow cell: a predecessor of an enabled state.</div>
  <table id="matrix" style="border-collapse:collapse;font-size:12px"></table>
  <div style="margin:22px 0 6px;color:#555"><b>Constraint matrix E(t) = M &middot; diag(S(t))</b>:
    column j of M is kept if state j is enabled at t, otherwise it is zeroed.</div>
  <div id="s-vector" style="margin-bottom:6px;font-family:monospace"></div>
  <table id="emat" style="border-collapse:collapse;font-size:12px"></table>
  <h3 style="margin:28px 0 6px">Common-index verification (S<sub>1</sub> = S(t&minus;1), S<sub>2</sub> = S(t))</h3>
  <div id="common-box" style="margin-bottom:10px;padding:10px 12px;border:1px solid #ccc;border-radius:6px;
       display:inline-block;min-width:420px"></div>
  <div style="margin-bottom:6px;color:#555">A = E<sub>1</sub><sup>T</sup> + E<sub>2</sub>
    (E<sub>1</sub> = M&middot;diag(S(t&minus;1)), E<sub>2</sub> = M&middot;diag(S(t)); the 1.0 coefficients are omitted).
    Last column: func(A&middot;1) = apply_check_common_i_matrix(A&middot;1), hover a value to see the row sum A&middot;1.</div>
  <table id="amat" style="border-collapse:collapse;font-size:11px"></table>
  <h3 style="margin:28px 0 6px">Transition verification with &Phi; (S<sub>1</sub> = S(t&minus;1), S<sub>2</sub> = S(t))</h3>
  <div id="trans-box" style="margin-bottom:10px;padding:10px 12px;border:1px solid #ccc;border-radius:6px;
       display:inline-block;min-width:420px"></div>
  <div style="margin-bottom:6px;color:#555">&Phi; = diag(l)&middot;M&middot;diag(n), l = S<sub>1</sub>&#8728;(1&minus;S<sub>2</sub>) (left states, orange rows),
    n = S<sub>2</sub>&#8728;(1&minus;S<sub>1</sub>) (entered states, green columns). Last column: &Phi;&middot;1, last row: 1<sup>T</sup>&middot;&Phi;.</div>
  <table id="phimat" style="border-collapse:collapse;font-size:11px"></table>
</div>
<script>
var GRAPH_EDGES = __EDGES__, N_STATES = __N__;
function showTab(name) {
  var info = name === "info";
  document.getElementById("info-view").style.display = info ? "block" : "none";
  document.getElementById("tab-graph").style.fontWeight = info ? "normal" : "bold";
  document.getElementById("tab-info").style.fontWeight = info ? "bold" : "normal";
}
function currentEnabled() { return (frame >= 0 && rows.length) ? rows[frame].active : []; }
var verifySeq = 0;
function requestVerify(en) {
  var box = document.getElementById("verify-box"), seq = ++verifySeq;
  if (frame < 0 || !rows.length) { box.innerHTML = "<b>Verification</b>: no CSV / not started"; return; }
  var S = [], Sp = null, prev = frame > 0 ? rows[frame - 1].active : null;
  for (var j = 0; j < N_STATES; j++) S.push(en.indexOf(j) >= 0 ? 1 : 0);
  if (prev) { Sp = []; for (j = 0; j < N_STATES; j++) Sp.push(prev.indexOf(j) >= 0 ? 1 : 0); }
  fetch("/verify", {method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({S: S, S_prev: Sp})})
    .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then(function (v) {
      if (seq !== verifySeq) return;                   // a newer time step was requested
      box.style.background = v.valid ? "#e8f5e9" : "#ffebee";
      box.style.borderColor = v.valid ? "#2e7d32" : "#c62828";
      renderCommon(v.common);
      renderTransition(v.transition);
      box.innerHTML = "<div><b>control_logic_verifier_ones_M_ones(M, S(t))</b> at t = " + rows[frame].t + " s</div>" +
        "<div style='font-size:20px;margin:4px 0;color:" + (v.valid ? "#2e7d32" : "#c62828") + "'><b>" +
        (v.valid ? "VALID" : "VIOLATION") + "</b></div>" +
        "<div>int(ones_M_ones(E, mode=&quot;not_xor&quot;)) = <b>" + v.not_xor + "</b></div>" +
        "<div>get_incident_neighbors(M, enabled) = <b>" + v.neighbors + "</b></div>";
    })
    .catch(function (e) {
      if (seq !== verifySeq) return;
      box.style.background = "#fff8e1"; box.style.borderColor = "#f9a825";
      box.innerHTML = "<b>Verification unavailable</b> (" + e.message + "). Open the page through " +
        "<code>python3 matrix_graph.py</code> (http://127.0.0.1:8765), not as a file.";
      renderCommon(null);
      renderTransition(null);
    });
}
function prettyExpr(s) {
  if (s === "0") return "";
  return s.replace(/1\.0\*/g, "").replace(/ \+ /g, "+")
          .replace(/[A-Za-z]+_(\d+)_(\d+)\^(\d+)/g, "&delta;<sub>$1,$2</sub><sup>$3</sup>");
}
function renderCommon(c) {
  var box = document.getElementById("common-box"), tbl = document.getElementById("amat");
  if (!c) {
    box.style.background = ""; box.style.borderColor = "#ccc";
    box.innerHTML = frame === 0 ? "<b>control_logic_verifier_common_index</b>: n/a at the first time step (no S(t&minus;1))"
                                : "<b>control_logic_verifier_common_index</b>: unavailable";
    tbl.innerHTML = ""; return;
  }
  box.style.background = c.valid ? "#e8f5e9" : "#ffebee";
  box.style.borderColor = c.valid ? "#2e7d32" : "#c62828";
  box.innerHTML = "<div><b>control_logic_verifier_common_index(M, S(t&minus;1), S(t))</b></div>" +
    "<div style='font-size:20px;margin:4px 0;color:" + (c.valid ? "#2e7d32" : "#c62828") + "'><b>" +
    (c.valid ? "True" : "False") + "</b></div>" +
    "<div>ones_M_ones(E_1.T + E_2, mode=&quot;common_index&quot;) = <b>" + c.scalar + "</b></div>" +
    "<div>get_incident_neighbors(M, S(t)) = <b>" + c.neighbors + "</b></div>" +
    "<div>E_1 == E_2: <b>" + (c.e_equal ? "True" : "False") + "</b></div>";
  var html = "<tr><th></th>", i, j;
  for (j = 0; j < N_STATES; j++) html += "<th style='border:1px solid #bbb;padding:2px 4px;background:#eee'>" + j + "</th>";
  html += "<th style='border:1px solid #bbb;border-left:3px solid #555;padding:2px 6px;background:#fff3cd'>func(A&middot;1)</th></tr>";
  for (i = 0; i < N_STATES; i++) {
    html += "<tr><th style='border:1px solid #bbb;padding:2px 4px;background:#eee'>" + i + "</th>";
    for (j = 0; j < N_STATES; j++) {
      var e = prettyExpr(c.matrix[i][j]);
      html += "<td style='border:1px solid #ddd;padding:2px 4px;text-align:center;white-space:nowrap" +
              (e ? ";background:#e3f2fd" : "") + "'>" + e + "</td>";
    }
    var on = c.vector[i] !== 0;
    html += "<td title='A&middot;1 row " + i + " = " + c.rowsum[i] + "' style='border:1px solid #ddd;border-left:3px solid #555;" +
            "text-align:center;font-weight:bold;" + (on ? "background:#c8e6c9" : "") + "'>" + c.vector[i] + "</td></tr>";
  }
  tbl.innerHTML = html;
}
function renderTransition(c) {
  var box = document.getElementById("trans-box"), tbl = document.getElementById("phimat");
  if (!c) {
    box.style.background = ""; box.style.borderColor = "#ccc";
    box.innerHTML = frame === 0 ? "<b>control_logic_verifier_transition</b>: n/a at the first time step (no S(t&minus;1))"
                                : "<b>control_logic_verifier_transition</b>: unavailable";
    tbl.innerHTML = ""; return;
  }
  var idx = function (v) { var o = []; v.forEach(function (x, i) { if (x) o.push(i); }); return o.length ? o.join(", ") : "&empty;"; };
  box.style.background = c.valid ? "#e8f5e9" : "#ffebee";
  box.style.borderColor = c.valid ? "#2e7d32" : "#c62828";
  box.innerHTML = "<div><b>control_logic_verifier_transition(M, S(t&minus;1), S(t), convergence)</b></div>" +
    "<div style='font-size:20px;margin:4px 0;color:" + (c.valid ? "#2e7d32" : "#c62828") + "'><b>" +
    (c.valid ? "True" : "False") + "</b></div>" +
    "<div>left states l: <b>" + idx(c.left) + "</b> &nbsp; entered states n: <b>" + idx(c.entered) + "</b></div>" +
    (c.errors.length ? "<ul style='margin:4px 0;color:#c62828'><li>" + c.errors.join("</li><li>") + "</li></ul>"
                     : "<div style='color:#555'>(1) no deadlock, (2) justification, (3) divergence, (4) convergence: all satisfied</div>");
  var cellS = "border:1px solid #ddd;padding:2px 4px;text-align:center;white-space:nowrap";
  var html = "<tr><th></th>", i, j;
  for (j = 0; j < N_STATES; j++)
    html += "<th style='border:1px solid #bbb;padding:2px 4px;" + (c.entered[j] ? "background:#2e7d32;color:#fff" : "background:#eee") + "'>" + j + "</th>";
  html += "<th style='border:1px solid #bbb;border-left:3px solid #555;padding:2px 6px;background:#fff3cd'>&Phi;&middot;1</th></tr>";
  for (i = 0; i < N_STATES; i++) {
    html += "<tr><th style='border:1px solid #bbb;padding:2px 4px;" + (c.left[i] ? "background:#ef6c00;color:#fff" : "background:#eee") + "'>" + i + "</th>";
    for (j = 0; j < N_STATES; j++) {
      var e = prettyExpr(c.phi[i][j]);
      html += "<td style='" + cellS + (e ? ";background:#e3f2fd" : "") + "'>" + e + "</td>";
    }
    html += "<td style='" + cellS + ";border-left:3px solid #555'>" + prettyExpr(c.row_sums[i]) + "</td></tr>";
  }
  html += "<tr><th style='border:1px solid #bbb;border-top:3px solid #555;padding:2px 4px;background:#fff3cd'>1<sup>T</sup>&Phi;</th>";
  for (j = 0; j < N_STATES; j++) html += "<td style='" + cellS + ";border-top:3px solid #555'>" + prettyExpr(c.col_sums[j]) + "</td>";
  html += "<td style='border-top:3px solid #555;border-left:3px solid #555'></td></tr>";
  tbl.innerHTML = html;
}
function updateInfo() {
  var en = currentEnabled(), enSet = {}, preds = {}, uniq = {}, lines = [];
  en.forEach(function (v) { enSet[v] = true; preds[v] = []; });
  GRAPH_EDGES.forEach(function (e) {
    if (enSet[e[1]]) { preds[e[1]].push(e[0]); uniq[e[0]] = true; }
  });
  en.forEach(function (v) {
    lines.push("<li>state " + v + ": " + (preds[v].length ? preds[v].join(", ") : "none") + "</li>");
  });
  var t = (frame >= 0 && rows.length) ? rows[frame].t + " s" : "- (no CSV / not started)";
  document.getElementById("pred-summary").innerHTML =
    "<div><b>t = " + t + "</b></div>" +
    "<div>Enabled states: <b>" + (en.length ? en.join(", ") : "none") + "</b></div>" +
    "<div style='font-size:18px;margin:6px 0'>Total predecessor states of the enabled states "
    + "(unique): <b>" + Object.keys(uniq).length + "</b></div>" +
    "<ul style='margin:4px 0'>" + lines.join("") + "</ul>";
  buildE(enSet);
  requestVerify(en);
  var cells = document.querySelectorAll("#matrix td, #matrix th");
  cells.forEach(function (c) {
    var i = c.getAttribute("data-i"), j = c.getAttribute("data-j");
    if (c.tagName === "TH") {
      var isCol = i === null, id = isCol ? j : i;
      c.style.background = (isCol && enSet[id]) ? "#2e7d32" : "#eee";
      c.style.color = (isCol && enSet[id]) ? "#fff" : "#222";
    } else if (c.textContent) {
      c.style.background = enSet[j] ? "#ffe08a" : "";
    }
  });
}
function buildE(enSet) {
  var cell = {};
  GRAPH_EDGES.forEach(function (e) { cell[e[0] + "," + e[1]] = e[2]; });
  var html = "<tr><th></th>", i, j;
  for (j = 0; j < N_STATES; j++)
    html += "<th style='border:1px solid #bbb;padding:2px 5px;background:" + (enSet[j] ? "#2e7d32;color:#fff" : "#eee") + "'>" + j + "</th>";
  html += "</tr>";
  for (i = 0; i < N_STATES; i++) {
    html += "<tr><th style='border:1px solid #bbb;padding:2px 5px;background:#eee'>" + i + "</th>";
    for (j = 0; j < N_STATES; j++) {
      var k = enSet[j] ? cell[i + "," + j] : undefined;
      html += "<td style='border:1px solid #ddd;padding:2px 5px;text-align:center;min-width:22px" +
              (k ? ";background:#c8e6c9' title='delta(" + i + "," + j + ")^" + k + "'>&delta;<sup>" + k + "</sup>" : "'>") + "</td>";
    }
    html += "</tr>";
  }
  document.getElementById("emat").innerHTML = html;
  var S = []; for (j = 0; j < N_STATES; j++) S.push(enSet[j] ? 1 : 0);
  document.getElementById("s-vector").textContent = "S(t) = [" + S.join(" ") + "]";
}
window.addEventListener("load", function () {
  var cell = {};
  GRAPH_EDGES.forEach(function (e) { cell[e[0] + "," + e[1]] = e[2]; });
  var html = "<tr><th></th>";
  for (var j = 0; j < N_STATES; j++) html += "<th data-j='" + j + "' style='border:1px solid #bbb;padding:2px 5px'>" + j + "</th>";
  html += "</tr>";
  for (var i = 0; i < N_STATES; i++) {
    html += "<tr><th data-i='" + i + "' style='border:1px solid #bbb;padding:2px 5px'>" + i + "</th>";
    for (var j = 0; j < N_STATES; j++) {
      var k = cell[i + "," + j];
      html += "<td data-i='" + i + "' data-j='" + j + "' style='border:1px solid #ddd;padding:2px 5px;text-align:center;min-width:22px'" +
              (k ? " title='delta(" + i + "," + j + ")^" + k + "'>&delta;<sup>" + k + "</sup>" : ">") + "</td>";
    }
    html += "</tr>";
  }
  document.getElementById("matrix").innerHTML = html;
  document.getElementById("tab-graph").onclick = function () { showTab("graph"); };
  document.getElementById("tab-info").onclick = function () { updateInfo(); showTab("info"); };
  updateInfo();
});
</script>"""


def load_control_logic():
    # control_logic runs a script on import (prints, reads "data.csv" relative to the cwd),
    # so import it from src/ with stdout silenced.
    cwd = os.getcwd()
    os.chdir(os.path.dirname(HERE))
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import control_logic
    finally:
        os.chdir(cwd)
    return control_logic


def make_verifier(cl):
    import numpy as np
    import sympy as sp

    @lru_cache(maxsize=4096)
    def common(S_prev, S):
        S1, S2 = np.array(S_prev, dtype=int), np.array(S, dtype=int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid = cl.control_logic_verifier_common_index(cl.M, S1, S2)
            E_1 = cl.M * sp.diag(*[float(s) for s in S1])
            E_2 = cl.M * sp.diag(*[float(s) for s in S2])
            A = E_1.T + E_2
            rowsum = A * sp.ones(A.rows, 1)
            vector = cl.apply_check_common_i_matrix(rowsum)
            scalar = cl.ones_M_ones(A, mode="common_index")
        return {
            "valid": bool(valid),
            "matrix": [[str(A[i, j]) for j in range(A.cols)] for i in range(A.rows)],
            "rowsum": [str(rowsum[i]) for i in range(A.rows)],
            "vector": [int(vector[i]) for i in range(A.rows)],
            "scalar": int(scalar),
            "neighbors": int(cl.get_incident_neighbors(cl.M, np.where(S2 == 1)[0])),
            "e_equal": bool(E_1 == E_2),
        }

    @lru_cache(maxsize=4096)
    def transition(S_prev, S):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid, d = cl.control_logic_verifier_transition(
                cl.M, np.array(S_prev), np.array(S), getattr(cl, "convergence", None), details=True)
        n = cl.M.rows
        return {
            "valid": bool(valid),
            "left": [int(x) for x in d["left"]],
            "entered": [int(x) for x in d["entered"]],
            "phi": [[str(d["Phi"][i, j]) for j in range(n)] for i in range(n)],
            "row_sums": [str(d["row_sums"][i]) for i in range(n)],
            "col_sums": [str(d["col_sums"][j]) for j in range(n)],
            "errors": d["errors"],
        }

    def verify(S, S_prev=None):
        out = dict(single(S))
        out["common"] = common(S_prev, S) if S_prev is not None else None
        out["transition"] = transition(S_prev, S) if S_prev is not None else None
        return out

    @lru_cache(maxsize=4096)
    def single(S):
        S = np.array(S, dtype=int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid = cl.control_logic_verifier_ones_M_ones(cl.M, S)
            E = cl.M * sp.diag(*[float(s) for s in S])
            not_xor = int(cl.ones_M_ones(E, mode="not_xor"))
        neighbors = cl.get_incident_neighbors(cl.M, np.where(S == 1)[0])
        return {"valid": bool(valid), "not_xor": not_xor, "neighbors": int(neighbors)}
    return verify


def serve(directory, port, verify, page):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def do_POST(self):
            if self.path != "/verify":
                return self.send_error(404)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prev = body.get("S_prev")
                out = json.dumps(verify(tuple(int(x) for x in body["S"]),
                                        tuple(int(x) for x in prev) if prev else None)).encode()
            except Exception as e:
                return self.send_error(400, str(e))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/{page}"
    print("Serving", url, "(Ctrl+C to stop)")
    return httpd, url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "graph.html"))
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(HERE), "data.csv"),
                    help="state activation CSV to preload (can also be chosen in the page)")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-serve", action="store_true", help="only write the HTML file")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    cl = load_control_logic()
    tuples = matrix_to_tuples(cl.M)
    preload = None
    if os.path.exists(args.csv):
        with open(args.csv, encoding="utf-8") as f:
            preload = {"name": os.path.basename(args.csv), "text": f.read()}
    player_html = PLAYER.replace("__CSV__", json.dumps(preload).replace("</", "<\\/"))
    tabs_html = (TABS.replace("__EDGES__", json.dumps([list(t) for t in tuples]))
                     .replace("__N__", str(max(n for t in tuples for n in t[:2]) + 1)))
    net = build_graph(tuples)
    net.write_html(args.out, notebook=False)
    with open(args.out, encoding="utf-8") as f:
        html = f.read()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html.replace("<body>", "<body>" + LEGEND + player_html + tabs_html, 1))

    kinds = classify_divergences(tuples)
    print(f"{len(tuples)} edges; AND: {[n for n, k in kinds.items() if k == 'AND']}; "
          f"OR: {[n for n, k in kinds.items() if k == 'OR']}")
    print("Wrote", args.out)
    if args.no_serve:
        if not args.no_open:
            webbrowser.open("file://" + os.path.abspath(args.out))
        return
    out = os.path.abspath(args.out)
    httpd, url = serve(os.path.dirname(out), args.port, make_verifier(cl), os.path.basename(out))
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

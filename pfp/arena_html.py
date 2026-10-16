"""
Self-contained HTML5/Canvas Tron arena player (Piece 7 renderer).

A dumb renderer on the append-only event stream: the full stream is embedded
as JSON and every frame is a pure function of "events up to cursor". The same
file is the live-UI replay, the post-run review, and the demo-video source.

Safety: all model-generated text reaches the page via `textContent` only; the
embedded JSON escapes `<`, `>`, `&`, U+2028/U+2029 so no string in the stream
can close the <script> tag or inject markup.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, List


def _safe_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def build_arena_html(events: List[Dict[str, Any]], problem: str, ascii_snapshot: str) -> str:
    values = {
        "N": str(len(events)),
        "ASCII": html.escape(ascii_snapshot),
        "PROBLEM_JSON": _safe_json(problem or ""),
        "EVENTS_JSON": _safe_json(events),
    }
    # Single pass over the template: substituted text is never re-scanned for placeholders
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], _TEMPLATE)


_PLACEHOLDER = re.compile(r"__(N|ASCII|PROBLEM_JSON|EVENTS_JSON)__")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PFP Darwinism — Tron Arena</title>
<style>
  :root { --bg:#03050b; --panel:rgba(7,13,26,.93); --cyan:#00f0ff; --green:#00ff9d; --red:#ff3366;
          --amber:#ffd479; --dim:#5b7a99; --text:#d8f3ff; }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; color:var(--text);
         background: radial-gradient(ellipse at top, #0b1630 0%, var(--bg) 62%);
         font-family: "JetBrains Mono", Consolas, Menlo, monospace; }
  header { display:flex; flex-wrap:wrap; align-items:center; gap:10px 22px; padding:14px 20px;
           border-bottom:1px solid #12304d; }
  h1 { margin:0; font-size:17px; letter-spacing:2px; color:var(--cyan); text-shadow:0 0 10px rgba(0,240,255,.7); }
  .tag { color:var(--dim); font-size:12px; }
  .problem { flex-basis:100%; font-size:13px; color:#9cc7ee; }
  .chips { display:flex; gap:8px; flex-wrap:wrap; }
  .chip { border:1px solid #1d4466; border-radius:3px; padding:3px 8px; font-size:12px; }
  .chip b { color:#fff; }
  main { display:grid; grid-template-columns:minmax(0,2.2fr) minmax(330px,1fr); gap:16px; padding:16px 20px; }
  @media (max-width: 1050px) { main { grid-template-columns: 1fr; } }
  .panel { border:1px solid #16405f; background:var(--panel); border-radius:4px; padding:12px;
           box-shadow:0 0 18px rgba(0,240,255,.08); min-width:0; }
  h2 { margin:0 0 8px 0; font-size:13px; letter-spacing:1px; color:var(--cyan); }
  .controls { display:flex; align-items:center; gap:8px; margin-bottom:10px; font-size:12px; flex-wrap:wrap; }
  button, select { background:#07223b; color:var(--cyan); border:1px solid var(--cyan); padding:4px 10px;
                   font-family:inherit; font-size:12px; cursor:pointer; border-radius:2px; }
  button:hover { background:#0b3357; }
  input[type="range"] { flex:1; min-width:160px; accent-color:var(--cyan); }
  #arenaWrap { position:relative; }
  canvas { width:100%; height:580px; display:block; border:1px solid #102a44; background:#02040a; }
  #tooltip { position:absolute; pointer-events:none; display:none; max-width:360px; z-index:2;
             background:rgba(2,8,18,.97); border:1px solid var(--cyan); padding:8px; font-size:11px; line-height:1.45; }
  #tooltip .tt-id { color:var(--cyan); font-weight:bold; }
  #halt { display:none; margin-top:8px; border:1px solid var(--red); color:var(--red); padding:6px 10px; font-size:12px; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:var(--dim); margin-top:8px; }
  .legend i { display:inline-block; width:18px; height:3px; vertical-align:middle; margin-right:5px; }
  .winners-box { margin-top:12px; border:2px solid var(--green); box-shadow:0 0 22px rgba(0,255,157,.25);
                 padding:12px; background:rgba(0,40,25,.45); }
  .winner { margin-bottom:10px; padding-bottom:8px; border-bottom:1px dashed #1f5e44; }
  .winner:last-child { border-bottom:none; margin-bottom:0; padding-bottom:0; }
  .winner .head { color:var(--green); font-weight:bold; }
  .winner .sol { margin-top:4px; color:#e9fff6; white-space:pre-wrap; font-size:12.5px; }
  .winner .meta { margin-top:4px; color:#7fbfa4; font-size:11px; }
  .awaiting { color:var(--dim); }
  table { width:100%; border-collapse:collapse; font-size:11.5px; }
  th, td { border-bottom:1px solid #12283f; padding:4px 3px; text-align:left; white-space:nowrap; }
  th { color:var(--dim); font-weight:normal; position:sticky; top:0; background:#081324; }
  tbody tr:hover { background:#0b1c33; }
  .hud { max-height:340px; overflow:auto; }
  .state-Pass { color:var(--green); font-weight:bold; }
  .state-Possible { color:var(--cyan); }
  .state-Absorbed { color:#ffb84d; }
  .state-Fail { color:var(--red); text-decoration:line-through; }
  #ticker { list-style:none; margin:0; padding:0; font-size:11.5px; max-height:240px; overflow:auto; }
  #ticker li { padding:3px 0; border-bottom:1px solid #0f2236; }
  .ev-fail { color:var(--red); } .ev-pass, .ev-winner_exit { color:var(--green); }
  .ev-merge { color:var(--amber); } .ev-revocation { color:#ff8a3d; } .ev-ns_tick { color:#c49bff; }
  .ev-branch { color:var(--cyan); } .ev-halt { color:var(--red); font-weight:bold; }
  .ev-falsification_inconclusive { color:#9aa7b5; }
  details { margin-top:12px; } summary { cursor:pointer; color:var(--dim); font-size:12px; }
  pre { font-size:10.5px; overflow:auto; color:#9cdbff; max-height:420px; }
</style>
</head>
<body>
<header>
  <h1>NEBIUS — SURVIVAL OF THE FITTEST // TRON ARENA</h1>
  <span class="tag">PFP Darwinism · Pass · Fail · Possible</span>
  <div class="chips" id="chips"></div>
  <div class="problem" id="problem"></div>
</header>
<main>
  <section class="panel">
    <div class="controls">
      <button id="playBtn" title="Play / pause (space)">&#9654; Play</button>
      <button id="backBtn" title="Step back (&larr;)">&#9664;</button>
      <button id="fwdBtn" title="Step forward (&rarr;)">&#9654;&#xFE0E;</button>
      <select id="speedSel" title="Playback speed (events per second)">
        <option value="4">0.5&times;</option>
        <option value="10" selected>1&times;</option>
        <option value="25">2.5&times;</option>
        <option value="80">8&times;</option>
      </select>
      <input id="seqSlider" type="range" min="0" max="__N__" value="__N__" />
      <span id="seqLabel"></span>
    </div>
    <div id="arenaWrap">
      <canvas id="arenaCanvas"></canvas>
      <div id="tooltip"></div>
    </div>
    <div class="legend">
      <span><i style="background:#00ff9d"></i>verified main line / Pass</span>
      <span><i style="background:#ff3366"></i>crash into main line (Fail)</span>
      <span><i style="background:#ffd479"></i>merge (cooperation)</span>
      <span><i style="background:#c49bff"></i>NS blind variation</span>
      <span>colors = starter lineage (cosmetic)</span>
    </div>
    <div id="halt"></div>
    <div class="winners-box" id="winnersBox">
      <h2>WINNER'S BOX — VERIFIED SURVIVORS</h2>
      <div id="winnerContent"></div>
    </div>
  </section>
  <aside class="panel">
    <h2>SCOREBOARD HUD (TIME ALIVE + STATE)</h2>
    <div class="hud">
      <table>
        <thead><tr><th>Agent</th><th>Origin</th><th>State</th><th>Alive</th><th>Cert</th><th>Budget</th></tr></thead>
        <tbody id="hudBody"></tbody>
      </table>
    </div>
    <h2 style="margin-top:14px;">EVENT FEED</h2>
    <ul id="ticker"></ul>
    <details>
      <summary>STREAM SNAPSHOT (ASCII render of the same feed)</summary>
      <pre id="ascii">__ASCII__</pre>
    </details>
  </aside>
</main>
<script>
const ARENA_EVENTS = __EVENTS_JSON__;
const PROBLEM = __PROBLEM_JSON__;
(function () {
  "use strict";
  const N = ARENA_EVENTS.length;
  const LINEAGE = ["#00f0ff", "#ff9f1c", "#ff4ecd", "#f5f749", "#7cff6b", "#4d9dff", "#ff6b6b", "#2ef2c4", "#c8a2ff"];
  const C_PASS = "#00ff9d", C_FAIL = "#ff3366", C_MERGE = "#ffd479", C_NS = "#c49bff";
  const STATE_RANK = { Pass: 3, Possible: 2, Absorbed: 1, Fail: 0 };
  const TICKER_TYPES = new Set(["spawn", "branch", "merge", "fail", "pass", "revocation", "ns_tick",
                                "winner_exit", "falsification_survived", "falsification_inconclusive", "halt"]);
  const el = (id) => document.getElementById(id);
  el("problem").textContent = PROBLEM ? "PROBLEM: " + PROBLEM : "";

  // ---- Layout pass over the full stream: lanes are stable while scrubbing ----
  const B = {};
  const roots = [];
  function born(id, idx, snap, parent, mergeParents, origin) {
    if (B[id]) return B[id];
    const b = { id, parent, mergeParents: mergeParents || [], origin: origin || snap.origin || "starter",
                birth: idx, end: null, endType: null, passAt: [], revokedAt: [], exitAt: null,
                children: [], lane: 0, color: "#ffffff" };
    B[id] = b;
    if (parent && B[parent]) B[parent].children.push(id);
    else if (!mergeParents || mergeParents.length === 0) roots.push(id);
    return b;
  }
  ARENA_EVENTS.forEach((e, i) => {
    const s = e.snapshot || {};
    if (e.event_type === "spawn") born(e.branch_id, i, s, s.parent || null, s.merge_parents, s.origin);
    else if (e.event_type === "merge") born(e.branch_id, i, s, null, [e.details.parent_a_id, e.details.parent_b_id], "merge");
    const b = B[e.branch_id];
    if (!b) return;
    if (b.end === null) {
      if (e.event_type === "fail" || s.state === "Fail") { b.end = i; b.endType = "fail"; }
      else if (s.state === "Absorbed") { b.end = i; b.endType = "absorbed"; }
    }
    if (e.event_type === "pass") b.passAt.push(i);
    if (e.event_type === "revocation") b.revokedAt.push(i);
    if (e.event_type === "winner_exit" && b.exitAt === null) b.exitAt = i;
  });

  // DFS preorder: every branch owns a vertical lane; children sit beside their parent
  const laneOrder = [];
  function walk(id, color) {
    const b = B[id];
    b.color = color;
    laneOrder.push(id);
    b.children.forEach((c) => walk(c, color));
  }
  roots.forEach((r, k) => walk(r, LINEAGE[k % LINEAGE.length]));
  const merged = Object.values(B).filter((b) => b.origin === "merge").sort((a, b) => a.birth - b.birth);
  let slots = laneOrder.length;
  laneOrder.forEach((id, k) => { B[id].lane = k; });
  merged.forEach((m, k) => {
    const ps = m.mergeParents.map((p) => B[p]).filter(Boolean);
    m.lane = ps.length ? ps.reduce((a, p) => a + p.lane, 0) / ps.length + 0.35 : slots + k;
    m.color = C_MERGE;
  });

  // ---- Canvas geometry ----
  const canvas = el("arenaCanvas");
  const ctx = canvas.getContext("2d");
  let W = 900, H = 580;
  const TOP = 46, BOX_H = 56;
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    W = canvas.clientWidth || 900;
    H = canvas.clientHeight || 580;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  const midX = () => W / 2;
  const BOTTOM = () => H - BOX_H - 18;
  function laneX(lane) {
    const n = Math.max(1, slots), gap = 46, margin = 26;
    const half = Math.ceil(n / 2);
    const leftW = midX() - gap / 2 - margin, rightW = W - margin - (midX() + gap / 2);
    if (lane < half) return margin + (half === 1 ? leftW / 2 : (lane / (half - 1)) * leftW);
    const r = lane - half, nr = Math.max(1, n - half);
    return midX() + gap / 2 + (nr === 1 ? rightW / 2 : (r / (nr - 1)) * rightW);
  }
  const yOf = (idx) => TOP + (Math.min(Math.max(idx, 0), Math.max(1, N - 1)) / Math.max(1, N - 1)) * (BOTTOM() - TOP);

  function glowLine(points, color, width, blur) {
    if (points.length < 2) return;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "miter"; ctx.lineCap = "square";
    ctx.shadowColor = color; ctx.shadowBlur = blur;
    ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
    for (let k = 1; k < points.length; k++) ctx.lineTo(points[k][0], points[k][1]);
    ctx.stroke(); ctx.restore();
  }
  function spark(x, y, color, t) {
    ctx.save(); ctx.strokeStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 14; ctx.lineWidth = 1.5;
    const r = 7 + 3 * Math.sin(t);
    for (let a = 0; a < 8; a++) {
      const th = (Math.PI / 4) * a + 0.3;
      ctx.beginPath(); ctx.moveTo(x + Math.cos(th) * 3, y + Math.sin(th) * 3);
      ctx.lineTo(x + Math.cos(th) * r, y + Math.sin(th) * r); ctx.stroke();
    }
    ctx.fillStyle = color; ctx.fillRect(x - 3, y - 3, 6, 6); ctx.restore();
  }

  // ---- State at cursor: pure reduction over events[0 : upto] ----
  function stateAt(upto) {
    const snaps = {}; const winners = []; let halt = null; let spend = 0;
    for (let i = 0; i < upto; i++) {
      const e = ARENA_EVENTS[i];
      if (e.event_type === "halt") { halt = e.details; continue; }
      snaps[e.branch_id] = e.snapshot;
      const d = e.details || {};
      if (e.event_type === "budget_change" && d.spent_usd) spend += Number(d.spent_usd) || 0;
      if (e.event_type === "merge" && d.synthesis_cost_usd) spend += Number(d.synthesis_cost_usd) || 0;
      if (e.event_type === "winner_exit" && !winners.some((w) => w.id === e.branch_id)) winners.push({ id: e.branch_id, snap: e.snapshot, seq: e.seq });
      if (e.event_type === "revocation") {
        const k = winners.findIndex((w) => w.id === e.branch_id);
        if (k >= 0) winners.splice(k, 1);
      }
    }
    return { snaps, winners, halt, spend };
  }

  let cursor = N, playing = false, lastTs = 0, hoverXY = null;

  function draw(c) {
    const upto = Math.floor(c);
    const frac = c - upto;
    const yNow = yOf(upto - 1 + frac);
    ctx.clearRect(0, 0, W, H);
    // Tron floor grid
    ctx.strokeStyle = "rgba(0,240,255,0.06)"; ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 28) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 28) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    // Ignition bar + verified main line (trunk) brightening with every Pass
    ctx.fillStyle = "rgba(0,240,255,0.08)"; ctx.fillRect(0, 0, W, TOP - 16);
    ctx.fillStyle = "#5b7a99"; ctx.font = "10px monospace"; ctx.fillText("IGNITION", 8, 18);
    let passes = 0;
    Object.values(B).forEach((b) => b.passAt.forEach((p) => { if (p < upto) passes++; }));
    glowLine([[midX(), TOP - 18], [midX(), BOTTOM() + 10]], C_PASS, 2 + Math.min(4, passes * 0.6), 10 + passes * 3);
    ctx.fillStyle = C_PASS; ctx.save(); ctx.translate(midX() + 6, TOP + 4); ctx.rotate(Math.PI / 2);
    ctx.fillText("VERIFIED MAIN LINE", 0, 0); ctx.restore();
    // Winner's box gate at the bottom
    const gx = midX() - 120, gy = H - BOX_H - 6;
    ctx.save(); ctx.strokeStyle = C_PASS; ctx.shadowColor = C_PASS; ctx.shadowBlur = 16; ctx.lineWidth = 2;
    ctx.strokeRect(gx, gy, 240, BOX_H); ctx.restore();
    ctx.fillStyle = C_PASS; ctx.font = "11px monospace"; ctx.fillText("WINNER'S BOX", midX() - 40, gy + 18);

    const order = Object.values(B).sort((a, b) => a.birth - b.birth);
    let exitsShown = 0;
    order.forEach((b) => {
      if (b.birth >= upto) return;
      const x = laneX(b.lane), y0 = yOf(b.birth);
      const ended = b.end !== null && b.end < upto;
      const yEnd = ended ? yOf(b.end) : yNow;
      const baseColor = b.origin === "ns" ? C_NS : b.color;
      // Entry: starters drop from the top; children turn off their parent's lane at a right angle
      if (b.origin === "merge") {
        b.mergeParents.forEach((pid) => {
          const p = B[pid]; if (!p) return;
          const px = laneX(p.lane), py = p.end !== null ? yOf(p.end) : y0;
          glowLine([[px, py], [px, y0], [x, y0]], C_MERGE, 2.2, 12);
        });
      } else if (b.parent && B[b.parent]) {
        glowLine([[laneX(B[b.parent].lane), y0], [x, y0]], baseColor, 1.6, 8);
      } else {
        glowLine([[x, TOP - 18], [x, y0]], baseColor, 2.4, 12);
      }
      // Vertical trail; segments after a live Pass glow green
      let segStart = y0, color = baseColor;
      const marks = [];
      b.passAt.forEach((p) => marks.push([p, C_PASS]));
      b.revokedAt.forEach((p) => marks.push([p, baseColor]));
      marks.sort((m1, m2) => m1[0] - m2[0]);
      marks.forEach(([idx, col]) => {
        if (idx >= upto) return;
        const ym = Math.min(yOf(idx), yEnd);
        glowLine([[x, segStart], [x, ym]], color, color === C_PASS ? 2.6 : 1.8, 10);
        segStart = ym; color = col;
      });
      glowLine([[x, segStart], [x, yEnd]], color, color === C_PASS ? 2.6 : 1.8, 10);
      if (ended && b.endType === "fail") {
        // Tron rules: contradiction with established truth = crash into the main line
        glowLine([[x, yEnd], [midX(), yEnd]], C_FAIL, 2, 12);
        spark(midX(), yEnd, C_FAIL, c);
      } else if (!ended) {
        ctx.save(); ctx.fillStyle = "#ffffff"; ctx.shadowColor = color; ctx.shadowBlur = 18;
        ctx.fillRect(x - 3.5, yEnd - 3.5, 7, 7); ctx.restore();
      }
      if (b.exitAt !== null && b.exitAt < upto) {
        const slot = midX() - 90 + (exitsShown++ % 4) * 60;
        glowLine([[x, yOf(b.exitAt)], [x, gy - 8], [slot, gy - 8], [slot, gy + BOX_H / 2]], C_PASS, 2.8, 18);
      }
    });

    renderPanels(upto);
  }

  function fmt(n, d) { return Number(n || 0).toFixed(d); }

  function renderPanels(upto) {
    const st = stateAt(upto);
    el("seqSlider").value = String(upto);
    el("seqLabel").textContent = "Event " + upto + " / " + N;
    const rows = Object.values(st.snaps).filter((r) => r && r.id);
    const count = (s) => rows.filter((r) => r.state === s).length;
    const chips = el("chips"); chips.textContent = "";
    [["events", upto + "/" + N], ["alive", count("Possible") + count("Pass")], ["pass", count("Pass")],
     ["fail", count("Fail")], ["absorbed", count("Absorbed")], ["spend", "$" + fmt(st.spend, 4)]].forEach(([k, v]) => {
      const span = document.createElement("span"); span.className = "chip";
      const b = document.createElement("b"); b.textContent = String(v);
      span.append(k + " ", b); chips.appendChild(span);
    });
    // Scoreboard HUD
    const tbody = el("hudBody"); tbody.textContent = "";
    rows.sort((a, b) => (STATE_RANK[b.state] - STATE_RANK[a.state]) || (b.certainty_score - a.certainty_score))
      .forEach((r) => {
        const tr = document.createElement("tr");
        [[r.id, ""], [r.origin || "", ""], [r.state, "state-" + r.state], [fmt(r.time_alive, 3) + "s", ""],
         [fmt(r.certainty_score, 2), ""], ["$" + fmt(r.budget_remaining, 4), ""]].forEach(([txt, cls]) => {
          const td = document.createElement("td"); td.textContent = String(txt); if (cls) td.className = cls; tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    // Event feed (newest first)
    const ul = el("ticker"); ul.textContent = "";
    const recent = [];
    for (let i = upto - 1; i >= 0 && recent.length < 16; i--) {
      if (TICKER_TYPES.has(ARENA_EVENTS[i].event_type)) recent.push(ARENA_EVENTS[i]);
    }
    recent.forEach((e) => {
      const li = document.createElement("li"); li.className = "ev-" + e.event_type;
      li.textContent = "#" + e.seq + " " + e.event_type.toUpperCase() + " " + e.branch_id + describe(e);
      ul.appendChild(li);
    });
    // Winner's box (scrub-aware: revoked winners leave the box)
    const wDiv = el("winnerContent"); wDiv.textContent = "";
    if (st.winners.length === 0) {
      const p = document.createElement("div"); p.className = "awaiting";
      p.textContent = st.halt ? "Run halted before any survivor exited." : "Awaiting verified survivor...";
      wDiv.appendChild(p);
    }
    st.winners.forEach((w, k) => {
      const s = w.snap || {};
      const box = document.createElement("div"); box.className = "winner";
      const head = document.createElement("div"); head.className = "head";
      head.textContent = "#" + (k + 1) + "  " + w.id + "  [" + (s.state || "") + "]  certainty " + fmt(s.certainty_score, 2) +
        "  survived " + (s.survived_falsifications || 0);
      const sol = document.createElement("div"); sol.className = "sol";
      sol.textContent = s.solution || s.fork_under_exploration || "";
      const meta = document.createElement("div"); meta.className = "meta";
      const angles = (s.pass_provenance || []).map((p) => p.falsifier_angle).join(", ");
      const lineage = (s.merge_parents || []).length ? "  ·  cooperation lineage: " + s.merge_parents.join(" + ") : "";
      meta.textContent = "proof trail: " + (angles || "—") + lineage;
      box.append(head, sol, meta); wDiv.appendChild(box);
    });
    const haltDiv = el("halt");
    if (st.halt) { haltDiv.style.display = "block"; haltDiv.textContent = "RUN HALTED (" + st.halt.reason + "): " + (st.halt.detail || ""); }
    else { haltDiv.style.display = "none"; }
    renderTooltip(st);
  }

  function describe(e) {
    const d = e.details || {};
    switch (e.event_type) {
      case "branch": return " — " + String(d.trigger || "").toUpperCase() + " split → " + (d.children_ids || []).length + " children";
      case "merge": return " ← " + d.parent_a_id + " + " + d.parent_b_id;
      case "fail": return " — " + (d.angle ? "[" + d.angle + "] " : "") + String(d.objection || "crashed into main line").slice(0, 90);
      case "pass": return " — survived " + (d.angles_survived || []).join(", ");
      case "revocation": return " — " + (d.reason || "") + " → " + (d.demoted_to || "");
      case "ns_tick": return " — blind variation (" + (d.spawned_count || 0) + ")";
      case "winner_exit": return " — exits to the winner's box";
      case "falsification_survived": return " — survived [" + (d.angle || "") + "] " + (d.schedule_trigger || "");
      case "falsification_inconclusive": return " — judge gave no verdict";
      case "halt": return " — " + (d.reason || "");
      default: return "";
    }
  }

  function renderTooltip(st) {
    const tip = el("tooltip");
    if (!hoverXY) { tip.style.display = "none"; return; }
    const [mx, my] = hoverXY;
    let best = null, bestD = 9;
    Object.values(B).forEach((b) => {
      if (!st.snaps[b.id]) return;
      const dx = Math.abs(laneX(b.lane) - mx);
      const yEnd = b.end !== null ? yOf(b.end) : BOTTOM();
      if (dx < bestD && my >= yOf(b.birth) - 6 && my <= yEnd + 6) { best = b; bestD = dx; }
    });
    if (!best) { tip.style.display = "none"; return; }
    const s = st.snaps[best.id];
    tip.textContent = "";
    const head = document.createElement("div"); head.className = "tt-id";
    head.textContent = best.id + " · " + (s.origin || "") + " · " + s.state + " · cert " + fmt(s.certainty_score, 2);
    const fork = document.createElement("div"); fork.textContent = "fork: " + (s.fork_under_exploration || "");
    const last = document.createElement("div"); last.style.color = "#9cc7ee"; last.textContent = (s.last_step || "").slice(0, 260);
    tip.append(head, fork, last);
    tip.style.display = "block";
    tip.style.left = Math.min(mx + 14, W - 370) + "px";
    tip.style.top = Math.min(my + 14, H - 120) + "px";
  }

  // Public hook used by tests / embedding pages: render the arena as of event `n`
  function renderUpTo(n) { cursor = Math.max(0, Math.min(N, Number(n))); draw(cursor); }
  window.renderUpTo = renderUpTo;

  function tick(ts) {
    if (!playing) return;
    const dt = lastTs ? (ts - lastTs) / 1000 : 0;
    lastTs = ts;
    cursor = Math.min(N, cursor + dt * Number(el("speedSel").value));
    draw(cursor);
    if (cursor >= N) { setPlaying(false); return; }
    requestAnimationFrame(tick);
  }
  function setPlaying(on) {
    playing = on; lastTs = 0;
    el("playBtn").innerHTML = on ? "&#10074;&#10074; Pause" : "&#9654; Play";
    if (on) { if (cursor >= N) cursor = 0; requestAnimationFrame(tick); }
  }
  el("playBtn").addEventListener("click", () => setPlaying(!playing));
  el("backBtn").addEventListener("click", () => { setPlaying(false); renderUpTo(Math.ceil(cursor) - 1); });
  el("fwdBtn").addEventListener("click", () => { setPlaying(false); renderUpTo(Math.floor(cursor) + 1); });
  el("seqSlider").addEventListener("input", (e) => { setPlaying(false); renderUpTo(e.target.value); });
  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect(); hoverXY = [e.clientX - r.left, e.clientY - r.top]; renderPanels(Math.floor(cursor));
  });
  canvas.addEventListener("mouseleave", () => { hoverXY = null; el("tooltip").style.display = "none"; });
  document.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;
    if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
    else if (e.key === "ArrowLeft") { setPlaying(false); renderUpTo(Math.ceil(cursor) - 1); }
    else if (e.key === "ArrowRight") { setPlaying(false); renderUpTo(Math.floor(cursor) + 1); }
    else if (e.key === "Home") { setPlaying(false); renderUpTo(0); }
    else if (e.key === "End") { setPlaying(false); renderUpTo(N); }
  });
  window.addEventListener("resize", () => { resize(); draw(cursor); });
  resize();
  // Deep link: arena.html#seq=120 opens at event 120; #play autoplays from the start (demo capture)
  const hash = new URLSearchParams((location.hash || "").replace(/^#/, ""));
  if (hash.has("seq")) renderUpTo(Number(hash.get("seq")) || 0);
  else renderUpTo(N);
  if (hash.has("play")) { cursor = 0; setPlaying(true); }
})();
</script>
</body>
</html>
"""

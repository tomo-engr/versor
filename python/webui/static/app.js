// versor WebUI frontend. Stateless engine calls; the browser holds the
// game state (board string, side to move, move history for undo/export
// and an eval graph). All rules and scoring come from the backend.
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
const CELL = 50;
const OFF = 20; // margin for coordinate labels

// Game state.
let state = null; // latest engine state object {board, stm, legal, ...}
let history = []; // list of {move, state} applied from the initial position
let evalPoints = []; // [{ply, value_black}]

function api(path, body) {
  return fetch("api/" + path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  }).then(async (r) => {
    if (!r.ok) {
      const detail = await r.json().catch(() => ({}));
      throw new Error(detail.detail || ("HTTP " + r.status));
    }
    return r.json();
  });
}

function sqName(row, col) {
  return "abcdefgh"[col] + (row + 1);
}
function sqIndex(name) {
  return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
}

// ---- rendering ----
const board = document.getElementById("board");

function render() {
  board.innerHTML = "";
  // coordinate labels
  for (let c = 0; c < 8; c++) {
    addText(OFF + c * CELL + CELL / 2, 14, "abcdefgh"[c], "coord");
    addText(OFF + c * CELL + CELL / 2, OFF + 8 * CELL + 14, "abcdefgh"[c], "coord");
  }
  for (let r = 0; r < 8; r++) {
    addText(10, OFF + r * CELL + CELL / 2 + 4, String(r + 1), "coord");
    addText(OFF + 8 * CELL + 12, OFF + r * CELL + CELL / 2 + 4, String(r + 1), "coord");
  }
  const b = state.board;
  const legal = new Set(state.legal || []);
  const isHumanTurn = mode() === "analysis" || state.stm !== engineColor();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const x = OFF + c * CELL;
      const y = OFF + r * CELL;
      const name = sqName(r, c);
      const rect = el("rect", { x, y, width: CELL, height: CELL, class: "cell" });
      rect.addEventListener("click", () => onCellClick(name));
      board.appendChild(rect);
      const ch = b[r * 8 + c];
      if (ch === "X") disc(x, y, "disc-black");
      else if (ch === "O") disc(x, y, "disc-white");
      if (legal.has(name) && !state.game_over && isHumanTurn && !showHints()) {
        board.appendChild(el("circle", { cx: x + CELL / 2, cy: y + CELL / 2, r: 6, class: "legal-dot" }));
      }
    }
  }
  // last move marker
  const last = history.length ? history[history.length - 1].move : null;
  if (last && last !== "pass") {
    const i = sqIndex(last);
    const x = OFF + (i % 8) * CELL, y = OFF + Math.floor(i / 8) * CELL;
    board.appendChild(el("rect", { x: x + 3, y: y + 3, width: CELL - 6, height: CELL - 6, class: "last-move" }));
  }
  updateStatus();
}

function disc(x, y, cls) {
  board.appendChild(el("circle", { cx: x + CELL / 2, cy: y + CELL / 2, r: CELL / 2 - 5, class: cls }));
}
function el(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function addText(x, y, text, cls) {
  const t = el("text", { x, y, class: cls });
  t.textContent = text;
  board.appendChild(t);
}

function updateStatus() {
  document.getElementById("counts").textContent = t("status.counts", {
    black: t("board.black"), b: state.black,
    white: t("board.white"), w: state.white,
  });
  const turnEl = document.getElementById("turn");
  if (state.game_over) {
    const key = { black: "result.black", white: "result.white", draw: "result.draw" };
    turnEl.textContent = t("status.gameOver", {
      result: key[state.result] ? t(key[state.result]) : "",
    });
  } else {
    turnEl.textContent =
      t("status.turn", { color: t(state.stm === "X" ? "board.black" : "board.white") })
      + (state.must_pass ? t("status.mustPass") : "");
  }
  document.getElementById("pass").hidden = !(state.must_pass && !state.game_over);
}

// ---- controls ----
function mode() { return document.getElementById("mode").value; }
function engineColor() { return document.getElementById("engine-color").value; }
function showHints() { return document.getElementById("show-hints").checked; }
function hintDepth() { return parseInt(document.getElementById("hint-depth").value, 10) || 6; }

function limitBody(base) {
  const kind = document.getElementById("limit-kind").value;
  const v = parseInt(document.getElementById("limit-value").value, 10) || 8;
  base[kind] = v;
  return base;
}

async function onCellClick(name) {
  if (!state || state.game_over) return;
  if (!(state.legal || []).includes(name)) return;
  const humanTurn = mode() === "analysis" || state.stm !== engineColor();
  if (!humanTurn) return;
  await applyMove(name);
  if (mode() === "play") await maybeEngineMove();
}

async function applyMove(move) {
  const res = await api("play", { board: state.board, stm: state.stm, move });
  history.push({ move, state: res });
  state = res;
  await appendEvalPoint();
  render();
  if (showHints()) await refreshHints();
  if (mode() === "analysis" || state.stm !== engineColor()) await refreshAnalysis();
}

async function maybeEngineMove() {
  // Auto-play engine moves (and passes) while it is the engine's turn.
  while (!state.game_over && state.stm === engineColor()) {
    if (state.must_pass) {
      await applyMove("pass");
      continue;
    }
    const a = await api("analyze", limitBody({ board: state.board, stm: state.stm }));
    showAnalysis(a);
    if (a.best_move) await applyMove(a.best_move);
    else await applyMove("pass");
  }
}

async function refreshAnalysis() {
  if (state.game_over || state.must_pass) { showAnalysis(null); return; }
  const a = await api("analyze", limitBody({ board: state.board, stm: state.stm }));
  showAnalysis(a);
}

function showAnalysis(a) {
  if (!a) {
    document.getElementById("eval-value").textContent = "-";
    document.getElementById("pv").textContent = "-";
    document.getElementById("search-stats").textContent = "-";
    return;
  }
  document.getElementById("eval-value").textContent =
    (a.value > 0 ? "+" : "") + a.value
    + (a.best_move ? t("analysis.best", { move: a.best_move }) : "");
  document.getElementById("pv").textContent = a.pv.join(" ") || "-";
  document.getElementById("search-stats").textContent =
    t("analysis.stats", { depth: a.depth, nodes: a.nodes, ms: a.elapsed_ms });
}

async function refreshHints() {
  // Remove old hint text, then draw per-move values.
  Array.from(board.querySelectorAll(".hint-text")).forEach((n) => n.remove());
  if (state.game_over || state.must_pass) return;
  const h = await api("hints", { board: state.board, stm: state.stm, depth: hintDepth() });
  for (const item of h.hints) {
    const i = sqIndex(item.move);
    const x = OFF + (i % 8) * CELL, y = OFF + Math.floor(i / 8) * CELL;
    const t = el("text", { x: x + CELL / 2, y: y + CELL / 2 + 4, class: "hint-text" });
    t.textContent = (item.value > 0 ? "+" : "") + item.value;
    board.appendChild(t);
  }
}

// ---- eval graph ----
async function appendEvalPoint() {
  const ply = history.length;
  let vb;
  if (state.game_over) {
    vb = state.black - state.white;
    if (state.black > state.white) vb += state.empty;
    else if (state.white > state.black) vb -= state.empty;
  } else if (state.must_pass) {
    vb = evalPoints.length ? evalPoints[evalPoints.length - 1].value_black : 0;
  } else {
    const a = await api("analyze", { board: state.board, stm: state.stm, depth: 4 });
    vb = state.stm === "X" ? a.value : -a.value;
  }
  evalPoints.push({ ply, value_black: vb });
  drawGraph();
}

function drawGraph() {
  const g = document.getElementById("graph");
  g.innerHTML = "";
  const W = 440, H = 140, pad = 10;
  const maxAbs = Math.max(8, ...evalPoints.map((p) => Math.abs(p.value_black)));
  const n = Math.max(1, evalPoints.length);
  const xOf = (i) => pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
  const yOf = (v) => H / 2 - (v / maxAbs) * (H / 2 - pad);
  const zero = el("line", { x1: pad, y1: H / 2, x2: W - pad, y2: H / 2, class: "graph-zero" });
  g.appendChild(zero);
  if (evalPoints.length >= 1) {
    let d = "";
    evalPoints.forEach((p, i) => { d += (i ? "L" : "M") + xOf(i) + " " + yOf(p.value_black) + " "; });
    g.appendChild(el("path", { d, class: "graph-line" }));
    evalPoints.forEach((p, i) => g.appendChild(el("circle", { cx: xOf(i), cy: yOf(p.value_black), r: 2, class: "graph-dot" })));
  }
}

// ---- game flow ----
async function newGame() {
  const init = await api("initial");
  state = init;
  history = [];
  evalPoints = [];
  await appendEvalPointInitial();
  render();
  showAnalysis(null);
  Array.from(board.querySelectorAll(".hint-text")).forEach((n) => n.remove());
  if (mode() === "play") await maybeEngineMove();
  else if (showHints()) await refreshHints();
}

async function appendEvalPointInitial() {
  const a = await api("analyze", { board: state.board, stm: state.stm, depth: 4 });
  evalPoints.push({ ply: 0, value_black: state.stm === "X" ? a.value : -a.value });
  drawGraph();
}

async function undo() {
  if (!history.length) return;
  history.pop();
  evalPoints.pop();
  if (history.length) state = history[history.length - 1].state;
  else { state = await api("initial"); }
  render();
  if (showHints()) await refreshHints();
  await refreshAnalysis();
}

async function passMove() {
  if (state.must_pass && !state.game_over) {
    await applyMove("pass");
    if (mode() === "play") await maybeEngineMove();
  }
}

function currentMovesString() {
  return history.filter((h) => h.move !== "pass").map((h) => h.move).join("");
}

async function importGame() {
  const moves = document.getElementById("moves-io").value.trim();
  const res = await api("import", { moves });
  // Rebuild history and graph from the returned states.
  history = res.states.map((s) => ({ move: s.move, state: s }));
  state = res.final;
  const graph = await api("evalgraph", { moves, depth: 4 });
  evalPoints = graph.points.map((p) => ({ ply: p.ply, value_black: p.value_black }));
  drawGraph();
  render();
  if (showHints()) await refreshHints();
  await refreshAnalysis();
}

function exportGame() {
  document.getElementById("moves-io").value = currentMovesString();
}

// ---- wiring ----
document.getElementById("newgame").addEventListener("click", () => newGame().catch(err));
document.getElementById("undo").addEventListener("click", () => undo().catch(err));
document.getElementById("pass").addEventListener("click", () => passMove().catch(err));
document.getElementById("engine-move").addEventListener("click", () => maybeEngineMoveOnce().catch(err));
document.getElementById("show-hints").addEventListener("change", () => {
  if (showHints()) refreshHints().catch(err);
  else { Array.from(board.querySelectorAll(".hint-text")).forEach((n) => n.remove()); render(); }
});
document.getElementById("mode").addEventListener("change", () => {
  const analysis = mode() === "analysis";
  document.getElementById("play-controls").hidden = analysis;
  document.getElementById("analysis-controls").hidden = !analysis;
  render();
});
document.getElementById("import-btn").addEventListener("click", () => importGame().catch(err));
document.getElementById("export-btn").addEventListener("click", exportGame);

// In analysis mode, let the button make exactly one engine move.
async function maybeEngineMoveOnce() {
  if (state.game_over) return;
  if (state.must_pass) { await applyMove("pass"); return; }
  const a = await api("analyze", limitBody({ board: state.board, stm: state.stm }));
  showAnalysis(a);
  if (a.best_move) await applyMove(a.best_move);
}

function err(e) { alert(t("error.prefix", { message: e.message })); console.error(e); }

// Language selector. Static text is refilled by I18N.apply(); the parts
// that app.js writes are redrawn here.
document.getElementById("lang").addEventListener("change", (ev) => {
  I18N.set(ev.target.value);
  if (state) { updateStatus(); render(); }
});

// Start in the default language (English), even if the browser restored
// another choice in the selector.
document.getElementById("lang").value = I18N.lang;
I18N.apply();
newGame().catch(err);

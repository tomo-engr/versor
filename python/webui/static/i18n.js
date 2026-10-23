// UI strings for the WebUI, in one place.
//
// Every user-visible string lives in STRINGS below; nothing is written
// into the markup or into app.js. Static text is tagged in index.html
// with data-i18n (textContent) or data-i18n-ph (placeholder) and filled
// in by apply(). Dynamic text goes through t(key, params).
//
// English is the default and the fallback for a missing key. Adding a
// language means adding one entry to STRINGS and one <option> to the
// language selector.
"use strict";

const STRINGS = {
  en: {
    "page.title": "versor Reversi",
    "page.heading": "versor Reversi",
    "lang.label": "Language",

    "sec.game": "Game",
    "btn.newgame": "New game",
    "btn.undo": "Undo",
    "btn.pass": "Pass",
    "label.mode": "Mode",
    "mode.play": "Play (human vs versor)",
    "mode.analysis": "Analysis (free setup)",
    "label.engineColor": "versor plays",
    "color.white": "White (second)",
    "color.black": "Black (first)",
    "label.limitKind": "Search limit",
    "limit.depth": "Depth",
    "limit.nodes": "Nodes",
    "limit.ms": "Time (ms)",
    "btn.engineMove": "Let versor move once",

    "sec.analysis": "Analysis",
    "label.showHints": "Show the value of each move on the board",
    "label.hintDepth": "Hint depth",
    "label.eval": "Value",
    "eval.unit": "(disc difference, side to move)",
    "label.pv": "Principal variation",
    "label.search": "Search",
    "sec.graph": "Value over the game (black's view)",

    "sec.record": "Move list (f5d6 format)",
    "btn.import": "Load",
    "btn.export": "Export current game",
    "ph.moves": "f5d6c3d3c4 ...",

    "board.black": "Black(X)",
    "board.white": "White(O)",
    "status.counts": "{black} {b} - {white} {w}",
    "status.turn": "{color} to move",
    "status.mustPass": " (no legal move: pass)",
    "status.gameOver": "Game over: {result}",
    "result.black": "Black wins",
    "result.white": "White wins",
    "result.draw": "Draw",
    "analysis.best": " (best {move})",
    "analysis.stats": "depth {depth}, {nodes} nodes, {ms} ms",
    "error.prefix": "Error: {message}",
  },

  ja: {
    "page.title": "versor リバーシ",
    "page.heading": "versor リバーシ",
    "lang.label": "言語",

    "sec.game": "対局",
    "btn.newgame": "新規対局",
    "btn.undo": "待った",
    "btn.pass": "パス",
    "label.mode": "モード",
    "mode.play": "対局 (人間 vs versor)",
    "mode.analysis": "検討 (自由入力)",
    "label.engineColor": "versor の色",
    "color.white": "白 (後手)",
    "color.black": "黒 (先手)",
    "label.limitKind": "探索条件",
    "limit.depth": "深さ",
    "limit.nodes": "ノード数",
    "limit.ms": "時間 (ms)",
    "btn.engineMove": "versor に一手打たせる",

    "sec.analysis": "検討",
    "label.showHints": "各手の評価値を盤上に表示",
    "label.hintDepth": "ヒント深さ",
    "label.eval": "評価値",
    "eval.unit": "(手番視点の石差)",
    "label.pv": "読み筋 (PV)",
    "label.search": "探索",
    "sec.graph": "評価値の推移 (黒視点)",

    "sec.record": "棋譜 (f5d6 形式)",
    "btn.import": "読み込み",
    "btn.export": "現在の棋譜を書き出し",
    "ph.moves": "f5d6c3d3c4 ...",

    "board.black": "黒(X)",
    "board.white": "白(O)",
    "status.counts": "{black} {b} - {white} {w}",
    "status.turn": "{color} の手番",
    "status.mustPass": " (合法手なし: パス)",
    "status.gameOver": "終局: {result}",
    "result.black": "黒の勝ち",
    "result.white": "白の勝ち",
    "result.draw": "引き分け",
    "analysis.best": " (最善 {move})",
    "analysis.stats": "深さ {depth}, {nodes} ノード, {ms} ms",
    "error.prefix": "エラー: {message}",
  },
};

const I18N = {
  lang: "en",

  // Look up a string and substitute {name} placeholders.
  t(key, params) {
    const table = STRINGS[this.lang] || STRINGS.en;
    let s = table[key];
    if (s === undefined) s = (STRINGS.en[key] !== undefined) ? STRINGS.en[key] : key;
    if (!params) return s;
    return s.replace(/\{(\w+)\}/g, (m, name) =>
      (params[name] !== undefined ? params[name] : m));
  },

  // Fill every tagged element in the document.
  apply() {
    document.documentElement.lang = this.lang;
    document.title = this.t("page.title");
    document.querySelectorAll("[data-i18n]").forEach((n) => {
      n.textContent = this.t(n.getAttribute("data-i18n"));
    });
    document.querySelectorAll("[data-i18n-ph]").forEach((n) => {
      n.placeholder = this.t(n.getAttribute("data-i18n-ph"));
    });
  },

  set(lang) {
    if (!STRINGS[lang]) return;
    this.lang = lang;
    this.apply();
  },
};

function t(key, params) { return I18N.t(key, params); }

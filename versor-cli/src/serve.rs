// Resident JSON server mode for the WebUI backend.
//
// One JSON request per line on stdin, one JSON response per line on
// stdout. A single persistent engine keeps its transposition table
// warm across requests. versor-core is the sole source of truth for
// rules and the pattern evaluation for scoring, so the frontend and
// backend never re-implement othello logic (this avoids the
// side-to-move sign mistakes that path is prone to).
//
// Board encoding matches versor-core text format: 64 chars row-major
// (a1..h8), 'X' = black, 'O' = white, '-' = empty; "stm" is "X"/"O".
// All values are from the side to move's perspective on the disc
// scale (positive = good for the side to move), the same axis as the
// exact solver.
//
// On start the server prints {"ready": true}. Every request is an
// object with "cmd" and an optional "id", which the response echoes
// together with "ok" (true, or false with an "error" message).
//
// Commands:
//   ping                           -> {"pong": true}
//   state    board, stm            -> board info + legal moves
//   play     board, stm, move      -> resulting position ("pass" ok)
//   analyze  board, stm, depth|nodes|ms -> best move, value, pv, stats;
//            "solver" tells whether the exact solver answered
//   hints    board, stm, depth     -> per legal move value (stm view)

use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use versor_core::notation::{square_from_str, square_to_string};
use versor_core::{Color, GameResult, Position};
use versor_eval::PatternEval;
use versor_search::{MpcParams, SearchResult, Searcher, Solver};

pub struct ServeConfig {
    pub weights: String,
    pub mpc: Option<String>,
    pub mpc_t: f64,
    pub tt_mb: usize,
    pub hint_depth: u32,
    /// Empty count at or below which the exact endgame solver is used
    /// instead of the depth-limited search. 0 disables it.
    pub solve_at: u32,
    /// Wall-clock cap on a solve when the request carries no budget of
    /// its own. The solver is exact, so the only risk is latency; on
    /// timeout the depth-limited search answers instead.
    pub solve_ms: u64,
}

fn parse_stm(v: &Value) -> Result<Color, String> {
    match v.as_str() {
        Some("X") | Some("x") => Ok(Color::Black),
        Some("O") | Some("o") => Ok(Color::White),
        _ => Err("stm must be \"X\" or \"O\"".to_string()),
    }
}

fn stm_char(c: Color) -> &'static str {
    match c {
        Color::Black => "X",
        Color::White => "O",
    }
}

fn position_from_req(req: &Value) -> Result<Position, String> {
    let board = req
        .get("board")
        .and_then(|v| v.as_str())
        .ok_or("missing board")?;
    let stm = parse_stm(req.get("stm").ok_or("missing stm")?)?;
    let text = format!("{}{}", board, stm_char(stm));
    Position::from_text(&text)
}

fn board_str(pos: &Position) -> String {
    pos.to_text()
        .chars()
        .filter(|c| !c.is_whitespace())
        .take(64)
        .collect()
}

fn legal_list(pos: &Position) -> Vec<String> {
    let mut out = Vec::new();
    let mut m = pos.legal_moves();
    while m != 0 {
        let sq = m.trailing_zeros() as u8;
        m &= m - 1;
        out.push(square_to_string(sq));
    }
    out
}

fn state_value(pos: &Position) -> Value {
    let game_over = pos.is_game_over();
    let result = if game_over {
        match pos.result() {
            Some(GameResult::BlackWin) => Value::from("black"),
            Some(GameResult::WhiteWin) => Value::from("white"),
            Some(GameResult::Draw) => Value::from("draw"),
            None => Value::Null,
        }
    } else {
        Value::Null
    };
    json!({
        "board": board_str(pos),
        "stm": stm_char(pos.stm),
        "black": pos.black().count_ones(),
        "white": pos.white().count_ones(),
        "empty": pos.empty().count_ones(),
        "legal": legal_list(pos),
        "must_pass": pos.must_pass(),
        "game_over": game_over,
        "result": result,
    })
}

struct Engine {
    searcher: Searcher<PatternEval>,
    solver: Solver,
    hint_depth: u32,
    solve_at: u32,
    solve_ms: u64,
}

impl Engine {
    fn new(cfg: &ServeConfig) -> Result<Engine, String> {
        let eval = PatternEval::from_file(&cfg.weights)?;
        let mut searcher = Searcher::new(eval, cfg.tt_mb);
        if let Some(path) = &cfg.mpc {
            searcher.set_mpc(MpcParams::from_file(path)?, cfg.mpc_t);
        }
        let mut solver = Solver::new(cfg.tt_mb);
        if cfg.solve_at > 0 {
            solver.set_eval_order(Box::new(PatternEval::from_file(&cfg.weights)?));
        }
        Ok(Engine {
            searcher,
            solver,
            hint_depth: cfg.hint_depth,
            solve_at: cfg.solve_at,
            solve_ms: cfg.solve_ms,
        })
    }

    /// Try the exact endgame solver, honouring whatever budget the
    /// request carries. Returns None when the solver is disabled, the
    /// position is not deep enough, or the budget ran out; the caller
    /// then falls back to the depth-limited search.
    fn try_solve(&mut self, pos: &Position, req: &Value) -> Option<versor_search::SolveResult> {
        if self.solve_at == 0 || pos.empty().count_ones() > self.solve_at {
            return None;
        }
        let nodes = req.get("nodes").and_then(|v| v.as_u64());
        self.solver.set_node_limit(nodes);
        let limit = if nodes.is_some() {
            None
        } else {
            let ms = req
                .get("ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(self.solve_ms);
            Some(Duration::from_millis(ms))
        };
        self.solver.solve_exact(pos, limit)
    }

    fn analyze(&mut self, pos: &Position, req: &Value) -> Value {
        let start = Instant::now();
        // The endgame is solved exactly when it is close enough, the
        // same rule as the CLI's `search --solve-at`.
        if let Some(r) = self.try_solve(pos, req) {
            return json!({
                "best_move": r.best_move.map(square_to_string),
                "value": r.score,
                "pv": Vec::<String>::new(),
                "nodes": r.nodes,
                "depth": pos.empty().count_ones(),
                "elapsed_ms": start.elapsed().as_millis() as u64,
                "solver": true,
            });
        }
        let r: SearchResult = if let Some(nodes) = req.get("nodes").and_then(|v| v.as_u64()) {
            self.searcher.search_nodes(pos, nodes)
        } else if let Some(ms) = req.get("ms").and_then(|v| v.as_u64()) {
            // Iterative deepening bounded by wall time: grow depth until
            // the elapsed budget is used, running whole iterations.
            self.time_limited(pos, Duration::from_millis(ms))
        } else {
            let depth = req.get("depth").and_then(|v| v.as_u64()).unwrap_or(8) as u32;
            self.searcher.search(pos, depth.clamp(1, 60))
        };
        let pv: Vec<String> = r.pv.iter().map(|&m| square_to_string(m)).collect();
        json!({
            "best_move": r.best_move.map(square_to_string),
            "value": r.value,
            "pv": pv,
            "nodes": r.nodes,
            "depth": r.depth,
            "elapsed_ms": start.elapsed().as_millis() as u64,
            "solver": false,
        })
    }

    // Whole-iteration iterative deepening under a wall-clock budget.
    fn time_limited(&mut self, pos: &Position, budget: Duration) -> SearchResult {
        let start = Instant::now();
        let mut last = self.searcher.search(pos, 1);
        let mut depth = 2u32;
        while depth <= 60 && start.elapsed() < budget {
            let r = self.searcher.search(pos, depth);
            last = r;
            depth += 1;
        }
        last
    }

    // Per-legal-move value from the side to move's perspective, using a
    // child search one ply shallower, so that the best hint normally
    // agrees with analyze at the same depth.
    fn hints(&mut self, pos: &Position, req: &Value) -> Value {
        let depth = req
            .get("depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.hint_depth as u64) as u32;
        let depth = depth.clamp(1, 60);
        let mut hints = Vec::new();
        let mut m = pos.legal_moves();
        while m != 0 {
            let sq = m.trailing_zeros() as u8;
            m &= m - 1;
            let child = pos.play(sq).expect("legal move");
            // Value of playing sq, from the mover's perspective.
            let v = if child.is_game_over() {
                -child.final_score()
            } else {
                -self.searcher.search(&child, depth.saturating_sub(1).max(1)).value
            };
            hints.push(json!({"move": square_to_string(sq), "value": v}));
        }
        json!({ "hints": hints })
    }
}

fn handle(engine: &mut Engine, req: &Value) -> Value {
    let cmd = req.get("cmd").and_then(|v| v.as_str()).unwrap_or("");
    let result: Result<Value, String> = (|| match cmd {
        "ping" => Ok(json!({"pong": true})),
        "state" => Ok(state_value(&position_from_req(req)?)),
        "play" => {
            let pos = position_from_req(req)?;
            let mv = req.get("move").and_then(|v| v.as_str()).ok_or("missing move")?;
            let next = if mv.eq_ignore_ascii_case("pass") || mv == "--" {
                if !pos.must_pass() {
                    return Err("pass is only legal when there is no legal move".to_string());
                }
                pos.pass()
            } else {
                let sq = square_from_str(mv).ok_or(format!("bad move {:?}", mv))?;
                pos.play(sq).map_err(|_| format!("illegal move {}", mv))?
            };
            Ok(state_value(&next))
        }
        "analyze" => {
            let pos = position_from_req(req)?;
            Ok(engine.analyze(&pos, req))
        }
        "hints" => {
            let pos = position_from_req(req)?;
            Ok(engine.hints(&pos, req))
        }
        other => Err(format!("unknown cmd {:?}", other)),
    })();
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    match result {
        Ok(mut v) => {
            if let Value::Object(map) = &mut v {
                map.insert("id".to_string(), id);
                map.insert("ok".to_string(), Value::Bool(true));
            }
            v
        }
        Err(e) => json!({"id": id, "ok": false, "error": e}),
    }
}

pub fn run(cfg: ServeConfig) -> i32 {
    let mut engine = match Engine::new(&cfg) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("serve: engine init failed: {}", e);
            return 1;
        }
    };
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    // Signal readiness so the backend can wait for a warm engine.
    let _ = writeln!(stdout, "{}", json!({"ready": true}));
    let _ = stdout.flush();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<Value>(trimmed) {
            Ok(req) => handle(&mut engine, &req),
            Err(e) => json!({"ok": false, "error": format!("bad json: {}", e)}),
        };
        if writeln!(stdout, "{}", resp).is_err() {
            break;
        }
        if stdout.flush().is_err() {
            break;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        Engine::new(&ServeConfig {
            weights: concat!(env!("CARGO_MANIFEST_DIR"), "/../models/pattern_v2.bin").to_string(),
            mpc: None,
            mpc_t: 2.0,
            tt_mb: 16,
            hint_depth: 4,
            solve_at: 20,
            solve_ms: 3000,
        })
        .expect("engine")
    }

    #[test]
    fn analyze_and_hints_sign_consistent() {
        let mut e = engine();
        let pos = Position::initial();
        let req = json!({"board": board_str(&pos), "stm": "X", "depth": 6});
        let a = e.analyze(&pos, &req);
        let a_val = a["value"].as_i64().unwrap();
        let a_best = a["best_move"].as_str().unwrap().to_string();

        let h = e.hints(&pos, &json!({"board": board_str(&pos), "stm": "X", "depth": 6}));
        let hints = h["hints"].as_array().unwrap();
        // Best hint value equals the analyze value at the same depth.
        let max_hint = hints
            .iter()
            .map(|x| x["value"].as_i64().unwrap())
            .max()
            .unwrap();
        assert_eq!(max_hint, a_val, "max hint value must equal analyze value");
        // The move achieving the max hint is the analyzed best move's value.
        let best_hint_val = hints
            .iter()
            .find(|x| x["move"].as_str().unwrap() == a_best)
            .unwrap()["value"]
            .as_i64()
            .unwrap();
        assert_eq!(best_hint_val, a_val);
    }

    #[test]
    fn playing_best_move_flips_sign() {
        // After playing the best move, the opponent is to move; the
        // hint value for that move (mover's view) should equal the
        // negation of analyzing the child (opponent's view) at the
        // matching depth.
        let mut e = engine();
        let pos = Position::initial();
        let depth = 6u32;
        let h = e.hints(&pos, &json!({"board": board_str(&pos), "stm": "X", "depth": depth}));
        let hints = h["hints"].as_array().unwrap();
        let first = &hints[0];
        let mv = square_from_str(first["move"].as_str().unwrap()).unwrap();
        let hint_val = first["value"].as_i64().unwrap();
        let child = pos.play(mv).unwrap();
        let child_req = json!({"board": board_str(&child), "stm": stm_char(child.stm), "depth": depth - 1});
        let child_val = e.analyze(&child, &child_req)["value"].as_i64().unwrap();
        assert_eq!(hint_val, -child_val, "hint value must be negated child analysis");
    }

    #[test]
    fn state_reports_initial_legal_moves() {
        let mut e = engine();
        let pos = Position::initial();
        let s = handle(&mut e, &json!({"cmd": "state", "board": board_str(&pos), "stm": "X"}));
        assert_eq!(s["ok"], Value::Bool(true));
        let legal: Vec<String> = s["legal"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let mut sorted = legal.clone();
        sorted.sort();
        assert_eq!(sorted, vec!["c4", "d3", "e6", "f5"]);
    }

    #[test]
    fn play_rejects_illegal_and_accepts_legal() {
        let mut e = engine();
        let pos = Position::initial();
        let ok = handle(
            &mut e,
            &json!({"cmd": "play", "board": board_str(&pos), "stm": "X", "move": "d3"}),
        );
        assert_eq!(ok["ok"], Value::Bool(true));
        assert_eq!(ok["stm"], Value::from("O"));
        let bad = handle(
            &mut e,
            &json!({"cmd": "play", "board": board_str(&pos), "stm": "X", "move": "a1"}),
        );
        assert_eq!(bad["ok"], Value::Bool(false));
    }
}

#[cfg(test)]
mod solver_path_tests {
    use super::*;

    fn cfg() -> ServeConfig {
        ServeConfig {
            weights: concat!(env!("CARGO_MANIFEST_DIR"), "/../models/pattern_v5.bin").to_string(),
            mpc: None,
            mpc_t: 2.0,
            tt_mb: 8,
            hint_depth: 4,
            solve_at: 20,
            solve_ms: 60_000,
        }
    }

    /// Walk a random game to a low empty count and check that the server
    /// answers from the exact solver rather than the depth-limited
    /// search.
    ///
    /// A unit test of the solver alone cannot show that the play path
    /// actually calls it; this test goes through the path a caller uses.
    #[test]
    fn serve_reaches_the_solver_in_the_endgame() {
        if std::fs::metadata(&cfg().weights).is_err() {
            eprintln!("weights not present; skipping");
            return;
        }
        let mut engine = Engine::new(&cfg()).expect("engine");
        let mut rng = versor_core::rng::Xorshift64Star::new(7);
        let mut pos = Position::initial();
        // 46 discs placed leaves 18 empties, inside the solve threshold.
        while pos.empty().count_ones() > 18 {
            if pos.is_game_over() {
                panic!("game ended before reaching the endgame");
            }
            if pos.must_pass() {
                pos = pos.pass();
                continue;
            }
            let moves = pos.legal_moves();
            let sq = versor_core::bitboard::nth_set_bit(moves, rng.below(moves.count_ones()));
            pos = pos.play_unchecked(sq);
        }
        let req = json!({});
        let out = engine.analyze(&pos, &req);
        assert_eq!(
            out["solver"], json!(true),
            "the endgame was answered by the depth-limited search, not the solver"
        );
        assert!(out["best_move"].is_string(), "no move returned");
    }

    /// Above the threshold the search answers, so the switch is a switch
    /// and not an unconditional redirect.
    #[test]
    fn serve_uses_the_search_in_the_midgame() {
        if std::fs::metadata(&cfg().weights).is_err() {
            eprintln!("weights not present; skipping");
            return;
        }
        let mut engine = Engine::new(&cfg()).expect("engine");
        let out = engine.analyze(&Position::initial(), &json!({"depth": 4}));
        assert_eq!(out["solver"], json!(false));
    }
}

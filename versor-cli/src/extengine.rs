// Match driver for external engines. Drives a GPL engine binary (Edax,
// Egaroucid) as an external subprocess opponent. Only binaries are
// executed; no engine source is linked or copied. The protocol was
// learned from each binary's -h output and behavioral probing.
//
// Engines are described by an EngineSpec (binary path plus a launch
// argument template with a "{level}" placeholder), loaded from
// config/engines.json. The console dialogue is the Edax family style:
// "setboard <64 board chars><turn char>" then "go", and the engine
// answers with a line naming its move (e.g. "Edax plays F4"). The move
// parser is tolerant: it accepts "<name> plays <sq>", a GTP-style
// "= <sq>", or any lone a1..h8 token on the reply line, so Egaroucid's
// Edax-compatible mode works without engine-specific code.
//
// The board is re-synced with setboard before every engine move, so no
// engine-side state is trusted; versor-core is the single source of
// truth and every engine move is legality-checked. Games with illegal
// moves, timeouts or process errors are voided and counted. An engine
// that cannot be started stops the match, and run_match returns the
// reason with the engine's path.
//
// Every game is played by a newly started engine process and by a
// versor whose search state has been reset (`Searcher::new_game`), so a
// game does not depend on the games played before it by the same
// worker. A match therefore gives the same games however many are
// played at once (in fixed-depth play; under a clock the time each move
// gets still depends on the machine).

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use versor_core::notation::{move_seq_to_string, square_from_str};
use versor_core::{Color, Position};
use versor_eval::PatternEval;
use versor_search::{Evaluator, MpcParams, Searcher, Solver};

/// How to tell an engine how much time it has left.
///
/// Both supported engines take a whole-game clock per player rather
/// than a per-move cap, so the driver keeps the clock itself and hands
/// the engine its true remaining time before every move.
#[derive(Clone)]
pub struct ClockSpec {
    /// Command template. "{secs}" is the remaining whole seconds and
    /// "{color}" is B or W, the engine's own colour.
    pub template: String,
    /// Whether the command has to go out before `setboard`.
    ///
    /// This is not cosmetic. **Edax resets its clock whenever it is
    /// given a position**, so `game-time` sent after `setboard` has no
    /// effect and the engine plays every move with a full clock; sent
    /// before, it is the budget for the rest of the game. Measured: with
    /// `-t 30`, the same position re-sent six times took
    /// 1.96/1.96/2.05/1.92/2.19/2.07 s, i.e. the clock never ran down.
    /// Egaroucid's `settime` is independent of `setboard` and can go
    /// either side.
    pub before_setboard: bool,
    /// Tell the engine its remaining time less `engine_clock_margin`
    /// (the default). Without it the engine is told its true remaining
    /// time; only for showing what the margin is for.
    pub margin: bool,
}

/// Told-to-the-engine time is its real remaining time less this, to
/// absorb the gap between the engine's own accounting and the driver's.
///
/// The driver's clock runs from just before the position is sent to
/// just after the move comes back, so it also charges the engine for
/// pipe writes, its own reply parsing and any scheduling delay. The
/// engine budgets against what it thinks it has and, at least in Edax's
/// case, spends nearly all of it, so without a margin it will
/// occasionally overrun the driver's clock and lose on time for reasons
/// that have nothing to do with play. Measured: with a 20 s clock and no
/// margin, Edax flagged in 1 game out of 2.
fn engine_clock_margin(remaining: Duration) -> Duration {
    let one_percent = remaining / 100;
    let floor = Duration::from_secs(1);
    if one_percent > floor {
        one_percent
    } else {
        floor
    }
}

impl ClockSpec {
    fn render(&self, remaining: Duration, color: Color) -> String {
        let c = if color == Color::Black { "B" } else { "W" };
        let told = if self.margin {
            remaining.saturating_sub(engine_clock_margin(remaining))
        } else {
            remaining
        };
        // Whole seconds, rounded down: never claim more time than there
        // actually is.
        self.template
            .replace("{secs}", &told.as_secs().to_string())
            .replace("{color}", c)
    }
}

#[derive(Clone)]
pub struct EngineSpec {
    pub name: String,
    pub binary: String,
    /// Launch arguments; any "{level}" token is replaced by the level.
    pub args: Vec<String>,
    pub level: u32,
    pub reply_timeout: Duration,
    /// Present only for time-control play.
    pub clock: Option<ClockSpec>,
}

impl EngineSpec {
    fn resolved_args(&self) -> Vec<String> {
        self.args
            .iter()
            .map(|a| a.replace("{level}", &self.level.to_string()))
            .collect()
    }
}

// Extract a move square from an engine reply line. Handles "Edax plays
// F4", "Egaroucid plays f4", GTP "= f4", or a bare coordinate token.
fn parse_move_line(line: &str) -> Option<u8> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let after_plays = line.rsplit_once(" plays ").map(|(_, m)| m);
    let after_eq = line.strip_prefix("= ");
    let candidates: Vec<&str> = match (after_plays, after_eq) {
        (Some(m), _) => vec![m],
        (None, Some(m)) => vec![m],
        (None, None) => line.split_whitespace().collect(),
    };
    for tok in candidates {
        let t: String = tok
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect();
        if t.len() == 2 {
            if let Some(sq) = square_from_str(&t.to_ascii_lowercase()) {
                return Some(sq);
            }
        }
    }
    None
}

pub struct EngineChild {
    child: Child,
    rx: Receiver<String>,
}

impl EngineChild {
    pub fn spawn(cfg: &EngineSpec) -> Result<EngineChild, String> {
        let bin = std::path::Path::new(&cfg.binary);
        let dir = bin.parent().ok_or_else(|| {
            format!(
                "cannot start engine {} ({}): no parent directory",
                cfg.name, cfg.binary
            )
        })?;
        let mut child = Command::new(bin)
            .current_dir(dir)
            .args(cfg.resolved_args())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot start engine {} ({}): {}", cfg.name, cfg.binary, e))?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(EngineChild { child, rx })
    }

    fn send(&mut self, cmd: &str) -> Result<(), String> {
        let stdin = self.child.stdin.as_mut().ok_or("no stdin")?;
        stdin
            .write_all(cmd.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("engine stdin: {}", e))
    }

    /// Ask the engine for its move in `pos` (side to move must have a
    /// legal move) and return the square index.
    ///
    /// With `clock` set, the engine is first told how much time it has
    /// left in the game: its clock spec, its remaining time and its own
    /// colour. **The order of the three commands matters**; see
    /// `ClockSpec`.
    pub fn best_move_timed(
        &mut self,
        pos: &Position,
        timeout: Duration,
        clock: Option<(&ClockSpec, Duration, Color)>,
    ) -> Result<u8, String> {
        let flat: String = pos
            .to_text()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        // Drain any stale output before issuing the request.
        while self.rx.try_recv().is_ok() {}
        if let Some((spec, remaining, color)) = clock {
            if spec.before_setboard {
                self.send(&spec.render(remaining, color))?;
            }
        }
        self.send(&format!("setboard {}", flat))?;
        if let Some((spec, remaining, color)) = clock {
            if !spec.before_setboard {
                self.send(&spec.render(remaining, color))?;
            }
        }
        self.send("go")?;
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let left = deadline
                .checked_duration_since(std::time::Instant::now())
                .ok_or("engine reply timeout")?;
            match self.rx.recv_timeout(left) {
                Ok(line) => {
                    if let Some(sq) = parse_move_line(&line) {
                        return Ok(sq);
                    }
                }
                Err(RecvTimeoutError::Timeout) => return Err("engine reply timeout".into()),
                Err(RecvTimeoutError::Disconnected) => return Err("engine died".into()),
            }
        }
    }

    pub fn shutdown(&mut self) {
        let _ = self.send("quit");
        std::thread::sleep(Duration::from_millis(100));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct GameOutcome {
    /// Score from versor's perspective, or None if the game was voided.
    pub versor_score: Option<i32>,
    pub transcript: Vec<u8>,
    /// Moves versor decided with the exact endgame solver.
    pub solver_moves: u32,
    /// Time-control play only: who ran out of time, if anyone. A game
    /// lost on time is a loss, not a voided game: running the clock out
    /// is a way of losing, and voiding it would hide a broken time
    /// manager instead of showing it.
    pub versor_flagged: bool,
    pub engine_flagged: bool,
    /// Wall-clock actually consumed by each side, and how many moves
    /// each played. Reported so that a time manager that never uses its
    /// clock, or one that burns it in the opening, is visible.
    pub versor_time: Duration,
    pub engine_time: Duration,
    pub versor_moves: u32,
    pub engine_moves: u32,
    /// What versor's own midgame searches managed, over the moves it
    /// decided with the depth-limited search (the solver's moves are
    /// counted separately). Reported so that two runs of the same
    /// games can be compared for how deep each got in the same clock:
    /// that is the thing a busy machine takes away.
    pub versor_search_moves: u32,
    pub versor_depth_sum: u64,
    pub versor_nodes: u64,
    /// Nodes of all of versor's searches and solves in the game, in
    /// every mode (the game log carries it).
    pub nodes_all: u64,
}

/// Time control for a game: a whole-game clock per player.
#[derive(Clone, Copy, Debug)]
pub struct TimeControl {
    /// Per player, for all their moves in the game.
    pub game_time: Duration,
    /// How many moves' worth of the clock the endgame solver may spend
    /// on one call before the depth-limited search takes over.
    pub solver_factor: u32,
    /// Divide the clock over every move left in the game (BILL's scheme)
    /// instead of over the midgame moves only. For comparisons; play
    /// uses the midgame division.
    pub all_moves: bool,
}

/// Play one game from `opening`; versor takes `versor_color`.
///
/// versor switches to the exact endgame solver once the empty count
/// reaches `solve_at`, which is what the opposing engines do at their
/// configured levels, so both sides solve the endgame exactly. With
/// `solve_at` = 0 versor plays its endgame with the depth-limited search
/// while the opponent still solves it.
#[allow(clippy::too_many_arguments)]
pub fn play_game<E: Evaluator>(
    opening: &Position,
    opening_moves: &[u8],
    searcher: &mut Searcher<E>,
    solver: &mut Solver,
    depth: u32,
    solve_at: u32,
    engine: &mut EngineChild,
    timeout: Duration,
    versor_color: Color,
    tc: Option<TimeControl>,
    clock_spec: Option<&ClockSpec>,
) -> GameOutcome {
    searcher.new_game();
    solver.new_game();
    let mut pos = *opening;
    let mut transcript: Vec<u8> = opening_moves.to_vec();
    let mut solver_moves = 0u32;
    let mut versor_left = tc.map_or(Duration::ZERO, |t| t.game_time);
    let mut engine_left = versor_left;
    let mut versor_used = Duration::ZERO;
    let mut engine_used = Duration::ZERO;
    let mut versor_moves = 0u32;
    let mut engine_moves = 0u32;
    let mut search_moves = 0u32;
    let mut depth_sum = 0u64;
    let mut node_sum = 0u64;
    let mut nodes_all = 0u64;
    let void = |transcript: Vec<u8>, solver_moves, versor_used, engine_used, vm, em, nodes_all| GameOutcome {
        versor_score: None,
        transcript,
        solver_moves,
        versor_flagged: false,
        engine_flagged: false,
        versor_time: versor_used,
        engine_time: engine_used,
        versor_moves: vm,
        engine_moves: em,
        versor_search_moves: 0,
        versor_depth_sum: 0,
        versor_nodes: 0,
        nodes_all,
    };
    while !pos.is_game_over() {
        if pos.must_pass() {
            pos = pos.pass();
            continue;
        }
        let sq = if pos.stm == versor_color {
            let empties = pos.empty().count_ones();
            let start = Instant::now();
            let chosen = match tc {
                None => {
                    let solved = if solve_at > 0 && empties <= solve_at {
                        solver.solve_exact(&pos, None).and_then(|r| {
                            nodes_all += r.nodes;
                            r.best_move
                        })
                    } else {
                        None
                    };
                    match solved {
                        Some(m) => {
                            solver_moves += 1;
                            Some(m)
                        }
                        None => {
                            let r = searcher.search(&pos, depth);
                            nodes_all += r.nodes;
                            r.best_move
                        }
                    }
                }
                Some(t) => {
                    // Under a clock the solver gets several moves' worth
                    // of time, because one solve settles the rest of the
                    // game. If it does not finish in that, the
                    // depth-limited search answers instead.
                    let solved = if solve_at > 0 && empties <= solve_at {
                        let limit =
                            versor_search::solver_budget(versor_left, empties, t.solver_factor);
                        solver.solve_exact(&pos, Some(limit)).and_then(|r| {
                            nodes_all += r.nodes;
                            r.best_move
                        })
                    } else {
                        None
                    };
                    match solved {
                        Some(m) => {
                            solver_moves += 1;
                            Some(m)
                        }
                        None => {
                            let left = versor_left.saturating_sub(start.elapsed());
                            let budget = if t.all_moves {
                                versor_search::budget_for_move_all_moves(left, empties)
                            } else {
                                versor_search::budget_for_move(left, empties, solve_at)
                            };
                            let r = searcher.search_timed(&pos, budget);
                            search_moves += 1;
                            depth_sum += u64::from(r.depth);
                            node_sum += r.nodes;
                            nodes_all += r.nodes;
                            r.best_move
                        }
                    }
                }
            };
            if tc.is_some() {
                let took = start.elapsed();
                versor_used += took;
                versor_moves += 1;
                if took >= versor_left {
                    return GameOutcome {
                        versor_score: Some(-64),
                        transcript,
                        solver_moves,
                        versor_flagged: true,
                        engine_flagged: false,
                        versor_time: versor_used,
                        engine_time: engine_used,
                        versor_moves,
                        engine_moves,
                        versor_search_moves: search_moves,
                        versor_depth_sum: depth_sum,
                        versor_nodes: node_sum,
                        nodes_all,
                    };
                }
                versor_left -= took;
            }
            match chosen {
                Some(m) => m,
                None => {
                    return void(
                        transcript,
                        solver_moves,
                        versor_used,
                        engine_used,
                        versor_moves,
                        engine_moves,
                        nodes_all,
                    )
                }
            }
        } else {
            let start = Instant::now();
            // Give the engine its true remaining time, then wait for it
            // a good deal longer than that before calling it dead.
            let (clock, reply_timeout) = match (tc, clock_spec) {
                (Some(_), Some(spec)) => (
                    Some((spec, engine_left, pos.stm)),
                    engine_left + Duration::from_secs(60),
                ),
                _ => (None, timeout),
            };
            let got = engine.best_move_timed(&pos, reply_timeout, clock);
            if tc.is_some() {
                let took = start.elapsed();
                engine_used += took;
                engine_moves += 1;
                if took >= engine_left {
                    return GameOutcome {
                        versor_score: Some(64),
                        transcript,
                        solver_moves,
                        versor_flagged: false,
                        engine_flagged: true,
                        versor_time: versor_used,
                        engine_time: engine_used,
                        versor_moves,
                        engine_moves,
                        versor_search_moves: search_moves,
                        versor_depth_sum: depth_sum,
                        versor_nodes: node_sum,
                        nodes_all,
                    };
                }
                engine_left -= took;
            }
            match got {
                Ok(m) => m,
                Err(_) => {
                    return void(
                        transcript,
                        solver_moves,
                        versor_used,
                        engine_used,
                        versor_moves,
                        engine_moves,
                        nodes_all,
                    )
                }
            }
        };
        match pos.play(sq) {
            Ok(next) => {
                transcript.push(sq);
                pos = next;
            }
            Err(_) => {
                // Illegal move (necessarily from the engine): void the game.
                return void(
                    transcript,
                    solver_moves,
                    versor_used,
                    engine_used,
                    versor_moves,
                    engine_moves,
                    nodes_all,
                );
            }
        }
    }
    let score_black = match pos.stm {
        Color::Black => pos.final_score(),
        Color::White => -pos.final_score(),
    };
    let versor_score = match versor_color {
        Color::Black => score_black,
        Color::White => -score_black,
    };
    GameOutcome {
        versor_score: Some(versor_score),
        transcript,
        solver_moves,
        versor_flagged: false,
        engine_flagged: false,
        versor_time: versor_used,
        engine_time: engine_used,
        versor_moves,
        engine_moves,
        versor_search_moves: search_moves,
        versor_depth_sum: depth_sum,
        versor_nodes: node_sum,
        nodes_all,
    }
}

pub struct MatchParams {
    pub weights: String,
    pub games: u64,
    pub depth: u32,
    pub threads: usize,
    pub seed: u64,
    pub opening_plies: u32,
    pub tt_mb: usize,
    pub engine: EngineSpec,
    pub log_path: Option<String>,
    /// Early-stop reference rule: if after `probe_games` the score rate
    /// is below `probe_rate`, stop at `stop_games` and mark as
    /// reference-only.
    pub probe_games: u64,
    pub probe_rate: f64,
    pub stop_games: u64,
    /// Empty count at or below which versor switches to the exact
    /// endgame solver, as the opposing engines do. 0 disables the
    /// switch.
    pub solve_at: u32,
    /// Wall-clock time control. When set, `depth` is not used: both
    /// sides play to a whole-game clock instead.
    pub time_control: Option<TimeControl>,
    /// Multi-ProbCut for versor's midgame search: the coefficient file
    /// and the confidence level t. None plays without it.
    pub mpc: Option<(String, f64)>,
}

pub struct MatchResult {
    pub wins: u64,
    pub draws: u64,
    pub losses: u64,
    pub voided: u64,
    pub pair_scores: Vec<f64>,
    pub reference_only: bool,
    /// Total moves versor decided with the exact solver, across all
    /// games. Evidence that the solver was actually on the path.
    pub solver_moves: u64,
    /// Time-control play only.
    pub versor_flags: u64,
    pub engine_flags: u64,
    pub versor_time_ms: u64,
    pub engine_time_ms: u64,
    pub versor_move_count: u64,
    pub engine_move_count: u64,
    pub versor_search_moves: u64,
    pub versor_depth_sum: u64,
    pub versor_nodes: u64,
}

pub fn run_match(
    p: &MatchParams,
    opening_fn: impl Fn(u64, u64, u32) -> (Position, Vec<u8>) + Sync,
) -> Result<MatchResult, String> {
    PatternEval::from_file(&p.weights)?;
    let pairs = p.games.div_ceil(2);
    let next_pair = AtomicU64::new(0);
    let stop_flag = std::sync::atomic::AtomicBool::new(false);
    // The early-stop rule looks at the first `probe_games` and the first
    // `stop_games` games by game number, once all of them are in, never
    // at whichever games happened to finish first.
    let probe_n = p.probe_games.min(p.games);
    let stop_n = p.stop_games.min(p.games);
    let state = Mutex::new(Tally {
        games: (0..p.games).map(|_| None).collect(),
        done_below_probe: 0,
        done_below_stop: 0,
        reference_only: false,
        cutoff: p.games,
    });
    let log = match &p.log_path {
        Some(path) => Some(Mutex::new(
            std::fs::File::create(path).map_err(|e| format!("{}: {}", path, e))?,
        )),
        None => None,
    };
    // The first failure to start the engine (a wrong path in the config,
    // a file that is not executable, ...). It stops the match, and is
    // returned instead of a result.
    let start_error = Mutex::new(None::<String>);
    let fail_start = |e: String| {
        start_error.lock().unwrap().get_or_insert(e);
        stop_flag.store(true, Ordering::Relaxed);
    };

    std::thread::scope(|scope| {
        for _ in 0..p.threads {
            scope.spawn(|| {
                let eval = PatternEval::from_file(&p.weights).expect("weights");
                let mut searcher = Searcher::new(eval, p.tt_mb);
                if let Some((path, t)) = &p.mpc {
                    searcher.set_mpc(MpcParams::from_file(path).expect("mpc params"), *t);
                }
                let mut solver = Solver::new(p.tt_mb);
                if p.solve_at > 0 {
                    solver.set_eval_order(Box::new(
                        PatternEval::from_file(&p.weights).expect("weights"),
                    ));
                }
                loop {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    let k = next_pair.fetch_add(1, Ordering::Relaxed);
                    if k >= pairs {
                        break;
                    }
                    let (opening, omoves) = opening_fn(p.seed, k, p.opening_plies);
                    for vcolor in [Color::Black, Color::White] {
                        let g = k * 2 + u64::from(vcolor == Color::White);
                        if g >= p.games {
                            break;
                        }
                        // A newly started engine for every game: whatever
                        // the engine keeps from one game to the next (its
                        // hash table, for one) is gone, so the game does
                        // not depend on what the same process played
                        // before.
                        let mut engine = match EngineChild::spawn(&p.engine) {
                            Ok(c) => c,
                            Err(e) => return fail_start(e),
                        };
                        let out = play_game(
                            &opening,
                            &omoves,
                            &mut searcher,
                            &mut solver,
                            p.depth,
                            p.solve_at,
                            &mut engine,
                            p.engine.reply_timeout,
                            vcolor,
                            p.time_control,
                            p.engine.clock.as_ref(),
                        );
                        engine.shutdown();
                        if let (Some(log), Some(s)) = (&log, out.versor_score) {
                            let vc = if vcolor == Color::Black { 'B' } else { 'W' };
                            let line = format!(
                                "{} {} {:+} {} {}\n",
                                k,
                                vc,
                                s,
                                move_seq_to_string(&out.transcript),
                                out.nodes_all
                            );
                            let _ = log.lock().unwrap().write_all(line.as_bytes());
                        }
                        let mut st = state.lock().unwrap();
                        st.games[g as usize] = Some(out);
                        if g < probe_n {
                            st.done_below_probe += 1;
                            if st.done_below_probe == probe_n
                                && st.score_rate(probe_n) < p.probe_rate
                            {
                                st.reference_only = true;
                            }
                        }
                        if g < stop_n {
                            st.done_below_stop += 1;
                            if st.done_below_stop == stop_n
                                && stop_n < p.games
                                && st.score_rate(stop_n) < p.probe_rate
                            {
                                // Hopeless: count the first stop_n games
                                // only, and start no more.
                                st.reference_only = true;
                                st.cutoff = stop_n;
                                stop_flag.store(true, Ordering::Relaxed);
                            }
                        }
                    }
                }
            });
        }
    });

    if let Some(e) = start_error.into_inner().unwrap() {
        return Err(e);
    }
    let st = state.into_inner().unwrap();
    Ok(st.result(p.time_control.is_some()))
}

/// The games of a match by game number (pair k has games 2k, versor
/// black, and 2k+1), and the early-stop bookkeeping.
struct Tally {
    games: Vec<Option<GameOutcome>>,
    done_below_probe: u64,
    done_below_stop: u64,
    reference_only: bool,
    /// Games from this number on are not counted (an early stop).
    cutoff: u64,
}

impl Tally {
    /// Versor's score rate over the first `n` games, voided ones left out.
    fn score_rate(&self, n: u64) -> f64 {
        let (mut x2, mut played) = (0u64, 0u64);
        for g in self.games[..n as usize].iter().flatten() {
            if let Some(s) = g.versor_score {
                x2 += match s.cmp(&0) {
                    std::cmp::Ordering::Greater => 2,
                    std::cmp::Ordering::Equal => 1,
                    std::cmp::Ordering::Less => 0,
                };
                played += 1;
            }
        }
        if played == 0 {
            1.0
        } else {
            x2 as f64 / (2 * played) as f64
        }
    }

    /// The match result from the counted games, in game-number order, so
    /// that it is the same however the games were shared out.
    fn result(&self, timed: bool) -> MatchResult {
        let mut r = MatchResult {
            wins: 0,
            draws: 0,
            losses: 0,
            voided: 0,
            pair_scores: Vec::new(),
            reference_only: self.reference_only,
            solver_moves: 0,
            versor_flags: 0,
            engine_flags: 0,
            versor_time_ms: 0,
            engine_time_ms: 0,
            versor_move_count: 0,
            engine_move_count: 0,
            versor_search_moves: 0,
            versor_depth_sum: 0,
            versor_nodes: 0,
        };
        let counted = &self.games[..self.cutoff as usize];
        for pair in counted.chunks(2) {
            let mut sum = 0.0;
            let mut n = 0u64;
            for out in pair.iter().flatten() {
                r.solver_moves += u64::from(out.solver_moves);
                if timed {
                    r.versor_flags += u64::from(out.versor_flagged);
                    r.engine_flags += u64::from(out.engine_flagged);
                    r.versor_time_ms += out.versor_time.as_millis() as u64;
                    r.engine_time_ms += out.engine_time.as_millis() as u64;
                    r.versor_move_count += u64::from(out.versor_moves);
                    r.engine_move_count += u64::from(out.engine_moves);
                    r.versor_search_moves += u64::from(out.versor_search_moves);
                    r.versor_depth_sum += out.versor_depth_sum;
                    r.versor_nodes += out.versor_nodes;
                }
                match out.versor_score {
                    None => r.voided += 1,
                    Some(s) => {
                        sum += match s.cmp(&0) {
                            std::cmp::Ordering::Greater => {
                                r.wins += 1;
                                1.0
                            }
                            std::cmp::Ordering::Equal => {
                                r.draws += 1;
                                0.5
                            }
                            std::cmp::Ordering::Less => {
                                r.losses += 1;
                                0.0
                            }
                        };
                        n += 1;
                    }
                }
            }
            if n > 0 {
                r.pair_scores.push(sum / n as f64);
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEIGHTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/pattern_v2.bin");
    const MISSING: &str = "/nonexistent/versor-test/engine";

    fn spec(binary: &str) -> EngineSpec {
        EngineSpec {
            name: "test".to_string(),
            binary: binary.to_string(),
            args: Vec::new(),
            level: 1,
            reply_timeout: Duration::from_secs(10),
            clock: None,
        }
    }

    fn params(binary: &str, games: u64) -> MatchParams {
        MatchParams {
            weights: WEIGHTS.to_string(),
            games,
            depth: 1,
            threads: 2,
            seed: 1,
            opening_plies: 0,
            tt_mb: 1,
            engine: spec(binary),
            log_path: None,
            probe_games: 200,
            probe_rate: 0.05,
            stop_games: 500,
            solve_at: 0,
            time_control: None,
            mpc: None,
        }
    }

    fn initial(_: u64, _: u64, _: u32) -> (Position, Vec<u8>) {
        (Position::initial(), Vec::new())
    }

    #[test]
    fn spawn_error_names_the_engine_and_the_path() {
        let err = match EngineChild::spawn(&spec(MISSING)) {
            Ok(_) => panic!("{} started", MISSING),
            Err(e) => e,
        };
        assert!(err.contains("engine test"), "{}", err);
        assert!(err.contains(MISSING), "{}", err);
    }

    #[test]
    fn match_stops_when_the_engine_does_not_start() {
        if std::fs::metadata(WEIGHTS).is_err() {
            eprintln!("weights not present; skipping");
            return;
        }
        match run_match(&params(MISSING, 4), initial) {
            Ok(r) => panic!(
                "no error; {} games, {} voided",
                r.wins + r.draws + r.losses,
                r.voided
            ),
            Err(e) => assert!(e.contains(MISSING), "{}", e),
        }
    }

    /// An engine that starts but answers every move request with the
    /// illegal a1: the match runs to the end and every game is voided,
    /// so a working engine is not taken for one that does not start.
    #[cfg(unix)]
    #[test]
    fn match_runs_when_the_engine_starts() {
        use std::os::unix::fs::PermissionsExt;
        if std::fs::metadata(WEIGHTS).is_err() {
            eprintln!("weights not present; skipping");
            return;
        }
        let dir = std::env::temp_dir().join(format!("versor-test-engine-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("engine.sh");
        std::fs::write(
            &bin,
            "#!/bin/sh\nwhile read -r line; do\n  case \"$line\" in\n    go) echo \"= a1\" ;;\n    quit) exit 0 ;;\n  esac\ndone\n",
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let r = run_match(&params(bin.to_str().unwrap(), 4), initial);
        let _ = std::fs::remove_dir_all(&dir);
        let r = r.expect("the engine starts");
        assert_eq!(r.voided, 4);
        assert_eq!(r.wins + r.draws + r.losses, 0);
    }

    #[test]
    fn the_engine_is_told_its_time_less_the_margin_unless_turned_off() {
        let mut spec = ClockSpec {
            template: "game-time {color} {secs}".to_string(),
            before_setboard: true,
            margin: true,
        };
        let left = Duration::from_secs(20);
        assert_eq!(spec.render(left, Color::Black), "game-time B 19");
        spec.margin = false;
        assert_eq!(spec.render(left, Color::White), "game-time W 20");
    }

    fn flat(pos: &Position) -> String {
        pos.to_text().chars().filter(|c| !c.is_whitespace()).collect()
    }

    /// The engine's side of the two games of a pair from the initial
    /// position (versor black, then white), when versor searches to
    /// depth 1 from a fresh state and the engine plays its lowest legal
    /// square: each position the engine is handed, with its reply.
    fn scripted_replies() -> Vec<(String, u8)> {
        let mut replies = Vec::new();
        for vcolor in [Color::Black, Color::White] {
            let mut searcher = Searcher::new(PatternEval::from_file(WEIGHTS).unwrap(), 1);
            let mut pos = Position::initial();
            while !pos.is_game_over() {
                if pos.must_pass() {
                    pos = pos.pass();
                    continue;
                }
                let sq = if pos.stm == vcolor {
                    searcher.search(&pos, 1).best_move.unwrap()
                } else {
                    let sq = pos.legal_moves().trailing_zeros() as u8;
                    replies.push((flat(&pos), sq));
                    sq
                };
                pos = pos.play(sq).unwrap();
            }
        }
        replies
    }

    /// An engine that plays from a table and remembers what it has been
    /// handed: once a position comes in with no more discs than the one
    /// before, it is in its second game, and from then on it answers the
    /// illegal a1. A match that reuses an engine process for a second
    /// game therefore voids that game. Every start is logged.
    #[cfg(unix)]
    #[test]
    fn every_game_gets_a_newly_started_engine() {
        use std::os::unix::fs::PermissionsExt;
        if std::fs::metadata(WEIGHTS).is_err() {
            eprintln!("weights not present; skipping");
            return;
        }
        let dir = std::env::temp_dir().join(format!("versor-test-table-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let table: String = scripted_replies()
            .iter()
            .map(|(b, sq)| format!("{} {}\n", b, versor_core::notation::square_to_string(*sq)))
            .collect();
        std::fs::write(dir.join("table.txt"), table).unwrap();
        let bin = dir.join("engine.sh");
        std::fs::write(
            &bin,
            "#!/bin/sh\n\
             echo start >> starts.txt\n\
             last=0; game=0; board=\n\
             while read -r cmd arg; do\n\
             \x20 case \"$cmd\" in\n\
             \x20   setboard)\n\
             \x20     board=\"$arg\"\n\
             \x20     n=$(printf '%.64s' \"$board\" | tr -cd 'XO' | wc -c)\n\
             \x20     if [ \"$n\" -le \"$last\" ] || [ \"$game\" -eq 0 ]; then game=$((game + 1)); fi\n\
             \x20     last=$n ;;\n\
             \x20   go)\n\
             \x20     mv=$(grep -m1 \"^$board \" table.txt | cut -d' ' -f2)\n\
             \x20     if [ \"$game\" -gt 1 ] || [ -z \"$mv\" ]; then mv=a1; fi\n\
             \x20     echo \"= $mv\" ;;\n\
             \x20   quit) exit 0 ;;\n\
             \x20 esac\n\
             done\n",
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let log = dir.join("games.txt");
        let mut p = params(bin.to_str().unwrap(), 4);
        // One worker plays all four games, one after the other.
        p.threads = 1;
        p.log_path = Some(log.to_str().unwrap().to_string());
        let r = run_match(&p, initial).expect("the engine starts");
        let starts = std::fs::read_to_string(dir.join("starts.txt")).unwrap_or_default();
        let lines = std::fs::read_to_string(&log).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(r.voided, 0, "a game was played by an engine that had played before");
        assert_eq!(r.wins + r.draws + r.losses, 4);
        assert_eq!(starts.lines().count(), 4, "one engine start per game");
        // Both pairs start from the same position, so their games are the
        // same, move for move.
        let mut by_pair: Vec<Vec<&str>> = vec![Vec::new(), Vec::new()];
        for l in lines.lines() {
            let f: Vec<&str> = l.split(' ').collect();
            let k: usize = f[0].parse().unwrap();
            by_pair[k].push(l.split_once(' ').unwrap().1);
        }
        by_pair.iter_mut().for_each(|v| v.sort());
        assert_eq!(by_pair[0], by_pair[1]);
    }
}

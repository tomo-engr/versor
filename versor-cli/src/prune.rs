// Exactness checks and node-count comparisons for the two exact
// cutoffs: the stability cutoff and the enhanced transposition cutoff.
//
// Every figure produced here is a node count or a call count, never a
// wall-clock time, so the numbers are reproducible on a shared machine
// under arbitrary load.
//
// Both cutoffs are backward pruning: they only fire where the node was
// going to fail high or fail low anyway, so the value and the best move
// must be bit-identical to a search with them switched off. That is what
// the "exact" subcommands assert. On a mismatch the offending positions
// are printed as move strings so they can be replayed directly; the
// reference side is never adjusted to make the two agree.

use std::collections::HashMap;

use versor_core::notation::move_seq_to_string;
use versor_core::Position;
use versor_eval::PatternEval;
use versor_search::{Evaluator, PlaceholderEval, Searcher, Solver};

use versor_core::sample::positions_at_empties;

use crate::{expand_home, flag_num, fmt_move, split_flags};

/// One position where the pruned and unpruned searches disagreed.
struct Mismatch {
    moves: Vec<u8>,
    base_value: i32,
    base_move: Option<u8>,
    test_value: i32,
    test_move: Option<u8>,
}

fn report_mismatches(label: &str, checked: usize, mismatches: &[Mismatch]) -> bool {
    if mismatches.is_empty() {
        println!("  {label}: {checked} positions, 0 mismatches  OK");
        return true;
    }
    println!(
        "  {label}: {checked} positions, {} mismatches  FAIL",
        mismatches.len()
    );
    println!("  smallest mismatching positions (replay with: versor-cli search <depth> <moves>):");
    for m in mismatches.iter().take(3) {
        println!(
            "    moves {}  base value {} move {}  pruned value {} move {}",
            move_seq_to_string(&m.moves),
            m.base_value,
            fmt_move(m.base_move),
            m.test_value,
            fmt_move(m.test_move),
        );
    }
    false
}

// ---------------------------------------------------------------------
// Midgame.
//
// The stability cutoff bounds the FINAL score of the game. A depth
// limited midgame search returns a heuristic evaluation at its leaves,
// not a final score, so the midgame searcher applies the cutoff only
// where depth >= empties and every leaf is terminal. At the usual
// midgame settings (depth 8 with 34 empties) only ETC is active.
// ---------------------------------------------------------------------

fn exact_mid<E: Evaluator>(
    base: &mut Searcher<E>,
    test: &mut Searcher<E>,
    positions: &[(Position, Vec<u8>)],
    depth: u32,
) -> Vec<Mismatch> {
    let mut out = Vec::new();
    for (pos, moves) in positions {
        // A cleared table on both sides makes the comparison depend on
        // the pruning alone rather than on carried-over entries.
        base.new_game();
        test.new_game();
        let a = base.search(pos, depth);
        let b = test.search(pos, depth);
        if a.value != b.value || a.best_move != b.best_move {
            out.push(Mismatch {
                moves: moves.clone(),
                base_value: a.value,
                base_move: a.best_move,
                test_value: b.value,
                test_move: b.best_move,
            });
        }
    }
    out
}

fn setup_searcher<E: Evaluator>(s: &mut Searcher<E>, etc: bool, stab: bool, plain: bool) {
    s.opts.use_etc = etc;
    s.opts.use_stability = stab;
    if plain {
        s.opts.use_killers = false;
        s.opts.use_history = false;
        s.opts.use_aspiration = false;
    }
}

fn nodes_mid<E: Evaluator>(
    s: &mut Searcher<E>,
    positions: &[(Position, Vec<u8>)],
    depth: u32,
) -> (u64, u64, u64) {
    let mut nodes = 0u64;
    let mut calls = 0u64;
    let mut cuts = 0u64;
    for (pos, _) in positions {
        s.new_game();
        let r = s.search(pos, depth);
        nodes += r.nodes;
        let (c, k) = s.etc_stats();
        calls += c;
        cuts += k;
    }
    (nodes, calls, cuts)
}

// ---------------------------------------------------------------------
// Endgame solver: stability cutoff and ETC.
// ---------------------------------------------------------------------

struct SolverCfg {
    label: &'static str,
    stability: bool,
    etc: bool,
}

const SOLVER_CFGS: [SolverCfg; 4] = [
    SolverCfg { label: "off", stability: false, etc: false },
    SolverCfg { label: "stability", stability: true, etc: false },
    SolverCfg { label: "etc", stability: false, etc: true },
    SolverCfg { label: "both", stability: true, etc: true },
];

fn make_solver(tt_mb: usize, weights: Option<&str>, stability: bool, etc: bool) -> Solver {
    let mut s = Solver::new(tt_mb);
    s.opts.use_stability = stability;
    s.opts.use_etc = etc;
    if let Some(w) = weights {
        if let Ok(p) = PatternEval::from_file(w) {
            s.set_eval_order(Box::new(p));
        }
    }
    s
}

/// Solve every position once without any cutoff, then once per cutoff
/// configuration, and collect the disagreements. The unpruned solve is
/// the expensive one, so it is done a single time and reused across the
/// configurations rather than repeated for each.
fn exact_end_all(
    positions: &[(Position, Vec<u8>)],
    tt_mb: usize,
    weights: Option<&str>,
    configs: &[(&str, bool, bool)],
) -> Vec<Vec<Mismatch>> {
    let mut base = make_solver(tt_mb, weights, false, false);
    let mut tests: Vec<Solver> = configs
        .iter()
        .map(|(_, s, e)| make_solver(tt_mb, weights, *s, *e))
        .collect();
    let mut out: Vec<Vec<Mismatch>> = configs.iter().map(|_| Vec::new()).collect();
    for (pos, moves) in positions {
        base.new_game();
        let a = base.solve_exact(pos, None).expect("no time limit");
        for (i, t) in tests.iter_mut().enumerate() {
            t.new_game();
            let b = t.solve_exact(pos, None).expect("no time limit");
            if a.score != b.score || a.best_move != b.best_move {
                out[i].push(Mismatch {
                    moves: moves.clone(),
                    base_value: a.score,
                    base_move: a.best_move,
                    test_value: b.score,
                    test_move: b.best_move,
                });
            }
        }
    }
    out
}

pub fn cmd_prune(rest: &[String]) -> i32 {
    let (positional, flags) = match split_flags(rest) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let sub = positional.first().map(|s| s.as_str()).unwrap_or("all");
    let tt_mb: usize = match flag_num(&flags, "tt", 64usize) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let seed: u64 = match flag_num(&flags, "seed", 8080u64) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let weights = flags.get("weights").map(|w| expand_home(w));
    let weights = weights.as_deref();
    if let Some(w) = weights {
        if let Err(e) = PatternEval::from_file(w) {
            eprintln!("cannot load weights {w}: {e}");
            return 1;
        }
    }

    match sub {
        "exact-mid" => cmd_exact_mid(&flags, tt_mb, seed, weights),
        "exact-end" => cmd_exact_end(&flags, tt_mb, seed, weights),
        "nodes-mid" => cmd_nodes_mid(&flags, tt_mb, seed, weights),
        "nodes-end" => cmd_nodes_end(&flags, tt_mb, seed, weights),
        "sweep" => cmd_sweep(&flags, tt_mb, seed, weights),
        "solveset" => cmd_solveset(&flags, tt_mb, seed, weights),
        "all" => {
            let mut rc = 0;
            rc |= cmd_exact_mid(&flags, tt_mb, seed, weights);
            rc |= cmd_exact_end(&flags, tt_mb, seed, weights);
            rc |= cmd_nodes_mid(&flags, tt_mb, seed, weights);
            rc |= cmd_nodes_end(&flags, tt_mb, seed, weights);
            rc
        }
        other => {
            eprintln!("unknown prune subcommand {other:?}");
            eprintln!("expected one of: exact-mid exact-end nodes-mid nodes-end sweep solveset all");
            2
        }
    }
}

/// Solve a fixed set of positions under exactly one configuration and
/// print the node count. One configuration per process, so the run can
/// be wrapped in `perf stat -e instructions` to obtain the cost of the
/// cutoffs including the work that node counting cannot see.
///
/// Instructions retired is a property of this process alone, so unlike
/// wall-clock time it is unaffected by what else the machine is running.
fn cmd_solveset(
    flags: &HashMap<String, String>,
    tt_mb: usize,
    seed: u64,
    weights: Option<&str>,
) -> i32 {
    let n: usize = flag_num(flags, "positions", 16usize).unwrap_or(16);
    let e: u32 = flag_num(flags, "solve-empties", 18u32).unwrap_or(18);
    let stability: u32 = flag_num(flags, "stability", 0u32).unwrap_or(0);
    let etc: u32 = flag_num(flags, "etc", 0u32).unwrap_or(0);
    let stab_min: u32 = flag_num(flags, "stab-min", 8u32).unwrap_or(8);
    let etc_min: u32 = flag_num(flags, "etc-min", 12u32).unwrap_or(12);
    let positions = positions_at_empties(seed.wrapping_add(u64::from(e)), e, n);
    let mut s = make_solver(tt_mb, weights, stability != 0, etc != 0);
    s.opts.stability_min_empties = stab_min;
    s.opts.etc_empties = etc_min;
    let mut nodes = 0u64;
    let mut stat = versor_search::CutoffStats::default();
    for (pos, _) in &positions {
        s.new_game();
        let r = s.solve_exact(pos, None).expect("no time limit");
        nodes += r.nodes;
        let c = s.stats();
        stat.stability_calls += c.stability_calls;
        stat.stability_cuts += c.stability_cuts;
        stat.etc_calls += c.etc_calls;
        stat.etc_cuts += c.etc_cuts;
    }
    println!(
        "solveset empties={e} positions={} stability={stability} etc={etc} stab_min={stab_min} etc_min={etc_min}",
        positions.len()
    );
    println!(
        "nodes={nodes} stability_calls={} stability_cuts={} etc_calls={} etc_cuts={}",
        stat.stability_calls, stat.stability_cuts, stat.etc_calls, stat.etc_cuts
    );
    0
}

/// Sweep the activation thresholds of both cutoffs, one at a time, and
/// report node counts plus how often each cutoff was computed. Nodes
/// saved per stability call is the figure to judge by: the node count
/// alone always favours computing stability everywhere, but each call
/// costs work that node counting cannot see.
fn cmd_sweep(
    flags: &HashMap<String, String>,
    tt_mb: usize,
    seed: u64,
    weights: Option<&str>,
) -> i32 {
    let n: usize = flag_num(flags, "positions", 60usize).unwrap_or(60);
    let e: u32 = flag_num(flags, "solve-empties", 20u32).unwrap_or(20);
    let positions = positions_at_empties(seed.wrapping_add(u64::from(e)), e, n);
    println!(
        "threshold sweep at {e} empties, {} positions, seed {seed}",
        positions.len()
    );

    let run = |stab: bool, etc: bool, stab_min: u32, etc_min: u32| -> (u64, u64, u64) {
        let mut s = make_solver(tt_mb, weights, stab, etc);
        s.opts.stability_min_empties = stab_min;
        s.opts.etc_empties = etc_min;
        let mut nodes = 0u64;
        let mut calls = 0u64;
        let mut cuts = 0u64;
        for (pos, _) in &positions {
            s.new_game();
            let r = s.solve_exact(pos, None).expect("no time limit");
            nodes += r.nodes;
            let c = s.stats();
            calls += c.stability_calls + c.etc_calls;
            cuts += c.stability_cuts + c.etc_cuts;
        }
        (nodes, calls, cuts)
    };

    let (base_nodes, _, _) = run(false, false, 99, 99);
    println!("  baseline (both off): {base_nodes} nodes");
    println!();
    println!("  stability, varying the minimum empties it is attempted at:");
    println!("    stab_min   nodes         ratio    calls        cuts       nodes_saved_per_call");
    for stab_min in [4u32, 6, 8, 10, 12, 14] {
        let (nodes, calls, cuts) = run(true, false, stab_min, 99);
        let saved = base_nodes.saturating_sub(nodes);
        let per_call = if calls == 0 {
            0.0
        } else {
            saved as f64 / calls as f64
        };
        println!(
            "    {stab_min:>8}   {nodes:>12}  {:>7.4}  {calls:>10}  {cuts:>10}  {per_call:>10.2}",
            nodes as f64 / base_nodes as f64
        );
    }
    println!();
    println!("  etc, varying the minimum empties it is attempted at:");
    println!("    etc_min    nodes         ratio    calls        cuts       nodes_saved_per_call");
    for etc_min in [8u32, 10, 12, 14, 16] {
        let (nodes, calls, cuts) = run(false, true, 99, etc_min);
        let saved = base_nodes.saturating_sub(nodes);
        let per_call = if calls == 0 {
            0.0
        } else {
            saved as f64 / calls as f64
        };
        println!(
            "    {etc_min:>8}   {nodes:>12}  {:>7.4}  {calls:>10}  {cuts:>10}  {per_call:>10.2}",
            nodes as f64 / base_nodes as f64
        );
    }
    0
}

fn cmd_exact_mid(
    flags: &HashMap<String, String>,
    tt_mb: usize,
    seed: u64,
    weights: Option<&str>,
) -> i32 {
    let n: usize = flag_num(flags, "positions", 1000usize).unwrap_or(1000);
    let depth: u32 = flag_num(flags, "depth", 8u32).unwrap_or(8);
    let empties: u32 = flag_num(flags, "empties", 34u32).unwrap_or(34);
    let positions = positions_at_empties(seed, empties, n);
    println!(
        "midgame exactness: depth {depth}, {} random positions at {empties} empties, seed {seed}",
        positions.len()
    );
    if depth < empties {
        println!("  note: depth < empties here, so the stability cutoff is inactive by design");
    }
    // With --plain the ordering heuristics that carry state between
    // iterative-deepening iterations (killers, history, aspiration) are
    // switched off on both sides. Pruning necessarily skips subtrees, so
    // the cutoffs recorded inside them differ, which perturbs move
    // ordering in later iterations. That cannot change any value, but it
    // can change which of several equally valued moves is met first.
    let plain = flags.contains_key("plain");
    if plain {
        println!("  --plain: killers, history and aspiration disabled on both sides");
    }
    let mut all_ok = true;
    for (label, etc, stab) in [
        ("etc only", true, false),
        ("stability only", false, true),
        ("both", true, true),
    ] {
        let m = match weights {
            Some(w) => {
                let mut base = Searcher::new(PatternEval::from_file(w).expect("weights"), tt_mb);
                setup_searcher(&mut base, false, false, plain);
                let mut test = Searcher::new(PatternEval::from_file(w).expect("weights"), tt_mb);
                setup_searcher(&mut test, etc, stab, plain);
                exact_mid(&mut base, &mut test, &positions, depth)
            }
            None => {
                let mut base = Searcher::new(PlaceholderEval, tt_mb);
                setup_searcher(&mut base, false, false, plain);
                let mut test = Searcher::new(PlaceholderEval, tt_mb);
                setup_searcher(&mut test, etc, stab, plain);
                exact_mid(&mut base, &mut test, &positions, depth)
            }
        };
        all_ok &= report_mismatches(label, positions.len(), &m);
    }
    i32::from(!all_ok)
}

fn cmd_exact_end(
    flags: &HashMap<String, String>,
    tt_mb: usize,
    seed: u64,
    weights: Option<&str>,
) -> i32 {
    let n: usize = flag_num(flags, "positions", 300usize).unwrap_or(300);
    let list = flags
        .get("empties")
        .cloned()
        .unwrap_or_else(|| String::from("12,14,16"));
    let mut all_ok = true;
    println!("endgame exactness: exact score and best move vs both cutoffs off, seed {seed}");
    for tok in list.split(',') {
        let Ok(e) = tok.trim().parse::<u32>() else {
            eprintln!("invalid --empties entry {tok:?}");
            return 2;
        };
        let positions = positions_at_empties(seed.wrapping_add(u64::from(e)), e, n);
        let configs: [(&str, bool, bool); 3] = [
            ("stability only", true, false),
            ("etc only", false, true),
            ("both", true, true),
        ];
        let results = exact_end_all(&positions, tt_mb, weights, &configs);
        for ((label, _, _), m) in configs.iter().zip(results.iter()) {
            all_ok &= report_mismatches(
                &format!("empties {e}, {label}"),
                positions.len(),
                m,
            );
        }
    }
    i32::from(!all_ok)
}

fn cmd_nodes_mid(
    flags: &HashMap<String, String>,
    tt_mb: usize,
    seed: u64,
    weights: Option<&str>,
) -> i32 {
    let n: usize = flag_num(flags, "positions", 200usize).unwrap_or(200);
    let empties: u32 = flag_num(flags, "empties", 34u32).unwrap_or(34);
    let depths = flags
        .get("depths")
        .cloned()
        .unwrap_or_else(|| String::from("8,10"));
    let positions = positions_at_empties(seed, empties, n);
    println!(
        "midgame node counts: {} random positions at {empties} empties, seed {seed}",
        positions.len()
    );
    println!("  the stability cutoff does not apply to a depth-limited heuristic search;");
    println!("  only ETC is measured here.");
    println!();
    println!("  depth  etc      nodes         ratio    etc_calls   etc_cuts");
    for tok in depths.split(',') {
        let Ok(d) = tok.trim().parse::<u32>() else {
            eprintln!("invalid --depths entry {tok:?}");
            return 2;
        };
        let mut off_nodes = 0u64;
        for (on, label) in [(false, "off"), (true, "on ")] {
            let (nodes, calls, cuts) = match weights {
                Some(w) => {
                    let mut s = Searcher::new(PatternEval::from_file(w).expect("weights"), tt_mb);
                    s.opts.use_etc = on;
                    nodes_mid(&mut s, &positions, d)
                }
                None => {
                    let mut s = Searcher::new(PlaceholderEval, tt_mb);
                    s.opts.use_etc = on;
                    nodes_mid(&mut s, &positions, d)
                }
            };
            if !on {
                off_nodes = nodes;
            }
            let ratio = if off_nodes == 0 {
                1.0
            } else {
                nodes as f64 / off_nodes as f64
            };
            println!(
                "  {d:>5}  {label}  {nodes:>12}  {ratio:>7.4}  {calls:>10}  {cuts:>9}"
            );
        }
    }
    0
}

fn cmd_nodes_end(
    flags: &HashMap<String, String>,
    tt_mb: usize,
    seed: u64,
    weights: Option<&str>,
) -> i32 {
    let n: usize = flag_num(flags, "positions", 100usize).unwrap_or(100);
    let list = flags
        .get("solve-empties")
        .cloned()
        .unwrap_or_else(|| String::from("17,20"));
    println!("endgame node counts: exact solve, seed {seed}");
    println!();
    println!("  empties  config     nodes           ratio    stab_calls  stab_cuts   etc_calls   etc_cuts");
    for tok in list.split(',') {
        let Ok(e) = tok.trim().parse::<u32>() else {
            eprintln!("invalid --solve-empties entry {tok:?}");
            return 2;
        };
        let positions = positions_at_empties(seed.wrapping_add(u64::from(e)), e, n);
        let mut off_nodes = 0u64;
        for cfg in SOLVER_CFGS.iter() {
            let mut s = make_solver(tt_mb, weights, cfg.stability, cfg.etc);
            let mut nodes = 0u64;
            let mut st = versor_search::CutoffStats::default();
            for (pos, _) in &positions {
                s.new_game();
                let r = s.solve_exact(pos, None).expect("no time limit");
                nodes += r.nodes;
                let c = s.stats();
                st.stability_calls += c.stability_calls;
                st.stability_cuts += c.stability_cuts;
                st.etc_calls += c.etc_calls;
                st.etc_cuts += c.etc_cuts;
            }
            if cfg.label == "off" {
                off_nodes = nodes;
            }
            let ratio = if off_nodes == 0 {
                1.0
            } else {
                nodes as f64 / off_nodes as f64
            };
            println!(
                "  {e:>7}  {:<9}  {nodes:>12}  {ratio:>7.4}  {:>10}  {:>9}  {:>10}  {:>9}",
                cfg.label,
                st.stability_calls,
                st.stability_cuts,
                st.etc_calls,
                st.etc_cuts
            );
        }
    }
    0
}

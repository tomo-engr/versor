// Node counts with each part of the search switched off.
//
// Midgame: 20 positions around move 20, searched to depth 8 with the
// pattern evaluation and the engine's default search options (the
// configuration used in play). One part is switched off at a time:
// the transposition table (which also disables ETC, a table probe),
// PVS, or move ordering (the positional heuristic, killers and
// history; the table move is still tried first). The last row
// switches everything off: plain alpha-beta with iterative deepening.
//
// Endgame: 20 positions at 17 empties, solved exactly. The solver
// orders moves by opponent mobility and corners (no evaluation); the
// transposition table, that ordering and the parity ordering are
// switched off one at a time. A last row gives the solver as used in
// play, with moves ordered by the pattern evaluation.
//
// Node counts only; nothing is timed. The positions come from the
// same random generator and seed as before, so they do not change.
//
// Usage: effects <weights.bin> [seed]

use versor_core::rng::Xorshift64Star;
use versor_core::sample::{random_position, random_position_with_empties};
use versor_core::Position;
use versor_eval::PatternEval;
use versor_search::{SearchOptions, Searcher, Solver, SolverOptions};

const DEPTH: u32 = 8;

fn midgame_nodes(weights: &str, positions: &[Position], opts: SearchOptions) -> u64 {
    let eval = PatternEval::from_file(weights).expect("weights");
    let mut s = Searcher::new(eval, 64);
    s.opts = opts;
    let mut nodes = 0u64;
    for p in positions {
        s.new_game();
        nodes += s.search(p, DEPTH).nodes;
    }
    nodes
}

fn solver_nodes(positions: &[Position], opts: SolverOptions, order_by: Option<&str>) -> u64 {
    let mut s = Solver::new(64);
    s.opts = opts;
    if let Some(weights) = order_by {
        s.set_eval_order(Box::new(PatternEval::from_file(weights).expect("weights")));
    }
    let mut nodes = 0u64;
    for p in positions {
        s.new_game();
        nodes += s.solve_exact(p, None).expect("no limit").nodes;
    }
    nodes
}

fn print_rows(rows: &[(&str, u64)]) {
    let full = rows[0].1;
    println!("| configuration | total nodes | ratio vs full |");
    println!("|:--------------|------------:|--------------:|");
    for (name, nodes) in rows {
        println!(
            "| {} | {} | {:.2} |",
            name,
            nodes,
            *nodes as f64 / full as f64
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let weights = args.first().expect("usage: effects <weights.bin> [seed]");
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut rng = Xorshift64Star::new(seed);
    let midgame: Vec<Position> = (0..20).map(|_| random_position(&mut rng, 20)).collect();
    let endgame: Vec<Position> = (0..20)
        .map(|_| random_position_with_empties(&mut rng, 17))
        .collect();

    println!(
        "midgame: 20 positions (move 20), depth {}, seed {}, weights {}",
        DEPTH, seed, weights
    );
    let full = SearchOptions::default();
    let no_tt = SearchOptions {
        use_tt: false,
        ..full
    };
    let no_pvs = SearchOptions {
        use_pvs: false,
        ..full
    };
    let no_ordering = SearchOptions {
        use_ordering: false,
        use_killers: false,
        use_history: false,
        ..full
    };
    let all_off = SearchOptions {
        use_tt: false,
        use_pvs: false,
        use_ordering: false,
        use_eval_ordering: false,
        use_killers: false,
        use_history: false,
        use_aspiration: false,
        use_etc: false,
        use_stability: false,
        ..full
    };
    print_rows(&[
        (
            "full (default options)",
            midgame_nodes(weights, &midgame, full),
        ),
        ("without TT", midgame_nodes(weights, &midgame, no_tt)),
        ("without PVS", midgame_nodes(weights, &midgame, no_pvs)),
        (
            "without ordering",
            midgame_nodes(weights, &midgame, no_ordering),
        ),
        (
            "plain alpha-beta (all off)",
            midgame_nodes(weights, &midgame, all_off),
        ),
    ]);

    println!();
    println!("solver: 20 positions at 17 empties, seed {}", seed);
    let sfull = SolverOptions::default();
    print_rows(&[
        (
            "full (TT + ordering + parity)",
            solver_nodes(&endgame, sfull, None),
        ),
        (
            "without TT",
            solver_nodes(
                &endgame,
                SolverOptions {
                    use_tt: false,
                    ..sfull
                },
                None,
            ),
        ),
        (
            "without ordering",
            solver_nodes(
                &endgame,
                SolverOptions {
                    use_ordering: false,
                    ..sfull
                },
                None,
            ),
        ),
        (
            "without parity",
            solver_nodes(
                &endgame,
                SolverOptions {
                    use_parity: false,
                    ..sfull
                },
                None,
            ),
        ),
        (
            "full, ordered by the pattern evaluation",
            solver_nodes(&endgame, sfull, Some(weights)),
        ),
    ]);
}

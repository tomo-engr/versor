// The two exact (backward) cutoffs must not change what the search
// computes.
//
// Both the stability cutoff and the enhanced transposition cutoff only
// fire where the node was already going to fail low or fail high, so the
// value a search returns must be identical with them on and off.
//
// The best move is compared under `--plain` conditions, that is with the
// ordering heuristics that carry state between iterative-deepening
// iterations switched off. Pruning necessarily skips subtrees, and the
// beta cutoffs inside those subtrees are what fill the killer and
// history tables; skipping them perturbs move ordering in later
// iterations, which can change which of several equally valued moves the
// root meets first. That is a tie-break difference, not a value error,
// so the value comparison below runs with the engine defaults while the
// move comparison runs with the stateful heuristics disabled.

use versor_core::sample::positions_at_empties;
use versor_core::Position;
use versor_search::{PlaceholderEval, Searcher, Solver};

/// `n` random reachable positions with exactly `empties` empty squares
/// and at least one legal move.
fn sample(seed: u64, empties: u32, n: usize) -> Vec<Position> {
    positions_at_empties(seed, empties, n)
        .into_iter()
        .map(|(p, _)| p)
        .collect()
}

fn make_solver(stability: bool, etc: bool) -> Solver {
    let mut s = Solver::new(8);
    s.opts.use_stability = stability;
    s.opts.use_etc = etc;
    s
}

#[test]
fn solver_cutoffs_preserve_score_and_move() {
    for empties in [10u32, 12, 13] {
        let positions = sample(4242 + u64::from(empties), empties, 40);
        for (stability, etc) in [(true, false), (false, true), (true, true)] {
            let mut base = make_solver(false, false);
            let mut test = make_solver(stability, etc);
            for pos in &positions {
                base.clear_tt();
                test.clear_tt();
                let a = base.solve_exact(pos, None).unwrap();
                let b = test.solve_exact(pos, None).unwrap();
                assert_eq!(
                    a.score, b.score,
                    "score changed at {empties} empties with stability={stability} etc={etc}"
                );
                assert_eq!(
                    a.best_move, b.best_move,
                    "best move changed at {empties} empties with stability={stability} etc={etc}"
                );
            }
        }
    }
}

/// The stability cutoff must actually fire somewhere in this sample,
/// otherwise the test above would pass vacuously.
#[test]
fn solver_stability_cutoff_actually_fires() {
    let positions = sample(99, 14, 12);
    let mut s = make_solver(true, false);
    let mut cuts = 0u64;
    for pos in &positions {
        s.clear_tt();
        s.solve_exact(pos, None).unwrap();
        cuts += s.stats().stability_cuts;
    }
    assert!(cuts > 0, "stability cutoff never fired in the sample");
}

#[test]
fn solver_etc_actually_fires() {
    let positions = sample(101, 16, 8);
    let mut s = make_solver(false, true);
    let mut cuts = 0u64;
    for pos in &positions {
        s.clear_tt();
        s.solve_exact(pos, None).unwrap();
        cuts += s.stats().etc_cuts;
    }
    assert!(cuts > 0, "etc never fired in the sample");
}

fn midgame_pair(etc: bool, stability: bool, plain: bool) -> Searcher<PlaceholderEval> {
    let mut s = Searcher::new(PlaceholderEval, 8);
    s.opts.use_etc = etc;
    s.opts.use_stability = stability;
    if plain {
        s.opts.use_killers = false;
        s.opts.use_history = false;
        s.opts.use_aspiration = false;
    }
    s
}

#[test]
fn midgame_cutoffs_preserve_value() {
    // depth < empties: only ETC is active here by design.
    let positions = sample(7, 30, 60);
    for (etc, stab) in [(true, false), (false, true), (true, true)] {
        let mut base = midgame_pair(false, false, false);
        let mut test = midgame_pair(etc, stab, false);
        for pos in &positions {
            base.clear_tt();
            test.clear_tt();
            let a = base.search(pos, 7);
            let b = test.search(pos, 7);
            assert_eq!(a.value, b.value, "value changed with etc={etc} stability={stab}");
        }
    }
}

#[test]
fn midgame_cutoffs_preserve_value_and_move_where_leaves_are_terminal() {
    // depth >= empties: every leaf is terminal, so the search value is
    // the exact final score and the stability cutoff is active.
    let positions = sample(31, 12, 40);
    for (etc, stab) in [(true, false), (false, true), (true, true)] {
        let mut base = midgame_pair(false, false, true);
        let mut test = midgame_pair(etc, stab, true);
        for pos in &positions {
            base.clear_tt();
            test.clear_tt();
            let a = base.search(pos, 12);
            let b = test.search(pos, 12);
            assert_eq!(a.value, b.value, "value changed with etc={etc} stability={stab}");
            assert_eq!(
                a.best_move, b.best_move,
                "best move changed with etc={etc} stability={stab}"
            );
        }
    }
}

/// Where every leaf is terminal, the depth-limited search must agree
/// with the exact solver. This is the premise the midgame stability
/// cutoff relies on, so it is checked directly.
#[test]
fn deep_midgame_search_matches_the_exact_solver() {
    let positions = sample(555, 11, 30);
    let mut searcher = midgame_pair(true, true, true);
    let mut solver = make_solver(true, true);
    for pos in &positions {
        searcher.clear_tt();
        solver.clear_tt();
        let s = searcher.search(pos, 11);
        let e = solver.solve_exact(pos, None).unwrap();
        assert_eq!(
            s.value, e.score,
            "depth-limited search disagreed with the exact solver"
        );
    }
}

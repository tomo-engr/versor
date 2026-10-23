// Agreement tests between the search and the reference implementations
// in versor_search::reference:
//
//   gate a: plain negamax (no pruning) and alpha-beta agree on the
//           value and on the value of the returned best move.
//   gate b: PVS with transposition table agrees with plain alpha-beta
//           without a table.
//   gate c: the endgame solver agrees with a full-width negamax played
//           out to the end of the game.
//
// The reference side is fixed and must not be modified to force
// agreement. All positions are generated from the initial position by
// playing a seeded-random number of random moves (6..=40 for gates a/b;
// until a target empty count for gate c), so every run is reproducible.
//
// The small variants run in the default test suite; the full-size
// variants are #[ignore]d and run explicitly in release mode.

use versor_core::rng::Xorshift64Star;
use versor_core::sample::{random_position, random_position_with_empties};
use versor_search::reference::{alphabeta_plain, endgame_plain, negamax_plain};
use versor_search::{PlaceholderEval, Searcher, Solver, Wld, SCORE_INF};

fn run_gate_a(n_positions: u32, depth: u32, seed: u64) {
    let mut rng = Xorshift64Star::new(seed);
    for i in 0..n_positions {
        let n_moves = 6 + rng.below(35);
        let p = random_position(&mut rng, n_moves);
        let v_ref = negamax_plain(&PlaceholderEval, &p, depth);
        let mut s = Searcher::new(PlaceholderEval, 1);
        s.opts.use_tt = false;
        s.opts.use_pvs = false;
        let r = s.search(&p, depth);
        assert_eq!(
            r.value,
            v_ref,
            "gate a: alpha-beta value differs from plain negamax at position {} (seed {}):\n{}",
            i,
            seed,
            p.to_text()
        );
        if let Some(mv) = r.best_move {
            let child = p.play(mv).expect("best move must be legal");
            let v_child = -negamax_plain(&PlaceholderEval, &child, depth - 1);
            assert_eq!(
                v_child,
                v_ref,
                "gate a: value of returned best move differs at position {} (seed {}):\n{}",
                i,
                seed,
                p.to_text()
            );
        }
    }
}

fn run_gate_b(n_positions: u32, depth: u32, seed: u64) {
    let mut rng = Xorshift64Star::new(seed);
    let mut s = Searcher::new(PlaceholderEval, 8);
    for i in 0..n_positions {
        let n_moves = 6 + rng.below(35);
        let p = random_position(&mut rng, n_moves);
        let v_ref = alphabeta_plain(&PlaceholderEval, &p, depth, -SCORE_INF, SCORE_INF);
        // Heuristic values are depth-relative, so the table is cleared
        // between unrelated root positions.
        s.clear_tt();
        let r = s.search(&p, depth);
        assert_eq!(
            r.value,
            v_ref,
            "gate b: PVS+TT value differs from plain alpha-beta at position {} (seed {}):\n{}",
            i,
            seed,
            p.to_text()
        );
    }
}

fn run_gate_c(n_positions: u32, empties: u32, seed: u64) {
    let mut rng = Xorshift64Star::new(seed);
    // The solver table is intentionally shared across positions: solver
    // values are absolute game values, so reuse is sound and stresses
    // the table logic harder.
    let mut solver = Solver::new(16);
    for i in 0..n_positions {
        let p = random_position_with_empties(&mut rng, empties);
        let v_ref = endgame_plain(&p);
        let r = solver
            .solve_exact(&p, None)
            .expect("no time limit was set");
        assert_eq!(
            r.score,
            v_ref,
            "gate c: solver score differs from full-width negamax at {} empties, position {} (seed {}):\n{}",
            empties,
            i,
            seed,
            p.to_text()
        );
        let (wld, _) = solver.solve_wld(&p, None).expect("no time limit was set");
        let expected = if v_ref > 0 {
            Wld::Win
        } else if v_ref < 0 {
            Wld::Loss
        } else {
            Wld::Draw
        };
        assert_eq!(wld, expected, "gate c: WLD inconsistent with exact score");
    }
}

#[test]
fn gate_a_small() {
    run_gate_a(30, 4, 0xa0);
}

#[test]
#[ignore = "full gate a (1000 positions x depth 5); run in release mode"]
fn gate_a_full() {
    run_gate_a(1000, 5, 0xa1);
}

#[test]
fn gate_b_small() {
    run_gate_b(30, 5, 0xb0);
}

#[test]
#[ignore = "full gate b (1000 positions x depth 7); run in release mode"]
fn gate_b_full() {
    run_gate_b(1000, 7, 0xb1);
}

#[test]
fn gate_c_small() {
    run_gate_c(20, 8, 0xc0);
}

#[test]
#[ignore = "full gate c (300 positions each at 8/10/12 empties); run in release mode"]
fn gate_c_full() {
    run_gate_c(300, 8, 0xc8);
    run_gate_c(300, 10, 0xca);
    run_gate_c(300, 12, 0xcc);
}

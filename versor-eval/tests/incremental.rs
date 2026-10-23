// Differential update correctness: along random playouts the
// incrementally maintained evaluation state must give exactly the same
// value as a from-scratch evaluation at every position (including
// after passes). The small variant runs by default; the full variant
// covers ten thousand games.

use versor_core::rng::Xorshift64Star;
use versor_core::{bitboard, Position};
use versor_eval::PatternEval;
use versor_search::{Eval, Evaluator};

fn run(n_games: u64, seed: u64) {
    let mut rng = Xorshift64Star::new(seed);
    let n = versor_eval::pattern::N_PHASES * versor_eval::weights::PHASE_STRIDE;
    let weights: Vec<i16> = (0..n).map(|_| (rng.next_u64() & 0x7ff) as i16 - 1024).collect();
    let eval = PatternEval::from_weights(weights).unwrap();
    run_eval(&eval, n_games, seed);
}

// Same random-playout check but against a real weight file (exercises
// the loaded file's full table set and scalars, e.g. v5's 15 tables).
fn run_file(path: &str, n_games: u64, seed: u64) {
    let eval = PatternEval::from_file(path).unwrap();
    run_eval(&eval, n_games, seed);
}

fn run_eval(eval: &PatternEval, n_games: u64, seed: u64) {
    let mut rng = Xorshift64Star::new(seed);
    let mut checked = 0u64;
    for _ in 0..n_games {
        let mut pos = Position::initial();
        let mut st = eval.init_state(&pos);
        loop {
            assert_eq!(
                eval.eval_with(&pos, &st),
                eval.eval(&pos),
                "incremental eval diverged:\n{}",
                pos.to_text()
            );
            checked += 1;
            let moves = pos.legal_moves();
            if moves == 0 {
                if pos.is_game_over() {
                    break;
                }
                pos = pos.pass();
                continue;
            }
            let sq = bitboard::nth_set_bit(moves, rng.below(moves.count_ones()));
            let flips = pos.flips_for(sq);
            st = eval.child_state(&pos, &st, sq, flips);
            pos = pos.play_with_flips(sq, flips);
        }
    }
    println!("incremental ok: {} positions checked", checked);
}

#[test]
fn incremental_matches_full_small() {
    run(300, 90001);
}

#[test]
#[ignore = "full 10000-game verification; run in release mode"]
fn incremental_matches_full_10k() {
    run(10_000, 90002);
}

// File-based full verification: differential update must match
// from-scratch eval over 10000 games with the actual trained weights
// (all pattern tables + scalars of the loaded version). Point
// VERSOR_WEIGHTS at the weight file.
#[test]
#[ignore = "full file-based 10000-game verification; set VERSOR_WEIGHTS, run in release mode"]
fn incremental_matches_full_file_10k() {
    let path = std::env::var("VERSOR_WEIGHTS").expect("set VERSOR_WEIGHTS to a weight file");
    run_file(&path, 10_000, 90003);
}

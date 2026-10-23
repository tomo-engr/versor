// Basic correctness tests for versor-search: zobrist consistency,
// terminal scoring, pass handling at the search root, and a
// hand-verified wipeout / solve case.

use versor_core::bitboard;
use versor_core::notation::{parse_move_seq, replay, square_from_str};
use versor_core::rng::Xorshift64Star;
use versor_core::Position;
use versor_search::{terminal_score, zobrist, PlaceholderEval, Searcher, Solver, Wld};

#[test]
fn zobrist_incremental_matches_full_hash() {
    let mut rng = Xorshift64Star::new(42);
    for _ in 0..300 {
        let mut p = Position::initial();
        let mut h = zobrist::hash(&p);
        loop {
            assert_eq!(h, zobrist::hash(&p), "incremental hash diverged:\n{}", p.to_text());
            let moves = p.legal_moves();
            if moves == 0 {
                if p.is_game_over() {
                    break;
                }
                p = p.pass();
                h = zobrist::pass_hash(h);
                continue;
            }
            let sq = bitboard::nth_set_bit(moves, rng.below(moves.count_ones()));
            let flips = p.flips_for(sq);
            h = zobrist::child_hash(h, p.stm, sq, flips);
            p = p.play_with_flips(sq, flips);
        }
    }
}

#[test]
fn terminal_score_awards_empties_to_winner() {
    // Final position of the shortest game: black 13, white 0, 51 empty.
    // White is the side to move, so the score is -64 (wipeout).
    let moves = parse_move_seq("e6f4e3f6g5d6e7f5c5").unwrap();
    let p = replay(Position::initial(), &moves).unwrap();
    assert!(p.is_game_over());
    assert_eq!(terminal_score(&p), -64);
}

#[test]
fn search_finds_immediate_wipeout() {
    // After e6f4e3f6g5d6e7f5, black to move: c5 wipes out white for a
    // +64 win. A depth 1 search must find it.
    let moves = parse_move_seq("e6f4e3f6g5d6e7f5").unwrap();
    let p = replay(Position::initial(), &moves).unwrap();
    let mut s = Searcher::new(PlaceholderEval, 1);
    let r = s.search(&p, 1);
    assert_eq!(r.value, 64);
    assert_eq!(r.best_move, Some(square_from_str("c5").unwrap()));
}

#[test]
fn search_and_solver_handle_forced_pass_at_root() {
    // White a1, black b1, black to move: black must pass, then white
    // plays c1 and wipes black out. Exact score for black: -64.
    let p = Position::from_text(
        "OX------\n\
         --------\n\
         --------\n\
         --------\n\
         --------\n\
         --------\n\
         --------\n\
         --------\n\
         X\n",
    )
    .unwrap();

    let mut solver = Solver::new(1);
    let r = solver.solve_exact(&p, None).unwrap();
    assert_eq!(r.score, -64);
    assert_eq!(r.best_move, None);
    let (wld, _) = solver.solve_wld(&p, None).unwrap();
    assert_eq!(wld, Wld::Loss);

    let mut s = Searcher::new(PlaceholderEval, 1);
    let sr = s.search(&p, 3);
    assert_eq!(sr.best_move, None);
    assert_eq!(sr.value, -64);
}

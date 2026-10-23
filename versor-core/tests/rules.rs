// Rule correctness tests: initial legal moves, known flip cases, pass
// handling, replay of a short hand-verified game, and text I/O.

use versor_core::notation::{
    move_seq_to_string, parse_move_seq, replay, square_from_str, square_to_string,
};
use versor_core::{Color, GameResult, Position};

fn sq(name: &str) -> u8 {
    square_from_str(name).unwrap()
}

fn bb(names: &[&str]) -> u64 {
    names.iter().fold(0u64, |acc, n| acc | 1u64 << sq(n))
}

#[test]
fn square_mapping() {
    assert_eq!(sq("a1"), 0);
    assert_eq!(sq("h1"), 7);
    assert_eq!(sq("a8"), 56);
    assert_eq!(sq("h8"), 63);
    assert_eq!(sq("d3"), 19);
    for i in 0..64u8 {
        assert_eq!(sq(&square_to_string(i)), i);
    }
    assert_eq!(square_from_str("i1"), None);
    assert_eq!(square_from_str("a9"), None);
    assert_eq!(square_from_str("d33"), None);
}

#[test]
fn initial_position() {
    let p = Position::initial();
    assert_eq!(p.stm, Color::Black);
    assert_eq!(p.black(), bb(&["e4", "d5"]));
    assert_eq!(p.white(), bb(&["d4", "e5"]));
    assert_eq!(p.black().count_ones(), 2);
    assert_eq!(p.white().count_ones(), 2);
}

#[test]
fn initial_legal_moves_are_d3_c4_f5_e6() {
    let p = Position::initial();
    assert_eq!(p.legal_moves(), bb(&["d3", "c4", "f5", "e6"]));
}

#[test]
fn known_flip_case_black_d3() {
    // Black plays d3 in the initial position: exactly d4 is flipped.
    let p = Position::initial();
    assert_eq!(p.flips_for(sq("d3")), bb(&["d4"]));
    let q = p.play(sq("d3")).unwrap();
    assert_eq!(q.stm, Color::White);
    assert_eq!(q.black(), bb(&["d3", "d4", "d5", "e4"]));
    assert_eq!(q.white(), bb(&["e5"]));
}

#[test]
fn known_flip_case_multi_direction() {
    // Hand-built position. Rank 1 is the first line, files a..h left to
    // right, 'X' = black, 'O' = white, '-' = empty, last line = side to move.
    // Black plays c1: flips b1 (west run) and c2 (south run), but not the
    // unbounded north-east or east runs.
    let p = Position::from_text(
        "XO--O---\n\
         --O-----\n\
         --X-----\n\
         --------\n\
         --------\n\
         --------\n\
         --------\n\
         --------\n\
         X\n",
    )
    .unwrap();
    assert_eq!(p.flips_for(sq("c1")), bb(&["b1", "c2"]));
    let q = p.play(sq("c1")).unwrap();
    assert_eq!(q.black(), bb(&["a1", "b1", "c1", "c2", "c3"]));
    assert_eq!(q.white(), bb(&["e1"]));
}

#[test]
fn occupied_or_non_flipping_moves_are_illegal() {
    let p = Position::initial();
    assert!(p.play(sq("d4")).is_err()); // occupied
    assert!(p.play(sq("a1")).is_err()); // empty but flips nothing
    assert_eq!(p.flips_for(sq("d4")), 0);
    assert_eq!(p.flips_for(sq("a1")), 0);
}

#[test]
fn pass_handling() {
    // White on a1, black on b1, black to move. Black has no legal move,
    // white can play c1, so black must pass (and the game is not over).
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
    assert_eq!(p.legal_moves(), 0);
    assert!(p.must_pass());
    assert!(!p.is_game_over());

    let q = p.pass();
    assert_eq!(q.stm, Color::White);
    assert_eq!(q.legal_moves(), bb(&["c1"]));

    // After white plays c1, black is wiped out: game over, white wins 3-0.
    let r = q.play(sq("c1")).unwrap();
    assert_eq!(r.white(), bb(&["a1", "b1", "c1"]));
    assert_eq!(r.black(), 0);
    assert!(r.is_game_over());
    assert!(!r.must_pass());
    assert_eq!(r.result(), Some(GameResult::WhiteWin));
    assert_eq!(r.disc_diff(), -3);
}

#[test]
fn game_over_when_neither_side_can_move() {
    // Two lone discs in opposite corners: no legal move for either side.
    let p = Position::from_text(
        "X-------\n\
         --------\n\
         --------\n\
         --------\n\
         --------\n\
         --------\n\
         --------\n\
         -------O\n\
         X\n",
    )
    .unwrap();
    assert!(p.is_game_over());
    assert_eq!(p.result(), Some(GameResult::Draw));
}

#[test]
fn replay_shortest_game() {
    // The well-known shortest possible game (9 moves): black wipes out
    // all white discs and wins 13-0. Flip-by-flip hand verification:
    //   e6 flips e5 / f4 flips e4 / e3 flips e4 / f6 flips e5
    //   g5 flips f4 / d6 flips d5,e6 / e7 flips e5,e6,f6
    //   f5 flips e5 / c5 flips d4,d5,e5,f5,d6
    let moves = parse_move_seq("e6f4e3f6g5d6e7f5c5").unwrap();
    let p = replay(Position::initial(), &moves).unwrap();
    assert!(p.is_game_over());
    assert_eq!(p.black().count_ones(), 13);
    assert_eq!(p.white(), 0);
    assert_eq!(p.result(), Some(GameResult::BlackWin));
    assert_eq!(
        p.black(),
        bb(&[
            "e3", "d4", "e4", "f4", "c5", "d5", "e5", "f5", "g5", "d6", "e6", "f6", "e7",
        ])
    );
}

#[test]
fn move_seq_parse_and_format() {
    let moves = parse_move_seq("f5d6C3 d3\nc4").unwrap();
    assert_eq!(
        moves,
        vec![sq("f5"), sq("d6"), sq("c3"), sq("d3"), sq("c4")]
    );
    assert_eq!(move_seq_to_string(&moves), "f5d6c3d3c4");
    assert!(parse_move_seq("f5d").is_err());
    assert!(parse_move_seq("f5z9").is_err());
}

#[test]
fn board_text_round_trip() {
    let p = Position::initial();
    let text = p.to_text();
    assert_eq!(
        text,
        "--------\n\
         --------\n\
         --------\n\
         ---OX---\n\
         ---XO---\n\
         --------\n\
         --------\n\
         --------\n\
         X\n"
    );
    assert_eq!(Position::from_text(&text).unwrap(), p);

    // Round trip through a few random-ish moves as well.
    let moves = parse_move_seq("f5d6c3d3c4").unwrap();
    let q = replay(Position::initial(), &moves).unwrap();
    assert_eq!(Position::from_text(&q.to_text()).unwrap(), q);

    assert!(Position::from_text("XX").is_err());
}

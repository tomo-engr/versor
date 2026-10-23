// Cross-check of versor-core move generation and flipping against the
// MIT-licensed magpie crate (dev-dependency only; not part of release
// builds). Random playouts are driven by versor-core; at every position
// the legal move sets of both sides and the full disc sets after each
// move must match magpie exactly.
//
// Bit order differs between the two crates: versor uses LSB = a1 while
// magpie uses MSB = A1 (both row-major), so u64::reverse_bits converts
// exactly between them.

use magpie::othello::{Board, Position as MagpiePosition, Stone};
use versor_core::bitboard;
use versor_core::notation::move_seq_to_string;
use versor_core::rng::Xorshift64Star;
use versor_core::{Color, Position};

fn stone(c: Color) -> Stone {
    match c {
        Color::Black => Stone::Black,
        Color::White => Stone::White,
    }
}

fn crosscheck_games(n_games: u64, seed: u64) {
    let mut rng = Xorshift64Star::new(seed);
    let mut positions_checked = 0u64;
    let mut moves_played = 0u64;
    for game in 0..n_games {
        let mut p = Position::initial();
        let mut b = Board::standard();
        let mut history: Vec<u8> = Vec::with_capacity(64);
        loop {
            let context = |p: &Position, history: &[u8]| {
                format!(
                    "game {} (seed {}), moves {:?}, position:\n{}",
                    game,
                    seed,
                    move_seq_to_string(history),
                    p.to_text()
                )
            };

            // Disc sets must match exactly.
            assert_eq!(
                p.black().reverse_bits(),
                b.bits_for(Stone::Black).raw(),
                "black discs differ: {}",
                context(&p, &history)
            );
            assert_eq!(
                p.white().reverse_bits(),
                b.bits_for(Stone::White).raw(),
                "white discs differ: {}",
                context(&p, &history)
            );

            // Legal move sets of both sides must match exactly.
            let own_moves = p.legal_moves();
            assert_eq!(
                own_moves.reverse_bits(),
                b.moves_for(stone(p.stm)).raw(),
                "legal moves for side to move differ: {}",
                context(&p, &history)
            );
            let opp_moves = bitboard::legal_moves(p.opp, p.own);
            assert_eq!(
                opp_moves.reverse_bits(),
                b.moves_for(stone(p.stm.opponent())).raw(),
                "legal moves for opponent differ: {}",
                context(&p, &history)
            );
            positions_checked += 1;

            if own_moves == 0 {
                if opp_moves == 0 {
                    break; // game over
                }
                p = p.pass();
                continue;
            }
            let sq = bitboard::nth_set_bit(own_moves, rng.below(own_moves.count_ones()));
            let magpie_pos = MagpiePosition::try_from(1u64 << (63 - sq))
                .expect("single bit is a valid magpie position");
            b.play(stone(p.stm), magpie_pos);
            p = p.play_unchecked(sq);
            history.push(sq);
            moves_played += 1;
        }
    }
    println!(
        "crosscheck ok: {} games, {} positions checked, {} moves compared",
        n_games, positions_checked, moves_played
    );
}

#[test]
fn crosscheck_1000_random_games() {
    crosscheck_games(1000, 20260716);
}

#[test]
#[ignore = "long run (100k games); execute with --release -- --ignored"]
fn crosscheck_100k_random_games() {
    crosscheck_games(100_000, 1);
}

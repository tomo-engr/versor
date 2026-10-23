// Stable discs: discs that can never be flipped for the rest of the
// game, no matter how play continues.
//
// The search only needs a lower bound on the stable set (a disc reported
// as stable must really be stable; missing some is merely a weaker
// cutoff), so the test below is deliberately conservative.
//
// A disc can be flipped along one of the four axes (horizontal, vertical
// and the two diagonals) only if, on that axis, one side of it supplies
// an empty square to play on and the other side supplies a bracketing
// disc of the opposite colour. So a disc is safe along an axis when any
// of the following holds:
//
//   1. Every square of its line on that axis is occupied. No move can
//      ever be played on that line, and occupied squares never become
//      empty again, so the condition is permanent.
//   2. It lies on the board border in that axis, i.e. one of the two
//      directions immediately leaves the board. Then a flip would need
//      both the empty square and the bracket on the remaining side,
//      which is impossible.
//   3. The adjacent disc along the axis is the same colour and is itself
//      stable. Any flip reaching this disc from that side would have to
//      flip the stable neighbour first, which cannot happen.
//
// A disc is stable when all four axes are safe. Rule 3 refers back to
// stability, so the whole test is iterated to a fixed point. The
// iteration starts from the empty set and grows monotonically, so it
// terminates and yields the largest set consistent with the rules.
//
// Corners come out stable immediately (rule 2 holds on all four axes),
// edge runs anchored at a corner follow by rule 3 along the edge line,
// and on a full board rule 1 holds everywhere, so every disc is stable.

use crate::bitboard::{shift_e, shift_n, shift_ne, shift_nw, shift_s, shift_se, shift_sw, shift_w};

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = 0x8080_8080_8080_8080;
const RANK_1: u64 = 0x0000_0000_0000_00ff;
const RANK_8: u64 = 0xff00_0000_0000_0000;
/// Squares on the outer ring of the board.
const BORDER: u64 = FILE_A | FILE_H | RANK_1 | RANK_8;

/// Spread every set bit up to seven steps in one direction (the longest
/// run of squares between two ends of a line on an 8x8 board).
#[inline(always)]
fn fill7(x: u64, step: fn(u64) -> u64) -> u64 {
    let mut r = x | step(x);
    r |= step(r);
    r |= step(r);
    r |= step(r);
    r |= step(r);
    r |= step(r);
    r |= step(r);
    r
}

/// Squares whose whole line along one axis is occupied. `pos` and `neg`
/// are the two directions of the axis. A square is excluded exactly when
/// an empty square can be reached from it along the axis.
#[inline(always)]
fn full_lines(empty: u64, pos: fn(u64) -> u64, neg: fn(u64) -> u64) -> u64 {
    !(fill7(empty, pos) | fill7(empty, neg))
}

/// Lower bound on the discs of `player` that can never be flipped.
/// `player` and `other` must be disjoint disc sets.
pub fn stable_discs(player: u64, other: u64) -> u64 {
    debug_assert_eq!(player & other, 0);
    let empty = !(player | other);

    // Rules 1 and 2, which do not depend on the fixed point.
    let base_h = full_lines(empty, shift_e, shift_w) | FILE_A | FILE_H;
    let base_v = full_lines(empty, shift_s, shift_n) | RANK_1 | RANK_8;
    // On either diagonal axis, every border square leaves the board in
    // one of the two directions.
    let base_d1 = full_lines(empty, shift_ne, shift_sw) | BORDER;
    let base_d2 = full_lines(empty, shift_nw, shift_se) | BORDER;

    // Rule 3 by fixed point. `shift_w(stable)` is the set of squares
    // whose eastern neighbour is stable, and so on for the other seven
    // directions.
    let mut stable = 0u64;
    loop {
        let next = player
            & (base_h | shift_w(stable) | shift_e(stable))
            & (base_v | shift_n(stable) | shift_s(stable))
            & (base_d1 | shift_sw(stable) | shift_ne(stable))
            & (base_d2 | shift_se(stable) | shift_nw(stable));
        if next == stable {
            return stable;
        }
        stable = next;
    }
}

/// Number of stable discs of `player`.
#[inline]
pub fn stable_count(player: u64, other: u64) -> u32 {
    stable_discs(player, other).count_ones()
}

/// Upper bound on the final score of the side to move, on the
/// empties-to-the-winner axis used by `Position::final_score`.
///
/// Let `w` be the number of stable discs of the opponent. The opponent
/// ends with at least `w` discs, so the side to move ends with at most
/// `64 - w`. Writing `o` for the opponent's final disc count and `s` for
/// the mover's, the final score is `64 - 2o` when the mover wins,
/// `2s - 64` when the mover loses and `0` on a draw; each of these is at
/// most `64 - 2w`.
#[inline]
pub fn score_upper_bound(own: u64, opp: u64) -> i32 {
    64 - 2 * stable_count(opp, own) as i32
}

/// Lower bound on the final score of the side to move, by the mirror of
/// the argument in `score_upper_bound` applied to the mover's own stable
/// discs.
#[inline]
pub fn score_lower_bound(own: u64, opp: u64) -> i32 {
    2 * stable_count(own, opp) as i32 - 64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xorshift64Star;
    use crate::{bitboard, Position};

    const CORNERS: u64 = 0x8100_0000_0000_0081;

    #[test]
    fn full_board_is_entirely_stable() {
        let mut rng = Xorshift64Star::new(1);
        for _ in 0..200 {
            let player = rng.next_u64();
            let other = !player;
            assert_eq!(stable_discs(player, other), player);
            assert_eq!(stable_discs(other, player), other);
        }
    }

    #[test]
    fn occupied_corners_are_stable() {
        let mut rng = Xorshift64Star::new(2);
        for _ in 0..500 {
            let a = rng.next_u64();
            let b = rng.next_u64() & !a;
            let stable = stable_discs(a, b);
            assert_eq!(stable & CORNERS, a & CORNERS);
        }
    }

    #[test]
    fn initial_position_has_no_stable_disc() {
        let pos = Position::initial();
        assert_eq!(stable_discs(pos.own, pos.opp), 0);
        assert_eq!(stable_discs(pos.opp, pos.own), 0);
    }

    #[test]
    fn stable_set_is_a_subset_of_the_players_discs() {
        let mut rng = Xorshift64Star::new(3);
        for _ in 0..2000 {
            let a = rng.next_u64();
            let b = rng.next_u64() & !a;
            assert_eq!(stable_discs(a, b) & !a, 0);
        }
    }

    /// A corner-anchored run along an edge is stable, and the run stops
    /// where the colour changes.
    #[test]
    fn edge_run_from_a_corner_is_stable() {
        // Rank 1: a1..d1 player, e1 other, rest empty.
        let player = 0b0000_1111u64;
        let other = 0b0001_0000u64;
        let stable = stable_discs(player, other);
        assert_eq!(stable, player, "the whole corner-anchored run is stable");
        // The lone opposing disc at e1 is not anchored to either corner.
        assert_eq!(stable_discs(other, player), 0);
    }

    /// The heart of the contract: a disc reported stable must survive
    /// every legal continuation. Random playouts from random reachable
    /// positions check this by construction.
    #[test]
    fn reported_stable_discs_never_flip_in_random_playouts() {
        let mut rng = Xorshift64Star::new(20260808);
        let mut checked_positions = 0u32;
        let mut total_stable = 0u64;
        for game in 0..1000u64 {
            let mut pos = Position::initial();
            // Walk a random game, checking stability claims at every ply.
            loop {
                if pos.is_game_over() {
                    break;
                }
                if pos.must_pass() {
                    pos = pos.pass();
                    continue;
                }
                // Claims for both sides, recorded as absolute colours.
                let stable_own = stable_discs(pos.own, pos.opp);
                let stable_opp = stable_discs(pos.opp, pos.own);
                let (black_stable, white_stable) = match pos.stm {
                    crate::Color::Black => (stable_own, stable_opp),
                    crate::Color::White => (stable_opp, stable_own),
                };
                total_stable += u64::from(black_stable.count_ones() + white_stable.count_ones());
                checked_positions += 1;

                // Play the rest of the game out randomly and verify that
                // every claimed disc kept both its square and its colour.
                let mut p = pos;
                let mut r = Xorshift64Star::new(game * 7919 + u64::from(checked_positions));
                loop {
                    if p.is_game_over() {
                        break;
                    }
                    if p.must_pass() {
                        p = p.pass();
                        continue;
                    }
                    let moves = p.legal_moves();
                    let k = r.next_u64() % u64::from(moves.count_ones());
                    let sq = bitboard::nth_set_bit(moves, k as u32);
                    p = p.play_unchecked(sq);
                }
                let (black_end, white_end) = (p.black(), p.white());
                assert_eq!(
                    black_stable & !black_end,
                    0,
                    "a disc claimed stable for black was flipped or lost"
                );
                assert_eq!(
                    white_stable & !white_end,
                    0,
                    "a disc claimed stable for white was flipped or lost"
                );

                let moves = pos.legal_moves();
                let k = rng.next_u64() % u64::from(moves.count_ones());
                let sq = bitboard::nth_set_bit(moves, k as u32);
                pos = pos.play_unchecked(sq);
            }
        }
        assert!(checked_positions > 1000);
        // Guard against the test passing vacuously by never claiming
        // anything: real games accumulate plenty of stable discs.
        assert!(
            total_stable > 10_000,
            "expected many stable discs across the sample, got {total_stable}"
        );
    }

    /// The score bounds must bracket the true final score. Checked by
    /// random playout from random positions.
    #[test]
    fn score_bounds_bracket_the_final_score() {
        let mut rng = Xorshift64Star::new(4242);
        for game in 0..500u64 {
            let mut pos = Position::initial();
            loop {
                if pos.is_game_over() {
                    break;
                }
                if pos.must_pass() {
                    pos = pos.pass();
                    continue;
                }
                let lo = score_lower_bound(pos.own, pos.opp);
                let hi = score_upper_bound(pos.own, pos.opp);
                assert!(lo <= hi);
                let mut p = pos;
                let mut r = Xorshift64Star::new(game * 104_729 + 1);
                loop {
                    if p.is_game_over() {
                        break;
                    }
                    if p.must_pass() {
                        p = p.pass();
                        continue;
                    }
                    let moves = p.legal_moves();
                    let k = r.next_u64() % u64::from(moves.count_ones());
                    let sq = bitboard::nth_set_bit(moves, k as u32);
                    p = p.play_unchecked(sq);
                }
                // final_score is from the side to move of `p`; convert to
                // the perspective of the side to move of `pos`.
                let final_from_pos = if p.stm == pos.stm {
                    p.final_score()
                } else {
                    -p.final_score()
                };
                assert!(
                    lo <= final_from_pos && final_from_pos <= hi,
                    "bounds [{lo}, {hi}] do not bracket {final_from_pos}"
                );
                let moves = pos.legal_moves();
                let k = rng.next_u64() % u64::from(moves.count_ones());
                let sq = bitboard::nth_set_bit(moves, k as u32);
                pos = pos.play_unchecked(sq);
            }
        }
    }
}

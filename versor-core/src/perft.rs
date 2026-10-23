// Perft: count leaf nodes of the game tree at a given depth.
//
// Pass convention:
//   - A pass is NOT counted as a move and does NOT consume depth. When
//     the side to move has no legal move but the opponent has, the turn
//     is handed over and counting continues at the same depth.
//   - When neither side has a legal move the game is over; the terminal
//     position counts as exactly one leaf, even if the target depth has
//     not been reached yet.
// With this convention perft(0..=9) from the initial position is:
//   1, 4, 12, 56, 244, 1396, 8200, 55092, 390216, 3005320
// (the widely quoted 3005288 for depth 9 belongs to the convention
// where a forced pass consumes one ply; examples/perft_modes.rs prints
// both)

use crate::position::Position;

pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = pos.legal_moves();
    if moves == 0 {
        let passed = pos.pass();
        if passed.legal_moves() == 0 {
            return 1;
        }
        return perft(&passed, depth);
    }
    let mut nodes = 0u64;
    let mut m = moves;
    while m != 0 {
        let sq = m.trailing_zeros() as u8;
        m &= m - 1;
        nodes += perft(&pos.play_unchecked(sq), depth - 1);
    }
    nodes
}

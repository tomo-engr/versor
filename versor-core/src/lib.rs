// versor-core: board representation, rules and move generation for the
// versor reversi (Othello) engine.
//
// Coordinate convention used across the whole crate:
//   - Files a..h run left to right, ranks 1..8 run top to bottom.
//   - Bit index of a square: sq = row * 8 + col, where col 0..7 = files
//     a..h and row 0..7 = ranks 1..8.
//   - Therefore a1 = bit 0, h1 = bit 7, a8 = bit 56, h8 = bit 63.
//   - The initial position has white discs on d4 and e5, black discs on
//     d5 and e4. Black moves first.

pub mod bitboard;
pub mod notation;
pub mod perft;
pub mod position;
pub mod rng;
pub mod sample;
pub mod stability;

pub use perft::perft;
pub use position::{Color, GameResult, IllegalMove, Position};
pub use stability::{score_lower_bound, score_upper_bound, stable_discs};

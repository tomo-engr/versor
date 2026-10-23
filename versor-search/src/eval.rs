// Evaluation interface and a small hand-written evaluator.
//
// Score scale: disc difference from the side to move's perspective,
// as an integer. Terminal positions are scored exactly with empty
// squares awarded to the winner (standard Othello counting, so a
// wipeout scores +64 or -64). The endgame solver uses the same axis,
// so heuristic and exact values are directly comparable.

use versor_core::{bitboard, Position};

/// Values are always strictly inside (-SCORE_INF, SCORE_INF).
pub const SCORE_INF: i32 = 30000;

/// Exact score of a finished game from the side to move's perspective.
/// Thin wrapper over `Position::final_score`, which is the single
/// implementation of the empties-to-winner convention.
pub fn terminal_score(pos: &Position) -> i32 {
    pos.final_score()
}

/// Object-safe static evaluation. Implementations must be pure
/// functions of the position. May be called on any position, including
/// must-pass positions.
pub trait Eval {
    /// Heuristic value from the side to move's perspective, disc scale.
    fn eval(&self, pos: &Position) -> i32;
}

/// Evaluation with incrementally updatable state, threaded through the
/// search (differential updates). Stateless evaluators use
/// `State = ()`. The state must be independent of the side to move
/// (a pass keeps it unchanged); `eval_with` receives the position for
/// side-to-move handling and must equal `Eval::eval` exactly.
pub trait Evaluator: Eval {
    type State: Copy;
    fn init_state(&self, pos: &Position) -> Self::State;
    /// State after the side to move of `pos` plays `sq` flipping `flips`.
    fn child_state(&self, pos: &Position, st: &Self::State, sq: u8, flips: u64) -> Self::State;
    fn eval_with(&self, pos: &Position, st: &Self::State) -> i32;
}

const CORNERS: u64 = 0x8100_0000_0000_0081;

/// A small linear combination of disc difference, mobility difference
/// and corner ownership, roughly on the disc scale. It needs no weight
/// file: the tests use it, the CLI falls back to it when no weights are
/// given, and `arena` uses it as the default opponent.
pub struct PlaceholderEval;

impl Evaluator for PlaceholderEval {
    type State = ();
    fn init_state(&self, _pos: &Position) -> () {}
    fn child_state(&self, _pos: &Position, _st: &(), _sq: u8, _flips: u64) -> () {}
    fn eval_with(&self, pos: &Position, _st: &()) -> i32 {
        self.eval(pos)
    }
}

impl Eval for PlaceholderEval {
    fn eval(&self, pos: &Position) -> i32 {
        let own_mob = pos.legal_moves().count_ones() as i32;
        let opp_mob = bitboard::legal_moves(pos.opp, pos.own).count_ones() as i32;
        let own_corners = (pos.own & CORNERS).count_ones() as i32;
        let opp_corners = (pos.opp & CORNERS).count_ones() as i32;
        let disc = pos.own.count_ones() as i32 - pos.opp.count_ones() as i32;
        disc + 2 * (own_mob - opp_mob) + 8 * (own_corners - opp_corners)
    }
}

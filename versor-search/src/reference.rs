// Reference implementations for the agreement tests (tests/gates.rs):
// plain negamax without pruning, plain fail-soft alpha-beta without
// transposition table or move ordering, and a full-width negamax that
// searches to the end of the game without any heuristic evaluation.
//
// These functions are the fixed reference side of those tests and are
// kept deliberately simple. They must never be changed to make them
// agree with the main search: a mismatch means the main search is
// wrong.
//
// All functions use the pass convention of perft: a forced pass
// switches the side to move without consuming depth; when neither side
// can move the position is terminal and scored exactly.

use versor_core::{bitboard, Position};

use crate::eval::{terminal_score, Eval, SCORE_INF};

/// Full-width negamax without any pruning.
pub fn negamax_plain<E: Eval>(eval: &E, pos: &Position, depth: u32) -> i32 {
    let moves = pos.legal_moves();
    if moves == 0 {
        if bitboard::legal_moves(pos.opp, pos.own) == 0 {
            return terminal_score(pos);
        }
        return -negamax_plain(eval, &pos.pass(), depth);
    }
    if depth == 0 {
        return eval.eval(pos);
    }
    let mut best = -SCORE_INF;
    let mut m = moves;
    while m != 0 {
        let sq = m.trailing_zeros() as u8;
        m &= m - 1;
        let v = -negamax_plain(eval, &pos.play_unchecked(sq), depth - 1);
        if v > best {
            best = v;
        }
    }
    best
}

/// Fail-soft alpha-beta without transposition table, move ordering or
/// PVS. Moves are visited in ascending square order.
pub fn alphabeta_plain<E: Eval>(
    eval: &E,
    pos: &Position,
    depth: u32,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    let moves = pos.legal_moves();
    if moves == 0 {
        if bitboard::legal_moves(pos.opp, pos.own) == 0 {
            return terminal_score(pos);
        }
        return -alphabeta_plain(eval, &pos.pass(), depth, -beta, -alpha);
    }
    if depth == 0 {
        return eval.eval(pos);
    }
    let mut best = -SCORE_INF;
    let mut m = moves;
    while m != 0 {
        let sq = m.trailing_zeros() as u8;
        m &= m - 1;
        let v = -alphabeta_plain(eval, &pos.play_unchecked(sq), depth - 1, -beta, -alpha);
        if v > best {
            best = v;
            if best > alpha {
                alpha = best;
                if alpha >= beta {
                    break;
                }
            }
        }
    }
    best
}

/// Full-width negamax to the end of the game (no pruning, no heuristic
/// evaluation). Returns the exact score on the same axis as
/// `terminal_score`.
pub fn endgame_plain(pos: &Position) -> i32 {
    let moves = pos.legal_moves();
    if moves == 0 {
        if bitboard::legal_moves(pos.opp, pos.own) == 0 {
            return terminal_score(pos);
        }
        return -endgame_plain(&pos.pass());
    }
    let mut best = -SCORE_INF;
    let mut m = moves;
    while m != 0 {
        let sq = m.trailing_zeros() as u8;
        m &= m - 1;
        let v = -endgame_plain(&pos.play_unchecked(sq));
        if v > best {
            best = v;
        }
    }
    best
}

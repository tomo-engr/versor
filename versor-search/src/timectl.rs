// Clock management for play under a game time control.
//
// The scheme is the one described for BILL: divide the remaining clock
// by the number of moves still to play, then decide iteration by
// iteration whether to keep going (that second half lives in
// `Searcher::search_timed`). No published technical write-up for the
// other engines describes their allocation, so BILL is the only source
// this follows.
//
// Reversi makes the first half easy: the number of moves left is not an
// estimate. Every move fills one square, so the side to move has at most
// ceil(empties / 2) moves left in the game. Chess and shogi programs
// have to guess this number; here it is known.

use std::time::Duration;

use crate::search::TimeBudget;

/// Never plan to spend the last of the clock: keep this much in hand
/// for the moves that follow.
fn reserve(remaining: Duration) -> Duration {
    let two_percent = remaining / 50;
    let floor = Duration::from_secs(5);
    if two_percent > floor {
        two_percent
    } else {
        floor
    }
}

/// Moves the side to move still has in this game, from the empty count.
/// Each move fills one square and the sides alternate, so half the
/// empties (rounded up) is an upper bound; passes only make it smaller.
pub fn moves_left(empties: u32) -> u32 {
    empties.div_ceil(2).max(1)
}

/// Moves the clock actually has to cover.
///
/// This is not the same as `moves_left`, and the difference is worth
/// several seconds a move. Once the empty count reaches the solver's
/// threshold the game is read out exactly, and every move after the
/// first solve is answered from the table at no real cost. Dividing the
/// clock over all remaining moves therefore sets aside time for a dozen
/// moves that will not use any, and the midgame - the part that decides
/// the game - is starved to pay for them.
///
/// So while there is still midgame to play, divide by the number of
/// midgame moves only. Below the threshold, fall back to counting every
/// move: a solve that does not finish leaves real moves to play.
///
/// (This is a departure from the scheme as described for BILL, which
/// divides by the moves remaining in the game. It follows from having a
/// hard endgame threshold, which BILL did not.)
fn budgeted_moves(empties: u32, solve_at: u32) -> u32 {
    if empties > solve_at {
        (empties - solve_at).div_ceil(2).max(1)
    } else {
        moves_left(empties)
    }
}

/// Budget for one midgame move.
///
/// `hard` is 1.5x the allocation so that a single iteration cannot run
/// away, but never more than what is actually left after the reserve.
pub fn budget_for_move(remaining: Duration, empties: u32, solve_at: u32) -> TimeBudget {
    budget_over(remaining, budgeted_moves(empties, solve_at))
}

/// Budget for one move with the clock divided over every move left in
/// the game, as described for BILL. Not used in play; it is kept so
/// that the two schemes can be compared under the same conditions.
pub fn budget_for_move_all_moves(remaining: Duration, empties: u32) -> TimeBudget {
    budget_over(remaining, moves_left(empties))
}

fn budget_over(remaining: Duration, moves: u32) -> TimeBudget {
    let usable = remaining.saturating_sub(reserve(remaining));
    let alloc = usable / moves;
    let hard = (alloc * 3 / 2).min(usable);
    TimeBudget { alloc, hard }
}

/// Budget for one call of the exact endgame solver.
///
/// The endgame is where the clock is worth most: one successful solve
/// decides the rest of the game, and every later move in that game is
/// then almost free. So the solver is allowed several moves' worth of
/// time. If it does not finish, the caller falls back to the
/// depth-limited search, which is why overspending here is not fatal.
pub fn solver_budget(remaining: Duration, empties: u32, factor: u32) -> Duration {
    let usable = remaining.saturating_sub(reserve(remaining));
    let alloc = usable / moves_left(empties);
    (alloc * factor).min(usable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_left_counts_half_the_empties() {
        assert_eq!(moves_left(60), 30);
        assert_eq!(moves_left(20), 10);
        assert_eq!(moves_left(1), 1);
        assert_eq!(moves_left(0), 1); // never divide by zero
    }

    #[test]
    fn only_the_midgame_moves_are_budgeted_for() {
        // 52 empties with the solver taking over at 20: 16 midgame
        // moves to pay for, not 26.
        assert_eq!(budgeted_moves(52, 20), 16);
        assert_eq!(budgeted_moves(21, 20), 1);
        // At and below the threshold every move counts again.
        assert_eq!(budgeted_moves(20, 20), 10);
        assert_eq!(budgeted_moves(8, 20), 4);
    }

    #[test]
    fn allocation_grows_as_the_game_shortens() {
        let r = Duration::from_secs(600);
        let early = budget_for_move(r, 40, 20);
        let late = budget_for_move(r, 24, 20);
        assert!(late.alloc > early.alloc);
        // 600 s less the 12 s reserve, over 10 midgame moves.
        assert_eq!(early.alloc, Duration::from_millis(58_800));
    }

    #[test]
    fn dividing_over_all_moves_gives_each_midgame_move_less() {
        // 40 empties: 20 moves left in all, 10 of them before the
        // solver takes over at 20.
        let r = Duration::from_secs(600);
        let all = budget_for_move_all_moves(r, 40);
        assert_eq!(all.alloc, Duration::from_millis(29_400));
        assert!(all.alloc < budget_for_move(r, 40, 20).alloc);
        // Below the threshold the two schemes agree.
        assert_eq!(
            budget_for_move_all_moves(r, 16).alloc,
            budget_for_move(r, 16, 20).alloc
        );
    }

    #[test]
    fn hard_cap_is_above_the_allocation_but_within_the_clock() {
        let b = budget_for_move(Duration::from_secs(600), 40, 20);
        assert!(b.hard > b.alloc);
        assert!(b.hard <= Duration::from_secs(600));
    }

    #[test]
    fn almost_no_clock_left_yields_a_zero_budget() {
        // Below the reserve there is nothing to allocate; the caller
        // still gets a move because depth 1 ignores the deadline.
        let b = budget_for_move(Duration::from_secs(3), 30, 20);
        assert_eq!(b.alloc, Duration::ZERO);
        assert_eq!(b.hard, Duration::ZERO);
    }

    #[test]
    fn solver_gets_more_than_one_move_worth() {
        let r = Duration::from_secs(600);
        let one_move = budget_for_move(r, 20, 20).alloc;
        let solver = solver_budget(r, 20, 3);
        assert_eq!(solver, one_move * 3);
        assert!(solver < r);
    }
}

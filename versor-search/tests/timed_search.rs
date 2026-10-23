// Verification for the time-limited midgame search
// (`Searcher::search_timed`).
//
// Two properties matter and they pull in opposite directions:
//
//   1. Given plenty of time, the answer must be the one the ordinary
//      depth-limited search gives. The deadline machinery must not
//      change what the search computes.
//   2. Given almost no time, it must still return, and return quickly.
//
// The first is checked over many random positions and is independent of
// how fast the machine is. The second is a wall-clock check, so it is
// written with a generous margin: it asserts that the search does not
// overrun by a wide factor, not that it hits a precise time.

use std::time::{Duration, Instant};

use versor_core::Position;
use versor_search::{PlaceholderEval, Searcher, TimeBudget};

/// Random-ish playout to get a spread of midgame positions.
fn positions(count: usize, plies: u32, seed: u64) -> Vec<Position> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let mut pos = Position::initial();
        for _ in 0..plies {
            if pos.is_game_over() {
                break;
            }
            if pos.must_pass() {
                pos = pos.pass();
                continue;
            }
            let moves = pos.legal_moves();
            let n = moves.count_ones() as u64;
            let mut pick = next() % n;
            let mut m = moves;
            let sq = loop {
                let sq = m.trailing_zeros() as u8;
                if pick == 0 {
                    break sq;
                }
                pick -= 1;
                m &= m - 1;
            };
            pos = pos.play(sq).expect("legal");
        }
        if !pos.is_game_over() && !pos.must_pass() {
            out.push(pos);
        }
    }
    out
}

/// The shape of budget the real time manager produces: a hard cap of
/// 1.5x the allocation. The depth that comes back is then always one
/// that ran to completion, which is what the equivalence check needs.
///
/// An allocation of hours is not a usable stand-in for "no limit" here:
/// the loop would deepen all the way to 60, which is a full-game solve.
fn bounded(alloc_ms: u64) -> TimeBudget {
    let alloc = Duration::from_millis(alloc_ms);
    TimeBudget {
        alloc,
        hard: alloc * 3 / 2,
    }
}

#[test]
fn timed_search_matches_the_depth_limited_search() {
    // With a budget that cannot expire, search_timed keeps deepening
    // until its own stop rule fires. Whatever depth it reaches, the
    // answer at that depth has to equal the plain search at that depth:
    // the deadline plumbing must be invisible to the result.
    let mut checked = 0;
    for pos in positions(500, 20, 0x5eed) {
        // A small allocation stops the deepening early, which keeps the
        // test quick.
        let budget = bounded(1);
        // Fresh searchers per position. Killer and history tables carry
        // over between searches on the same searcher, and the two sides
        // of this comparison do different numbers of searches, so a
        // shared searcher would leave them with different move ordering
        // and let equally-valued moves come out in a different order.
        // That is a property of the test, not of the deadline.
        let mut a = Searcher::new(PlaceholderEval, 4);
        let mut b = Searcher::new(PlaceholderEval, 4);
        let timed = a.search_timed(&pos, budget);
        let plain = b.search(&pos, timed.depth);
        assert_eq!(
            timed.best_move, plain.best_move,
            "move differs at depth {} for {:?}",
            timed.depth, pos
        );
        assert_eq!(
            timed.value, plain.value,
            "value differs at depth {} for {:?}",
            timed.depth, pos
        );
        checked += 1;
    }
    assert_eq!(checked, 500);
}

#[test]
fn deep_timed_search_matches_the_depth_limited_search() {
    // The same property with the deepening allowed to run further, on a
    // handful of positions (this one is slower).
    for pos in positions(8, 16, 0xabcd) {
        let mut a = Searcher::new(PlaceholderEval, 32);
        let mut b = Searcher::new(PlaceholderEval, 32);
        let timed = a.search_timed(&pos, bounded(300));
        let plain = b.search(&pos, timed.depth);
        assert_eq!(timed.best_move, plain.best_move);
        assert_eq!(timed.value, plain.value);
        assert!(timed.depth >= 1);
    }
}

#[test]
fn a_tiny_budget_still_returns_a_legal_move() {
    let mut s = Searcher::new(PlaceholderEval, 16);
    for pos in positions(200, 24, 0x1234) {
        let r = s.search_timed(
            &pos,
            TimeBudget {
                alloc: Duration::ZERO,
                hard: Duration::ZERO,
            },
        );
        let mv = r.best_move.expect("a move is always returned");
        assert!(
            pos.legal_moves() >> mv & 1 == 1,
            "returned an illegal move {} for {:?}",
            mv,
            pos
        );
        assert_eq!(r.depth, 1, "a zero budget should stop after depth 1");
    }
}

#[test]
fn the_hard_deadline_is_respected() {
    // A position deep enough that an unbounded search would run for far
    // longer than the budget, so the cap is what stops it.
    let pos = positions(1, 8, 0x77)[0];
    let mut s = Searcher::new(PlaceholderEval, 64);
    for ms in [50u64, 200, 500] {
        let budget = TimeBudget {
            alloc: Duration::from_millis(ms),
            hard: Duration::from_millis(ms),
        };
        let t0 = Instant::now();
        let r = s.search_timed(&pos, budget);
        let took = t0.elapsed();
        assert!(r.best_move.is_some());
        // The check granularity is 4096 nodes and depth 1 is exempt, so
        // some overshoot is expected; an order of magnitude is not.
        assert!(
            took < Duration::from_millis(ms * 10 + 500),
            "budget {} ms but took {:?}",
            ms,
            took
        );
    }
}

// A searcher or solver that has been used and then reset with
// `new_game` must behave exactly like a newly made one: the same value,
// move, principal variation and node count. This is what makes a match
// give the same games whichever worker plays them and in whatever order.

use versor_core::rng::Xorshift64Star;
use versor_core::sample::random_opening;
use versor_core::Position;
use versor_search::{PlaceholderEval, SearchResult, Searcher, Solver};

fn midgame_positions(n: u64, plies: u32) -> Vec<Position> {
    (0..n)
        .map(|i| random_opening(&mut Xorshift64Star::new(0x5eed + i), plies).0)
        .filter(|p| !p.is_game_over() && !p.must_pass())
        .collect()
}

fn same(a: &SearchResult, b: &SearchResult) -> bool {
    a.value == b.value && a.best_move == b.best_move && a.pv == b.pv && a.nodes == b.nodes
}

/// Search `others` with `s` first, so that its transposition table and
/// history hold what those searches left behind.
fn use_up(s: &mut Searcher<PlaceholderEval>, others: &[Position], depth: u32) {
    for p in others {
        s.search(p, depth);
    }
}

#[test]
fn a_reset_searcher_searches_like_a_new_one() {
    let positions = midgame_positions(12, 20);
    let (targets, others) = positions.split_at(6);
    let mut differs_without_reset = false;
    for p in targets {
        let fresh = Searcher::new(PlaceholderEval, 8).search(p, 6);

        let mut s = Searcher::new(PlaceholderEval, 8);
        use_up(&mut s, others, 6);
        s.new_game();
        let reset = s.search(p, 6);
        assert!(same(&fresh, &reset), "depth 6: reset {:?} vs new {:?}", reset, fresh);

        // Clearing the table alone leaves the history behind; at least
        // one of the positions has to show it, or this test would not be
        // testing anything.
        let mut t = Searcher::new(PlaceholderEval, 8);
        use_up(&mut t, others, 6);
        t.clear_tt();
        differs_without_reset |= !same(&fresh, &t.search(p, 6));
    }
    assert!(differs_without_reset, "the earlier searches left nothing that new_game removes");
}

#[test]
fn a_reset_searcher_spends_a_node_budget_like_a_new_one() {
    let positions = midgame_positions(10, 20);
    let (targets, others) = positions.split_at(5);
    for p in targets {
        let fresh = Searcher::new(PlaceholderEval, 8).search_nodes(p, 30_000);
        let mut s = Searcher::new(PlaceholderEval, 8);
        use_up(&mut s, others, 6);
        s.new_game();
        let reset = s.search_nodes(p, 30_000);
        assert!(same(&fresh, &reset), "node budget: reset {:?} vs new {:?}", reset, fresh);
    }
}

#[test]
fn a_reset_solver_solves_like_a_new_one() {
    // Positions with 14 empty squares: 46 discs on the board.
    let positions = midgame_positions(10, 46);
    let (targets, others) = positions.split_at(5);
    for p in targets {
        let fresh = Solver::new(8).solve_exact(p, None).expect("no limit");
        let mut s = Solver::new(8);
        for q in others {
            s.solve_exact(q, None).expect("no limit");
        }
        s.new_game();
        let reset = s.solve_exact(p, None).expect("no limit");
        assert_eq!(
            (fresh.score, fresh.best_move, fresh.nodes),
            (reset.score, reset.best_move, reset.nodes)
        );
    }
}

// Node counts of the midgame search with the killer, history and
// aspiration heuristics switched off and on, over the same 30 positions
// (20 random moves from the initial position, fixed seed) at depths 8
// and 10. The three only reorder moves, so the values do not change;
// the node ratio is the figure to read. Times are printed for reference
// only.
//
// Usage: cargo run --release -p versor-eval --example ordering_heuristics <weights.bin>

use versor_core::rng::Xorshift64Star;
use versor_core::sample::random_position;
use versor_core::Position;
use versor_eval::PatternEval;
use versor_search::Searcher;

fn run(weights: &str, on: bool, positions: &[Position], depth: u32) -> (u64, f64) {
    let mut s = Searcher::new(PatternEval::from_file(weights).unwrap(), 64);
    s.opts.use_killers = on;
    s.opts.use_history = on;
    s.opts.use_aspiration = on;
    let mut nodes = 0u64;
    let start = std::time::Instant::now();
    for p in positions {
        s.new_game();
        nodes += s.search(p, depth).nodes;
    }
    (nodes, start.elapsed().as_secs_f64())
}

fn main() {
    let weights = std::env::args()
        .nth(1)
        .expect("usage: ordering_heuristics <weights.bin>");
    let mut rng = Xorshift64Star::new(1);
    let positions: Vec<Position> = (0..30).map(|_| random_position(&mut rng, 20)).collect();
    for depth in [8u32, 10] {
        let (n_off, t_off) = run(&weights, false, &positions, depth);
        let (n_on, t_on) = run(&weights, true, &positions, depth);
        println!(
            "depth {}: off {} nodes ({:.2}s), on {} nodes ({:.2}s), node ratio {:.3}, time ratio {:.3}",
            depth,
            n_off,
            t_off,
            n_on,
            t_on,
            n_on as f64 / n_off as f64,
            t_on / t_off
        );
    }
}

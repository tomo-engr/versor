// Search benchmark with the trained pattern evaluation: same midgame
// positions and depths as versor-search's search_bench example, each
// depth with and without evaluation-based move ordering. Node counts are
// reproducible anywhere; times and nodes/s are only comparable within
// one run on one machine.
//
// Usage: cargo run --release -p versor-eval --example search_bench_pattern <weights.bin> [seed]

use std::time::Instant;

use versor_core::rng::Xorshift64Star;
use versor_core::sample::random_position;
use versor_core::Position;
use versor_eval::PatternEval;
use versor_search::Searcher;

fn main() {
    let mut args = std::env::args().skip(1);
    let weights = args.next().expect("usage: search_bench_pattern <weights.bin> [seed]");
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    const N_POSITIONS: usize = 20;
    const MOVES_PLAYED: u32 = 20;

    let mut rng = Xorshift64Star::new(seed);
    let positions: Vec<Position> = (0..N_POSITIONS)
        .map(|_| random_position(&mut rng, MOVES_PLAYED))
        .collect();

    for eval_ordering in [false, true] {
        println!(
            "pattern eval, eval_ordering {}: {} midgame positions (move {}), seed {}",
            eval_ordering, N_POSITIONS, MOVES_PLAYED, seed
        );
        println!("| depth | total nodes | time_s | Mnodes/s | avg nodes/pos |");
        println!("|------:|------------:|-------:|---------:|--------------:|");
        for depth in 8..=10u32 {
            let mut searcher =
                Searcher::new(PatternEval::from_file(&weights).expect("weights"), 64);
            searcher.opts.use_eval_ordering = eval_ordering;
            let mut nodes = 0u64;
            let start = Instant::now();
            for p in &positions {
                searcher.new_game();
                let r = searcher.search(p, depth);
                nodes += r.nodes;
            }
            let dt = start.elapsed().as_secs_f64();
            println!(
                "| {} | {} | {:.2} | {:.2} | {} |",
                depth,
                nodes,
                dt,
                nodes as f64 / dt / 1e6,
                nodes / N_POSITIONS as u64
            );
        }
        println!();
    }
}

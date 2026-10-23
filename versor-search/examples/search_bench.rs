// Search benchmark: midgame positions (around move 20) searched at
// depths 8-10 with the default configuration. Prints total nodes,
// time and nodes per second per depth. Single thread, fixed seed.
//
// Usage: cargo run --release -p versor-search --example search_bench [seed]
// (the pattern-eval variant lives in versor-eval/examples/search_bench_pattern.rs)

use std::time::Instant;

use versor_core::rng::Xorshift64Star;
use versor_core::sample::random_position;
use versor_core::Position;
use versor_search::{PlaceholderEval, Searcher};

fn main() {
    let seed: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    const N_POSITIONS: usize = 20;
    const MOVES_PLAYED: u32 = 20;

    let mut rng = Xorshift64Star::new(seed);
    let positions: Vec<Position> = (0..N_POSITIONS)
        .map(|_| random_position(&mut rng, MOVES_PLAYED))
        .collect();

    println!(
        "search benchmark: {} midgame positions (move {}), seed {}, single thread",
        N_POSITIONS, MOVES_PLAYED, seed
    );
    println!("| depth | total nodes | time_s | Mnodes/s | avg nodes/pos |");
    println!("|------:|------------:|-------:|---------:|--------------:|");
    for depth in 8..=10u32 {
        let mut searcher = Searcher::new(PlaceholderEval, 64);
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
}

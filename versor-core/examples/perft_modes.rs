// The two perft pass conventions side by side. Enumerates all positions
// after exactly 8 moves and reports the pass and game-over situations at
// the ply 8/9 boundary, which account for the whole difference at depth
// 9 (3005320 when a forced pass does not consume depth, 3005288 when it
// consumes one ply), then prints perft(8..=10) under both conventions.
//
// Usage: cargo run --release -p versor-core --example perft_modes

use versor_core::notation::move_seq_to_string;
use versor_core::Position;

struct Stats {
    after8: u64,
    must_pass_at_8: u64,
    game_over_at_8: u64,
    extra_leaves: u64, // sum over must-pass positions of (replies - 1)
    examples: Vec<(String, u32)>,
}

fn dfs(p: &Position, depth: u32, path: &mut Vec<u8>, st: &mut Stats) {
    if depth == 0 {
        st.after8 += 1;
        if p.legal_moves() == 0 {
            let passed = p.pass();
            let replies = passed.legal_moves().count_ones();
            if replies == 0 {
                st.game_over_at_8 += 1;
                st.examples.push((move_seq_to_string(path), 0));
            } else {
                st.must_pass_at_8 += 1;
                st.extra_leaves += (replies - 1) as u64;
                if st.examples.len() < 5 {
                    st.examples.push((move_seq_to_string(path), replies));
                }
            }
        }
        return;
    }
    let moves = p.legal_moves();
    if moves == 0 {
        let passed = p.pass();
        if passed.legal_moves() == 0 {
            println!("game over before ply 8: {}", move_seq_to_string(path));
            return;
        }
        println!("early pass before ply 8: {}", move_seq_to_string(path));
        dfs(&passed, depth, path, st);
        return;
    }
    let mut m = moves;
    while m != 0 {
        let sq = m.trailing_zeros() as u8;
        m &= m - 1;
        path.push(sq);
        dfs(&p.play_unchecked(sq), depth - 1, path, st);
        path.pop();
    }
}

// perft where a forced pass consumes one ply (counted as a move).
fn perft_pass_as_ply(p: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = p.legal_moves();
    if moves == 0 {
        let passed = p.pass();
        if passed.legal_moves() == 0 {
            return 1;
        }
        return perft_pass_as_ply(&passed, depth - 1);
    }
    let mut nodes = 0u64;
    let mut m = moves;
    while m != 0 {
        let sq = m.trailing_zeros() as u8;
        m &= m - 1;
        nodes += perft_pass_as_ply(&p.play_unchecked(sq), depth - 1);
    }
    nodes
}

fn main() {
    let mut st = Stats {
        after8: 0,
        must_pass_at_8: 0,
        game_over_at_8: 0,
        extra_leaves: 0,
        examples: Vec::new(),
    };
    let mut path = Vec::new();
    dfs(&Position::initial(), 8, &mut path, &mut st);
    println!("positions after exactly 8 moves: {}", st.after8);
    println!("  side to move must pass:        {}", st.must_pass_at_8);
    println!("  game already over:             {}", st.game_over_at_8);
    println!("  sum of (replies - 1):          {}", st.extra_leaves);
    for (seq, replies) in &st.examples {
        println!("  example: {} (replies after pass: {})", seq, replies);
    }
    for d in 8..=10u32 {
        println!(
            "perft({}) pass-transparent = {}, pass-consumes-ply = {}",
            d,
            versor_core::perft(&Position::initial(), d),
            perft_pass_as_ply(&Position::initial(), d)
        );
    }
}

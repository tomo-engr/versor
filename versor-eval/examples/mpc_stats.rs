// MPC statistics collection. For random midgame
// positions of phases 2..13, runs fixed-depth searches at depths
// 2/4/6/8 with the given evaluation and writes one CSV row per
// position: phase,v2,v4,v6,v8. The Python side (python/mpc_fit.py)
// fits the depth-pair regressions. Threaded (max 8), fixed seed.
//
// The positions depend only on the seed and the count: each phase is
// drawn in SLICES slices with their own random streams, whatever the
// number of threads, and the rows are written in slice order. Every
// search starts from a fresh searcher state (`new_game`). The output is
// therefore the same for any number of threads.
//
// Usage: mpc_stats <weights.bin> <out.csv> [positions_per_phase] [threads] [seed]

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use versor_core::bitboard;
use versor_core::rng::Xorshift64Star;
use versor_core::Position;
use versor_eval::PatternEval;
use versor_search::Searcher;

const DEPTHS: [u32; 4] = [2, 4, 6, 8];
const PHASES: std::ops::Range<usize> = 2..14;
/// Random streams per phase. Two, the number of threads the published
/// coefficients were collected with when the streams followed the
/// threads, so the same seed gives the same positions as then.
const SLICES: usize = 2;

fn random_position_in_phase(rng: &mut Xorshift64Star, phase: usize) -> Position {
    // discs in [4*phase+4, 4*phase+7]; pick a target disc count and
    // play random moves until reached (positions with a legal move).
    let target = 4 * phase as u32 + 4 + rng.below(4);
    loop {
        let mut p = Position::initial();
        loop {
            if p.is_game_over() {
                break;
            }
            if (p.own | p.opp).count_ones() == target && p.legal_moves() != 0 {
                return p;
            }
            if p.must_pass() {
                p = p.pass();
                continue;
            }
            let moves = p.legal_moves();
            let sq = bitboard::nth_set_bit(moves, rng.below(moves.count_ones()));
            p = p.play_unchecked(sq);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let weights = args.first().expect("usage: mpc_stats <weights> <out.csv>");
    let out_path = args.get(1).expect("out.csv");
    let per_phase: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20000);
    let threads: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
        .clamp(1, 8);
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(77);
    PatternEval::from_file(weights).expect("weights");

    let jobs: Vec<(usize, usize)> = PHASES
        .flat_map(|ph| (0..SLICES).map(move |t| (ph, t)))
        .collect();
    let next = AtomicUsize::new(0);
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path).expect("create out"));
    let rows: Mutex<Vec<String>> = Mutex::new(vec![String::new(); jobs.len()]);

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let eval = PatternEval::from_file(weights).expect("weights");
                let mut searcher = Searcher::new(eval, 16);
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= jobs.len() {
                        break;
                    }
                    let (phase, slice) = jobs[j];
                    let mut rng = Xorshift64Star::new(
                        seed ^ ((phase * 16 + slice) as u64 + 1)
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    );
                    let n = per_phase / SLICES;
                    let mut buf = String::with_capacity(n * 24);
                    for _ in 0..n {
                        let p = random_position_in_phase(&mut rng, phase);
                        let mut vals = [0i32; 4];
                        for (i, &d) in DEPTHS.iter().enumerate() {
                            searcher.new_game();
                            vals[i] = searcher.search(&p, d).value;
                        }
                        buf.push_str(&format!(
                            "{},{},{},{},{}\n",
                            phase, vals[0], vals[1], vals[2], vals[3]
                        ));
                    }
                    rows.lock().unwrap()[j] = buf;
                }
            });
        }
    });
    for buf in rows.into_inner().unwrap() {
        out.write_all(buf.as_bytes()).unwrap();
    }
    out.flush().unwrap();
    eprintln!("done");
}

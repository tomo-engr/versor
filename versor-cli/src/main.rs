// versor-cli: command line interface for the versor engine: rules and
// perft, search and the endgame solver, benchmarks, self-play (arena),
// matches against external engines, the pruning checks and the resident
// JSON server behind the WebUI. Run with "help" for usage.

use std::hint::black_box;
use std::io::Read;
use std::time::Instant;

use std::collections::HashMap;
use std::time::Duration;

use versor_core::bitboard;
use versor_core::notation::{move_seq_to_string, parse_move_seq, replay, square_to_string};
use versor_core::rng::Xorshift64Star;
use versor_core::{perft, Color, GameResult, Position};
use versor_eval::PatternEval;
use versor_search::{Evaluator, PlaceholderEval, Searcher, Solver, Wld};

mod extengine;
mod prune;
mod serve;

// Depth 9 is 3005320 under the implemented convention (a forced pass
// does not consume depth; see versor-core/src/perft.rs).
const PERFT_REFERENCE: [u64; 10] = [
    1, 4, 12, 56, 244, 1396, 8200, 55092, 390216, 3005320,
];

fn usage() {
    print!(
        "versor-cli: simple CLI for the versor reversi engine\n\
         \n\
         usage: versor-cli <command> [args]\n\
         \n\
         commands:\n\
         \x20 show               print the initial position\n\
         \x20 show -             read a position in text format from stdin and print it\n\
         \x20 play <moves>       replay a move string like f5d6c3 from the initial position\n\
         \x20 random [n] [seed]  random self-play; n games (default 1), seed (default 1)\n\
         \x20 perft [depth]      perft from the initial position up to depth (default 9)\n\
         \x20 bench [seed]       movegen / flip throughput benchmark (single thread)\n\
         \x20 search <depth> [pos] [--weights FILE] [--tt MB] [--solve-at N]\n\
         \x20                    search the position: best move, value, nodes, pv.\n\
         \x20                    switches to the exact solver at N empties (default 20).\n\
         \x20                    --weights selects the pattern evaluation (default: a small\n\
         \x20                    hand-written one); the solver then orders moves by it, as\n\
         \x20                    solve does\n\
         \x20 solve [pos] [--tt MB] [--wld] [--limit SECS] [--weights FILE]\n\
         \x20                    solve the position exactly (or win/loss/draw with --wld)\n\
         \x20 ffo <file> [--limit SECS] [--tt MB] [--from N] [--to N] [--weights FILE]\n\
         \x20                    run FFO problems from a file (id;board;stm;score lines)\n\
         \x20 arena --weights FILE [--games N] [--depth D] [--threads T<=8] [--seed S] [--log FILE]\n\
         \x20                    pattern eval vs placeholder from shared random openings\n\
         \x20 enginematch --weights FILE [--engine edax|egaroucid] [--level L] [--games N]\n\
         \x20             [--depth D | --game-time SECS] [--threads T] [--seed S] [--log FILE]\n\
         \x20             [--mpc PARAMS [--mpc-t T]] [--alloc-all-moves] [--no-clock-margin]\n\
         \x20                    match against an external engine (config/engines.json).\n\
         \x20                    --mpc turns on Multi-ProbCut for versor; --alloc-all-moves\n\
         \x20                    divides the clock over all moves left (default: midgame moves);\n\
         \x20                    --no-clock-margin tells the engine its full remaining time\n\
         \x20 prune <sub> [--weights FILE] [--positions N] [--depth D] [--seed S] [--tt MB]\n\
         \x20                    checks of the exact cutoffs. sub is one of exact-mid, exact-end,\n\
         \x20                    nodes-mid, nodes-end, sweep, solveset, all. Node counts only,\n\
         \x20                    never timings\n\
         \x20 serve --weights FILE [--mpc PARAMS] [--mpc-t T] [--tt MB] [--hint-depth D]\n\
         \x20                    resident JSON engine over stdin/stdout (WebUI backend)\n\
         \x20 help               show this help\n\
         \n\
         [pos] is a move string like f5d6c3 (from the initial position),\n\
         or '-' to read a position in text format from stdin; it defaults\n\
         to the initial position.\n"
    );
}

fn print_board(p: &Position) {
    let black = p.black();
    let white = p.white();
    println!("  a b c d e f g h");
    for row in 0..8u8 {
        let mut line = format!("{}", row + 1);
        for col in 0..8u8 {
            let sq = row * 8 + col;
            let c = if black >> sq & 1 == 1 {
                'X'
            } else if white >> sq & 1 == 1 {
                'O'
            } else {
                '.'
            };
            line.push(' ');
            line.push(c);
        }
        println!("{}", line);
    }
    let b = black.count_ones();
    let w = white.count_ones();
    match p.result() {
        Some(r) => {
            let rs = match r {
                GameResult::BlackWin => "black wins",
                GameResult::WhiteWin => "white wins",
                GameResult::Draw => "draw",
            };
            println!("Game over. X {} - O {} ({}).", b, w, rs);
        }
        None => {
            let stm = match p.stm {
                Color::Black => "X (black)",
                Color::White => "O (white)",
            };
            println!("{} to move. X {} - O {}.", stm, b, w);
            let mut moves = p.legal_moves();
            if moves == 0 {
                println!("No legal moves: side to move must pass.");
            } else {
                let mut names = Vec::new();
                while moves != 0 {
                    let sq = moves.trailing_zeros() as u8;
                    moves &= moves - 1;
                    names.push(square_to_string(sq));
                }
                println!("Legal moves: {}", names.join(" "));
            }
        }
    }
}

fn cmd_show(rest: &[String]) -> i32 {
    if rest.first().map(|s| s.as_str()) == Some("-") {
        let mut text = String::new();
        if let Err(e) = std::io::stdin().read_to_string(&mut text) {
            eprintln!("error reading stdin: {}", e);
            return 1;
        }
        match Position::from_text(&text) {
            Ok(p) => {
                print_board(&p);
                0
            }
            Err(e) => {
                eprintln!("parse error: {}", e);
                1
            }
        }
    } else {
        print_board(&Position::initial());
        0
    }
}

fn cmd_play(rest: &[String]) -> i32 {
    let Some(seq) = rest.first() else {
        eprintln!("usage: versor-cli play <moves>  (e.g. play f5d6c3)");
        return 2;
    };
    let moves = match parse_move_seq(seq) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse error: {}", e);
            return 1;
        }
    };
    match replay(Position::initial(), &moves) {
        Ok(p) => {
            println!("Moves: {}", move_seq_to_string(&moves));
            print_board(&p);
            0
        }
        Err(e) => {
            eprintln!("replay error: {}", e);
            1
        }
    }
}

// Play one random game; returns the final position and the move list.
fn random_game(rng: &mut Xorshift64Star, moves_out: &mut Vec<u8>) -> Position {
    moves_out.clear();
    let mut p = Position::initial();
    loop {
        let moves = p.legal_moves();
        if moves == 0 {
            if p.is_game_over() {
                return p;
            }
            p = p.pass();
            continue;
        }
        let n = moves.count_ones();
        let sq = bitboard::nth_set_bit(moves, rng.below(n));
        moves_out.push(sq);
        p = p.play_unchecked(sq);
    }
}

fn cmd_random(rest: &[String]) -> i32 {
    let n: u64 = match rest.first().map(|s| s.parse()) {
        None => 1,
        Some(Ok(v)) if v >= 1 => v,
        _ => {
            eprintln!("usage: versor-cli random [n_games >= 1] [seed]");
            return 2;
        }
    };
    let seed: u64 = match rest.get(1).map(|s| s.parse()) {
        None => 1,
        Some(Ok(v)) => v,
        Some(Err(_)) => {
            eprintln!("usage: versor-cli random [n_games] [seed]");
            return 2;
        }
    };
    let mut rng = Xorshift64Star::new(seed);
    let mut moves = Vec::with_capacity(64);
    if n == 1 {
        let p = random_game(&mut rng, &mut moves);
        println!("Moves: {}", move_seq_to_string(&moves));
        print_board(&p);
        return 0;
    }
    let (mut bw, mut ww, mut dr) = (0u64, 0u64, 0u64);
    let start = Instant::now();
    for _ in 0..n {
        let p = random_game(&mut rng, &mut moves);
        match p.result().expect("game must be over") {
            GameResult::BlackWin => bw += 1,
            GameResult::WhiteWin => ww += 1,
            GameResult::Draw => dr += 1,
        }
    }
    let dt = start.elapsed().as_secs_f64();
    println!(
        "{} random games (seed {}): black wins {}, white wins {}, draws {}",
        n, seed, bw, ww, dr
    );
    println!("elapsed {:.3} s ({:.0} games/s)", dt, n as f64 / dt);
    0
}

fn cmd_perft(rest: &[String]) -> i32 {
    let max_depth: u32 = match rest.first().map(|s| s.parse()) {
        None => 9,
        Some(Ok(v)) if v <= 12 => v,
        _ => {
            eprintln!("usage: versor-cli perft [max_depth <= 12]");
            return 2;
        }
    };
    let p = Position::initial();
    println!("depth {:>12} {:>10} {:>10}  reference", "nodes", "time_s", "Mnodes/s");
    for depth in 0..=max_depth {
        let start = Instant::now();
        let nodes = perft(&p, depth);
        let dt = start.elapsed().as_secs_f64();
        let rate = if dt > 0.0 {
            nodes as f64 / dt / 1e6
        } else {
            0.0
        };
        let status = match PERFT_REFERENCE.get(depth as usize) {
            Some(&r) if r == nodes => "ok",
            Some(_) => "MISMATCH",
            None => "-",
        };
        println!(
            "{:>5} {:>12} {:>10.3} {:>10.2}  {}",
            depth, nodes, dt, rate, status
        );
    }
    0
}

fn cmd_bench(rest: &[String]) -> i32 {
    let seed: u64 = match rest.first().map(|s| s.parse()) {
        None => 1,
        Some(Ok(v)) => v,
        Some(Err(_)) => {
            eprintln!("usage: versor-cli bench [seed]");
            return 2;
        }
    };
    const N_POSITIONS: usize = 100_000;
    const TARGET_SECS: f64 = 2.0;

    // Sample positions (side-to-move relative disc sets) from random
    // self-play so that the benchmark runs on a realistic mix of
    // opening, midgame and endgame positions.
    let mut rng = Xorshift64Star::new(seed);
    let mut positions: Vec<(u64, u64)> = Vec::with_capacity(N_POSITIONS);
    'outer: loop {
        let mut p = Position::initial();
        loop {
            let moves = p.legal_moves();
            if moves == 0 {
                if p.is_game_over() {
                    break;
                }
                p = p.pass();
                continue;
            }
            positions.push((p.own, p.opp));
            if positions.len() == N_POSITIONS {
                break 'outer;
            }
            let sq = bitboard::nth_set_bit(moves, rng.below(moves.count_ones()));
            p = p.play_unchecked(sq);
        }
    }
    println!(
        "benchmark on {} sampled positions (seed {}), single thread",
        positions.len(),
        seed
    );

    // 1. Legal move generation alone.
    {
        let mut acc = 0u64;
        let mut count = 0u64;
        let start = Instant::now();
        loop {
            for &(own, opp) in &positions {
                acc ^= bitboard::legal_moves(black_box(own), black_box(opp));
            }
            count += positions.len() as u64;
            if start.elapsed().as_secs_f64() >= TARGET_SECS {
                break;
            }
        }
        let dt = start.elapsed().as_secs_f64();
        black_box(acc);
        println!(
            "legal_moves only:          {:>7.2} Mpos/s ({} calls in {:.3} s)",
            count as f64 / dt / 1e6,
            count,
            dt
        );
    }

    // 2. Legal move generation plus one flip computation per position
    //    (a pseudo-randomly chosen legal move).
    {
        let mut acc = 0u64;
        let mut count = 0u64;
        let mut pick = 0u32;
        let start = Instant::now();
        loop {
            for &(own, opp) in &positions {
                let moves = bitboard::legal_moves(black_box(own), black_box(opp));
                let n = moves.count_ones();
                if n != 0 {
                    pick = pick.wrapping_add(1);
                    let sq = bitboard::nth_set_bit(moves, pick % n);
                    acc ^= bitboard::flips(own, opp, sq);
                }
            }
            count += positions.len() as u64;
            if start.elapsed().as_secs_f64() >= TARGET_SECS {
                break;
            }
        }
        let dt = start.elapsed().as_secs_f64();
        black_box(acc);
        println!(
            "legal_moves + one flips:   {:>7.2} Mpos/s ({} positions in {:.3} s)",
            count as f64 / dt / 1e6,
            count,
            dt
        );
    }

    // 3. Full random playouts: legal move generation, random choice,
    //    flip computation and position update for every move played.
    {
        let mut rng = Xorshift64Star::new(seed);
        let mut games = 0u64;
        let mut plies = 0u64;
        let start = Instant::now();
        loop {
            let mut p = Position::initial();
            loop {
                let moves = p.legal_moves();
                if moves == 0 {
                    if p.is_game_over() {
                        break;
                    }
                    p = p.pass();
                    continue;
                }
                let sq = bitboard::nth_set_bit(moves, rng.below(moves.count_ones()));
                p = p.play_unchecked(sq);
                plies += 1;
            }
            black_box(p.own);
            games += 1;
            if games.is_multiple_of(1024) && start.elapsed().as_secs_f64() >= TARGET_SECS {
                break;
            }
        }
        let dt = start.elapsed().as_secs_f64();
        println!(
            "random playout:            {:>7.2} Mmoves/s, {:.0} games/s ({} games in {:.3} s)",
            plies as f64 / dt / 1e6,
            games as f64 / dt,
            games,
            dt
        );
    }
    0
}

// Split trailing arguments into positional arguments and --name value
// flags (--wld is a boolean flag without a value).
pub(crate) fn split_flags(rest: &[String]) -> Result<(Vec<String>, HashMap<String, String>), String> {
    let mut positional = Vec::new();
    let mut flags = HashMap::new();
    let mut i = 0;
    while i < rest.len() {
        if let Some(name) = rest[i].strip_prefix("--") {
            if name == "wld" || name == "plain" || name == "b-plain-search"
                || name == "b-no-prune" || name == "no-prune" || name == "alloc-all-moves"
                || name == "no-clock-margin"
            {
                flags.insert(name.to_string(), String::from("1"));
            } else {
                i += 1;
                let v = rest
                    .get(i)
                    .ok_or_else(|| format!("flag --{} requires a value", name))?;
                flags.insert(name.to_string(), v.clone());
            }
        } else {
            positional.push(rest[i].clone());
        }
        i += 1;
    }
    Ok((positional, flags))
}

pub(crate) fn flag_num<T: std::str::FromStr>(
    flags: &HashMap<String, String>,
    name: &str,
    default: T,
) -> Result<T, String> {
    match flags.get(name) {
        None => Ok(default),
        Some(v) => v
            .parse()
            .map_err(|_| format!("invalid value {:?} for --{}", v, name)),
    }
}

// A position argument is a move string (replayed from the initial
// position), "-" for the text format on stdin, or absent (initial).
fn position_from_arg(arg: Option<&str>) -> Result<Position, String> {
    match arg {
        None => Ok(Position::initial()),
        Some("-") => {
            let mut text = String::new();
            std::io::stdin()
                .read_to_string(&mut text)
                .map_err(|e| format!("error reading stdin: {}", e))?;
            Position::from_text(&text)
        }
        Some(moves) => {
            let moves = parse_move_seq(moves)?;
            replay(Position::initial(), &moves)
        }
    }
}

pub(crate) fn fmt_move(mv: Option<u8>) -> String {
    match mv {
        Some(sq) => square_to_string(sq),
        None => String::from("pass"),
    }
}

fn cmd_search(rest: &[String]) -> i32 {
    let (positional, flags) = match split_flags(rest) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let Some(Ok(depth)) = positional.first().map(|s| s.parse::<u32>()) else {
        eprintln!("usage: versor-cli search <depth> [pos] [--weights FILE] [--tt MB] [--solve-at N]");
        return 2;
    };
    let run = || -> Result<i32, String> {
        let tt_mb: usize = flag_num(&flags, "tt", 64)?;
        let solve_at: u32 = flag_num(&flags, "solve-at", 20)?;
        let pos = position_from_arg(positional.get(1).map(|s| s.as_str()))?;
        print_board(&pos);
        if pos.is_game_over() {
            return Ok(0);
        }
        let empties = pos.empty().count_ones();
        if empties <= solve_at {
            println!(
                "{} empties <= solve threshold {}: using the exact endgame solver",
                empties, solve_at
            );
            // Move ordering as in `solve`: by the pattern evaluation when
            // a weight file is given.
            let mut solver = Solver::new(tt_mb);
            if let Some(wpath) = flags.get("weights") {
                solver.set_eval_order(Box::new(PatternEval::from_file(wpath)?));
            }
            let r = solver.solve_exact(&pos, None).expect("no limit set");
            let secs = r.elapsed.as_secs_f64();
            println!(
                "solve: score {:+}, best {}, {} nodes in {:.3} s ({:.2} Mnodes/s)",
                r.score,
                fmt_move(r.best_move),
                r.nodes,
                secs,
                r.nodes as f64 / secs / 1e6
            );
            return Ok(0);
        }
        // The midgame search runs with the engine defaults, as in play
        // (serve, arena, enginematch): `--weights` only chooses the
        // evaluation.
        let r = if let Some(wpath) = flags.get("weights") {
            let mut searcher = Searcher::new(PatternEval::from_file(wpath)?, tt_mb);
            searcher.search(&pos, depth)
        } else {
            let mut searcher = Searcher::new(PlaceholderEval, tt_mb);
            searcher.search(&pos, depth)
        };
        let secs = r.elapsed.as_secs_f64();
        let pv: Vec<String> = r.pv.iter().map(|&m| square_to_string(m)).collect();
        println!(
            "search depth {}: value {:+}, best {}, {} nodes in {:.3} s ({:.2} Mnodes/s)",
            r.depth,
            r.value,
            fmt_move(r.best_move),
            r.nodes,
            secs,
            r.nodes as f64 / secs / 1e6
        );
        println!("pv: {}", pv.join(" "));
        Ok(0)
    };
    match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            1
        }
    }
}

fn cmd_solve(rest: &[String]) -> i32 {
    let (positional, flags) = match split_flags(rest) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let run = || -> Result<i32, String> {
        let tt_mb: usize = flag_num(&flags, "tt", 64)?;
        let limit_secs: u64 = flag_num(&flags, "limit", 0)?;
        let limit = if limit_secs > 0 {
            Some(Duration::from_secs(limit_secs))
        } else {
            None
        };
        let pos = position_from_arg(positional.first().map(|s| s.as_str()))?;
        print_board(&pos);
        if pos.is_game_over() {
            return Ok(0);
        }
        let mut solver = Solver::new(tt_mb);
        if let Some(wpath) = flags.get("weights") {
            solver.set_eval_order(Box::new(PatternEval::from_file(wpath)?));
        }
        if flags.contains_key("wld") {
            match solver.solve_wld(&pos, limit) {
                Some((wld, r)) => {
                    let name = match wld {
                        Wld::Win => "win",
                        Wld::Draw => "draw",
                        Wld::Loss => "loss",
                    };
                    let secs = r.elapsed.as_secs_f64();
                    println!(
                        "wld: {} for the side to move, best {}, {} nodes in {:.3} s",
                        name,
                        fmt_move(r.best_move),
                        r.nodes,
                        secs
                    );
                }
                None => println!("wld: timeout after {} s", limit_secs),
            }
        } else {
            match solver.solve_exact(&pos, limit) {
                Some(r) => {
                    let secs = r.elapsed.as_secs_f64();
                    println!(
                        "solve: score {:+}, best {}, {} nodes in {:.3} s ({:.2} Mnodes/s)",
                        r.score,
                        fmt_move(r.best_move),
                        r.nodes,
                        secs,
                        r.nodes as f64 / secs / 1e6
                    );
                }
                None => println!("solve: timeout after {} s", limit_secs),
            }
        }
        Ok(0)
    };
    match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            1
        }
    }
}

// FFO problem file: one problem per line, "id;board64;stm;score" with
// '#' comment lines. board64 uses X (black), O (white), - (empty) from
// a1 to h8; stm is X or O; score is the exact score for the side to
// move, or ? if unknown.
fn cmd_ffo(rest: &[String]) -> i32 {
    let (positional, flags) = match split_flags(rest) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let Some(path) = positional.first() else {
        eprintln!("usage: versor-cli ffo <file> [--limit SECS] [--tt MB] [--from N] [--to N]");
        return 2;
    };
    let run = || -> Result<i32, String> {
        let tt_mb: usize = flag_num(&flags, "tt", 64)?;
        let limit_secs: u64 = flag_num(&flags, "limit", 600)?;
        let from: u32 = flag_num(&flags, "from", 0)?;
        let to: u32 = flag_num(&flags, "to", u32::MAX)?;
        // A node budget replaces the wall-clock deadline when given, so
        // the run is reproducible under arbitrary machine load. Times
        // are still printed, but only as a reference figure.
        let max_nodes: Option<u64> = match flags.get("max-nodes") {
            Some(v) => Some(v.parse().map_err(|_| "bad --max-nodes")?),
            None => None,
        };
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        let mut solver = Solver::new(tt_mb);
        solver.set_node_limit(max_nodes);
        // The two exact cutoffs off, to compare node counts with and
        // without them under the same node budget.
        if flags.contains_key("no-prune") {
            solver.opts.use_stability = false;
            solver.opts.use_etc = false;
            println!("# exact cutoffs (stability, ETC) disabled (--no-prune).");
        }
        if let Some(wpath) = flags.get("weights") {
            solver.set_eval_order(Box::new(PatternEval::from_file(wpath)?));
        }
        if max_nodes.is_some() {
            println!(
                "# node-bounded run: {} nodes per problem, no wall-clock limit.",
                max_nodes.unwrap()
            );
            println!("# times are reference values only (they depend on the machine and load).");
        }
        let mut n_ok = 0u32;
        let mut n_ng = 0u32;
        let mut n_skip = 0u32;
        println!("| id | empties | expected | got | result | best | nodes | time_s |");
        println!("|---:|--------:|---------:|----:|:-------|:-----|------:|-------:|");
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split(';').collect();
            if fields.len() < 4 {
                return Err(format!("{}:{}: expected id;board;stm;score", path, lineno + 1));
            }
            let id: u32 = fields[0]
                .trim()
                .parse()
                .map_err(|_| format!("{}:{}: bad id", path, lineno + 1))?;
            if id < from || id > to {
                continue;
            }
            let board = fields[1].trim();
            let stm = fields[2].trim();
            let expected: Option<i32> = match fields[3].trim() {
                "?" => None,
                s => Some(
                    s.parse()
                        .map_err(|_| format!("{}:{}: bad score", path, lineno + 1))?,
                ),
            };
            let pos = Position::from_text(&format!("{}{}", board, stm))
                .map_err(|e| format!("{}:{}: {}", path, lineno + 1, e))?;
            let empties = pos.empty().count_ones();
            let deadline = if max_nodes.is_some() {
                None
            } else {
                Some(Duration::from_secs(limit_secs))
            };
            // Each problem from a fresh solver, so that its node count
            // does not depend on which problems were solved before it.
            solver.new_game();
            match solver.solve_exact(&pos, deadline) {
                Some(r) => {
                    let secs = r.elapsed.as_secs_f64();
                    let verdict = match expected {
                        Some(e) if e == r.score => {
                            n_ok += 1;
                            "ok"
                        }
                        Some(_) => {
                            n_ng += 1;
                            "MISMATCH"
                        }
                        None => "-",
                    };
                    println!(
                        "| {} | {} | {} | {:+} | {} | {} | {} | {:.1} |",
                        id,
                        empties,
                        expected.map_or(String::from("?"), |e| format!("{:+}", e)),
                        r.score,
                        verdict,
                        fmt_move(r.best_move),
                        r.nodes,
                        secs
                    );
                }
                None => {
                    n_skip += 1;
                    // Report the budget that actually stopped the solve.
                    let reason = match max_nodes {
                        Some(n) => format!("over {} nodes", n),
                        None => format!("over {} s", limit_secs),
                    };
                    println!(
                        "| {} | {} | {} | - | SKIP ({}) | - | - | - |",
                        id,
                        empties,
                        expected.map_or(String::from("?"), |e| format!("{:+}", e)),
                        reason
                    );
                }
            }
        }
        println!();
        let budget = match max_nodes {
            Some(n) => format!("{} nodes", n),
            None => format!("{} s", limit_secs),
        };
        println!(
            "summary: {} correct, {} mismatched, {} skipped (limit {} per problem)",
            n_ok, n_ng, n_skip, budget
        );
        if n_ng > 0 {
            eprintln!("error: score mismatches on solved problems; the solver has a bug");
            return Ok(1);
        }
        Ok(0)
    };
    match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            1
        }
    }
}


/// Per-engine counters for one arena run.
#[derive(Clone, Copy, Default)]
pub(crate) struct ArenaStats {
    pub nodes: u64,
    pub moves: u64,
    /// Moves decided by the exact endgame solver.
    pub solver_moves: u64,
    /// Solver attempts that ran out of node budget and fell back to the
    /// depth-limited search.
    pub solver_aborts: u64,
}

// One game; returns the final score from black's perspective.
//
// When `solve_at` is non-zero the engines switch to the exact endgame
// solver once the empty count drops to it, which is what a real playing
// engine does and what `versor-cli search --solve-at` documents. It is
// off by default: the comparisons between evaluation functions play
// every move, the endgame included, with the depth-limited search.
//
// Under a node budget the solver is given that same budget; if it runs
// out it falls back to the depth-limited search, which then gets the
// budget again. Both engines follow the identical rule, so a failed
// solve costs both sides the same way. This is the whole point of the
// exercise: pruning lets the solver finish at higher empty counts within
// the same budget, so the fallback fires less often.
//
// Every game starts from the same engine state (`new_game`): no
// transposition table entries, no history. A game's moves then depend
// only on its opening and the two engines, not on the games the same
// worker played before, so a match gives the same games whichever
// worker plays which game and however many play at once. The moves
// played are appended to `moves` (passes are not written).
#[allow(clippy::too_many_arguments)]
fn arena_game<A: Evaluator, B: Evaluator>(
    opening: &Position,
    black: &mut Searcher<A>,
    white: &mut Searcher<B>,
    black_solver: &mut Solver,
    white_solver: &mut Solver,
    depth: u32,
    node_budget: Option<u64>,
    solve_at: u32,
    stats: &mut [ArenaStats; 2], // [black engine, white engine]
    moves: &mut Vec<u8>,
) -> i32 {
    black.new_game();
    white.new_game();
    black_solver.new_game();
    white_solver.new_game();
    let mut pos = *opening;
    while !pos.is_game_over() {
        if pos.must_pass() {
            pos = pos.pass();
            continue;
        }
        let is_black = pos.stm == Color::Black;
        let side = if is_black { 0 } else { 1 };
        let empties = pos.empty().count_ones();

        if solve_at > 0 && empties <= solve_at {
            let solver = if is_black {
                &mut *black_solver
            } else {
                &mut *white_solver
            };
            solver.set_node_limit(node_budget);
            if let Some(r) = solver.solve_exact(&pos, None) {
                stats[side].nodes += r.nodes;
                stats[side].moves += 1;
                stats[side].solver_moves += 1;
                let mv = r.best_move.expect("side to move has a legal move");
                pos = pos.play(mv).expect("solver returned a legal move");
                moves.push(mv);
                continue;
            }
            // Budget exhausted: fall through to the depth-limited search.
            stats[side].solver_aborts += 1;
            if let Some(n) = node_budget {
                stats[side].nodes += n;
            }
        }

        let r = match (is_black, node_budget) {
            (true, None) => black.search(&pos, depth),
            (false, None) => white.search(&pos, depth),
            (true, Some(n)) => black.search_nodes(&pos, n),
            (false, Some(n)) => white.search_nodes(&pos, n),
        };
        stats[side].nodes += r.nodes;
        stats[side].moves += 1;
        let mv = r.best_move.expect("side to move has a legal move");
        pos = pos.play(mv).expect("search returned a legal move");
        moves.push(mv);
    }
    match pos.stm {
        Color::Black => pos.final_score(),
        Color::White => -pos.final_score(),
    }
}

/// One arena game as it was played: engine A's colour, A's final disc
/// difference, every move from the initial position (the opening
/// included) and the nodes each engine searched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ArenaGame {
    pub a_black: bool,
    pub score_a: i32,
    pub moves: Vec<u8>,
    /// [engine A, engine B]
    pub nodes: [u64; 2],
}

impl ArenaGame {
    /// The game's line in an arena log: pair index, A's colour, A's disc
    /// difference, the moves, then the nodes of A and of B. The first
    /// four fields are those of an enginematch log, so the same tools
    /// read both.
    fn log_line(&self, pair: u64) -> String {
        format!(
            "{} {} {:+} {} {} {}\n",
            pair,
            if self.a_black { 'B' } else { 'W' },
            self.score_a,
            move_seq_to_string(&self.moves),
            self.nodes[0],
            self.nodes[1]
        )
    }
}

/// Wins, draws and losses for engine A and the pair scores, from the
/// games in game-number order. Done after the run rather than as games
/// finish, so that the result is the same however the games were shared
/// out among the workers.
pub(crate) fn arena_tally(games: &[Option<ArenaGame>]) -> (u64, u64, u64, Vec<f64>) {
    let (mut w, mut d, mut l) = (0u64, 0u64, 0u64);
    let mut pair_scores = Vec::new();
    for pair in games.chunks(2) {
        let mut sum = 0.0;
        let mut n = 0u64;
        for g in pair.iter().flatten() {
            let outcome = match g.score_a.cmp(&0) {
                std::cmp::Ordering::Greater => {
                    w += 1;
                    1.0
                }
                std::cmp::Ordering::Equal => {
                    d += 1;
                    0.5
                }
                std::cmp::Ordering::Less => {
                    l += 1;
                    0.0
                }
            };
            sum += outcome;
            n += 1;
        }
        if n > 0 {
            pair_scores.push(sum / n as f64);
        }
    }
    (w, d, l, pair_scores)
}

/// Play the arena match. Returns the games by game number (pair k has
/// games 2k, with A as black, and 2k+1), the per-engine counters and the
/// wall time. With `log`, each game's line is written as soon as the
/// game ends, so a run that is stopped keeps what it played.
#[allow(clippy::too_many_arguments)]
fn arena_run<EB: Evaluator, F: Fn() -> Searcher<EB> + Sync>(
    weights_a: &str,
    mk_b: F,
    games: u64,
    depth: u32,
    threads: usize,
    seed: u64,
    opening_plies: u32,
    tt_mb: usize,
    pairs: u64,
    node_budget: Option<u64>,
    mpc_a: Option<(String, f64)>,
    b_plain_search: bool,
    b_no_prune: bool,
    solve_at: u32,
    log: Option<&std::sync::Mutex<std::fs::File>>,
) -> (Vec<Option<ArenaGame>>, [ArenaStats; 2], f64) {
    let mpc_a = &mpc_a;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    let next_pair = AtomicU64::new(0);
    let played: Mutex<Vec<Option<ArenaGame>>> = Mutex::new(vec![None; games as usize]);
    let engine_stats = Mutex::new([ArenaStats::default(); 2]);
    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let pat = PatternEval::from_file(weights_a).expect("weights");
                let mut a = Searcher::new(pat, tt_mb);
                if let Some((path, t)) = mpc_a {
                    a.set_mpc(
                        versor_search::MpcParams::from_file(path).expect("mpc params"),
                        *t,
                    );
                }
                let mut b = mk_b();
                if b_plain_search {
                    // Engine B without killers, history and aspiration,
                    // so A (defaults) vs B measures what the three
                    // ordering heuristics are worth in play.
                    b.opts.use_killers = false;
                    b.opts.use_history = false;
                    b.opts.use_aspiration = false;
                }
                if b_no_prune {
                    // Engine B without the two exact cutoffs, so A
                    // (defaults) vs B measures them. Only meaningful at a
                    // fixed node budget: at a fixed depth the two return
                    // the same values by construction.
                    b.opts.use_etc = false;
                    b.opts.use_stability = false;
                }
                // Endgame solvers, one per engine, so the pruning options
                // can differ between them exactly as the searchers do.
                let mut a_solver = Solver::new(tt_mb);
                let mut b_solver = Solver::new(tt_mb);
                if solve_at > 0 {
                    a_solver.set_eval_order(Box::new(
                        PatternEval::from_file(weights_a).expect("weights"),
                    ));
                    b_solver.set_eval_order(Box::new(
                        PatternEval::from_file(weights_a).expect("weights"),
                    ));
                    if b_no_prune {
                        b_solver.opts.use_stability = false;
                        b_solver.opts.use_etc = false;
                    }
                }
                loop {
                    let k = next_pair.fetch_add(1, Ordering::Relaxed);
                    if k >= pairs {
                        break;
                    }
                    let (opening, opening_moves) = opening_with_moves(seed, k, opening_plies);
                    for a_is_black in [true, false] {
                        let g = k * 2 + if a_is_black { 0 } else { 1 };
                        if g >= games {
                            break;
                        }
                        let mut st = [ArenaStats::default(); 2];
                        let mut moves = opening_moves.clone();
                        let score_black = if a_is_black {
                            arena_game(
                                &opening, &mut a, &mut b, &mut a_solver, &mut b_solver,
                                depth, node_budget, solve_at, &mut st, &mut moves,
                            )
                        } else {
                            arena_game(
                                &opening, &mut b, &mut a, &mut b_solver, &mut a_solver,
                                depth, node_budget, solve_at, &mut st, &mut moves,
                            )
                        };
                        let (ai, bi) = if a_is_black { (0, 1) } else { (1, 0) };
                        let game = ArenaGame {
                            a_black: a_is_black,
                            score_a: if a_is_black { score_black } else { -score_black },
                            moves,
                            nodes: [st[ai].nodes, st[bi].nodes],
                        };
                        if let Some(log) = log {
                            let _ = log.lock().unwrap().write_all(game.log_line(k).as_bytes());
                        }
                        played.lock().unwrap()[g as usize] = Some(game);
                        let mut es = engine_stats.lock().unwrap();
                        for (dst, src) in [(0usize, ai), (1usize, bi)] {
                            es[dst].nodes += st[src].nodes;
                            es[dst].moves += st[src].moves;
                            es[dst].solver_moves += st[src].solver_moves;
                            es[dst].solver_aborts += st[src].solver_aborts;
                        }
                    }
                }
            });
        }
    });
    (
        played.into_inner().unwrap(),
        engine_stats.into_inner().unwrap(),
        start.elapsed().as_secs_f64(),
    )
}

fn elo_from_score(s: f64) -> f64 {
    let s = s.clamp(1e-9, 1.0 - 1e-9);
    400.0 * (s / (1.0 - s)).log10()
}

// Arena: pattern evaluation (candidate) vs placeholder evaluation
// (baseline) from shared random openings, both colors per opening,
// fixed search depth. The whole run is deterministic for a given
// (seed, games, depth): engines are deterministic, so variance comes
// from opening sampling; the confidence interval is computed over
// opening pairs (the independent units).
fn cmd_arena(rest: &[String]) -> i32 {
    let (_, flags) = match split_flags(rest) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let run = || -> Result<i32, String> {
        let weights = flags
            .get("weights")
            .ok_or("arena requires --weights FILE")?
            .clone();
        let games: u64 = flag_num(&flags, "games", 1000)?;
        let depth: u32 = flag_num(&flags, "depth", 6)?;
        // Games played at once. The games do not depend on it (each one
        // starts from a fresh engine state), only the wall time does.
        let threads: usize = flag_num(&flags, "threads", 4)?;
        let threads = threads.clamp(1, 8);
        let seed: u64 = flag_num(&flags, "seed", 3)?;
        let opening_plies: u32 = flag_num(&flags, "opening", 8)?;
        let tt_mb: usize = flag_num(&flags, "tt", 32)?;
        let pairs = games.div_ceil(2);
        PatternEval::from_file(&weights)?; // validate before spawning
        let log = match flags.get("log") {
            Some(path) => Some(std::sync::Mutex::new(
                std::fs::File::create(path).map_err(|e| format!("{}: {}", path, e))?,
            )),
            None => None,
        };
        let label_b = flags.get("weights-b").map(|s| s.as_str());
        let node_budget: Option<u64> = match flags.get("nodes") {
            Some(v) => Some(v.parse().map_err(|_| "bad --nodes")?),
            None => None,
        };
        let mpc_a: Option<(String, f64)> = match flags.get("mpc") {
            Some(path) => Some((path.clone(), flag_num(&flags, "mpc-t", 2.0f64)?)),
            None => None,
        };
        let b_plain_search = flags.contains_key("b-plain-search");
        let b_no_prune = flags.contains_key("b-no-prune");
        // 0 (the default) disables the switch to the endgame solver; see
        // `arena_game`.
        let arena_solve_at: u32 = flag_num(&flags, "solve-at", 0)?;
        let (played, es, dt) = if let Some(wb) = label_b {
            PatternEval::from_file(wb)?;
            arena_run(
                &weights,
                || Searcher::new(PatternEval::from_file(wb).expect("weights-b"), tt_mb),
                games, depth, threads, seed, opening_plies, tt_mb, pairs,
                node_budget, mpc_a, b_plain_search, b_no_prune, arena_solve_at, log.as_ref(),
            )
        } else {
            arena_run(
                &weights,
                || Searcher::new(PlaceholderEval, tt_mb),
                games, depth, threads, seed, opening_plies, tt_mb, pairs,
                node_budget, mpc_a, b_plain_search, b_no_prune, arena_solve_at, log.as_ref(),
            )
        };
        let (w, d, l, pair_scores) = arena_tally(&played);
        let n_games = w + d + l;
        let n_pairs = pair_scores.len() as f64;
        let s = pair_scores.iter().sum::<f64>() / n_pairs;
        let var = pair_scores.iter().map(|x| (x - s) * (x - s)).sum::<f64>() / n_pairs;
        let se = (var / n_pairs).sqrt();
        let (lo, hi) = ((s - 1.96 * se).max(0.0), (s + 1.96 * se).min(1.0));
        println!(
            "arena: pattern eval vs {}, depth {}, {} games ({} pairs, opening {} plies, seed {})",
            label_b.unwrap_or("placeholder"), depth, n_games, pair_scores.len(), opening_plies, seed
        );
        println!(
            "result for engine A: {} wins, {} draws, {} losses (score {:.4})",
            w, d, l, s
        );
        println!(
            "elo {:+.1} (95 pct CI {:+.1} .. {:+.1}), draws {}",
            elo_from_score(s),
            elo_from_score(lo),
            elo_from_score(hi),
            d
        );
        println!(
            "speed: A {:.2} Mnodes/s over {} moves, B {:.2} Mnodes/s over {} moves, {:.0} s total",
            es[0].nodes as f64 / dt / 1e6 / threads as f64,
            es[0].moves,
            es[1].nodes as f64 / dt / 1e6 / threads as f64,
            es[1].moves,
            dt
        );
        println!("note: per-engine Mnodes/s is approximate (wall clock shared)");
        // Node totals do not depend on the machine or on how many games
        // ran at once, so two runs of the same match can be compared on
        // them.
        println!("nodes: A {}, B {}", es[0].nodes, es[1].nodes);
        if arena_solve_at > 0 {
            // Evidence that the endgame solver was actually on the path,
            // and how often each side had to fall back to the search.
            println!(
                "solver: switch at {} empties. A {} of {} moves solved exactly ({} fell back), \
                 B {} of {} solved ({} fell back)",
                arena_solve_at,
                es[0].solver_moves, es[0].moves, es[0].solver_aborts,
                es[1].solver_moves, es[1].moves, es[1].solver_aborts,
            );
        } else {
            println!(
                "solver: not used (--solve-at 0). The engines play every move with the \
                 depth-limited search, including the endgame."
            );
        }
        Ok(0)
    };
    match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            1
        }
    }
}

// Openings read from a file, one game per line, as a move string like
// "f5d6c4d3c2b3b4b5". The XOT lists are in exactly this form.
//
// Using a fixed list instead of generating openings changes what the
// measurement covers: XOT positions are chosen to be near equal, while
// random plies are not. That is the point of the comparison, so the
// two paths deliberately share everything else.
fn load_openings(path: &str) -> Result<Vec<Vec<u8>>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let moves = versor_core::notation::parse_move_seq(line)
            .map_err(|e| format!("{}:{}: {}", path, n + 1, e))?;
        // Replay it now: an opening that does not play out is a broken
        // input, and finding that out mid-match would void games.
        let mut p = Position::initial();
        for &sq in &moves {
            if p.must_pass() {
                p = p.pass();
            }
            p = p
                .play(sq)
                .map_err(|_| format!("{}:{}: illegal move in {:?}", path, n + 1, line))?;
        }
        if p.is_game_over() {
            return Err(format!("{}:{}: opening ends the game", path, n + 1));
        }
        out.push(moves);
    }
    if out.is_empty() {
        return Err(format!("{}: no openings", path));
    }
    Ok(out)
}

// Deterministic opening for pair k of a match (arena and enginematch):
// play `plies` random moves from the initial position, retried if the
// game ends early (which cannot realistically happen at 8 plies). The
// moves are returned for the game logs.
fn opening_with_moves(seed: u64, pair: u64, plies: u32) -> (Position, Vec<u8>) {
    let mut rng = Xorshift64Star::new(
        seed ^ (pair + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
    );
    versor_core::sample::random_opening(&mut rng, plies)
}

pub(crate) fn expand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        format!("{}/{}", std::env::var("HOME").unwrap_or_default(), rest)
    } else {
        path.to_string()
    }
}

// Load an engine's launch spec (binary + arg template) from a config
// file (default config/engines.json) by name.
fn load_engine_spec(
    config_path: &str,
    name: &str,
    level: u32,
    timeout: Duration,
) -> Result<extengine::EngineSpec, String> {
    let text = std::fs::read_to_string(config_path)
        .map_err(|e| format!("{}: {}", config_path, e))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {}", config_path, e))?;
    let e = cfg
        .get(name)
        .ok_or_else(|| format!("engine {:?} not in {}", name, config_path))?;
    let binary = expand_home(
        e.get("binary")
            .and_then(|v| v.as_str())
            .ok_or("engine spec missing binary")?,
    );
    let args = e
        .get("args")
        .and_then(|v| v.as_array())
        .ok_or("engine spec missing args")?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    // Optional clock spec, used only for time-control play. The
    // "before_setboard" flag is load-bearing: Edax resets its clock on
    // setboard (measured; see `extengine::ClockSpec`), so its remaining
    // time has to be sent first or it plays every move with a full
    // clock.
    let clock = e.get("clock").map(|c| extengine::ClockSpec {
        template: c
            .get("template")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        before_setboard: c
            .get("before_setboard")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        margin: true,
    });
    Ok(extengine::EngineSpec {
        name: name.to_string(),
        binary,
        args,
        level,
        reply_timeout: timeout,
        clock,
    })
}

fn cmd_enginematch(rest: &[String]) -> i32 {
    let (_, flags) = match split_flags(rest) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let run = || -> Result<i32, String> {
        let engine_name = flags
            .get("engine")
            .cloned()
            .unwrap_or_else(|| "edax".to_string());
        let config_path = flags
            .get("engines")
            .cloned()
            .unwrap_or_else(|| "config/engines.json".to_string());
        let level: u32 = flag_num(&flags, "level", 3)?;
        let timeout = Duration::from_secs(flag_num(&flags, "timeout", 120u64)?);
        // An explicit --binary overrides the config path for the engine.
        let mut spec = load_engine_spec(&config_path, &engine_name, level, timeout)?;
        if let Some(bin) = flags.get("binary") {
            spec.binary = expand_home(bin);
        }
        // Tell the engine its true remaining time, without the margin
        // (to show why the margin is there).
        if flags.contains_key("no-clock-margin") {
            if let Some(c) = spec.clock.as_mut() {
                c.margin = false;
            }
        }
        // Wall-clock time control: a whole-game clock per player, the
        // format human tournaments use.
        //
        // Concurrent games are allowed here but default to one, because
        // they share the machine and could change the very thing being
        // measured. Four at once was checked on a dedicated machine: the
        // same ten games alone and beside three others reached depth
        // 12.95 vs 12.89, with nodes per move and nodes/s no lower. The
        // transposition table is far larger than the last-level cache
        // either way, which is why the neighbours cost nothing.
        let game_time_s: u64 = flag_num(&flags, "game-time", 0u64)?;
        let time_control = if game_time_s > 0 {
            Some(extengine::TimeControl {
                game_time: Duration::from_secs(game_time_s),
                solver_factor: flag_num(&flags, "solver-factor", 3u32)?,
                all_moves: flags.contains_key("alloc-all-moves"),
            })
        } else {
            None
        };
        if time_control.is_some() && spec.clock.is_none() {
            return Err(format!(
                "engine {:?} has no \"clock\" entry in {}; \
                 --game-time needs one to tell the engine its remaining time",
                engine_name, config_path
            ));
        }
        let params = extengine::MatchParams {
            weights: flags
                .get("weights")
                .ok_or("enginematch requires --weights")?
                .clone(),
            games: flag_num(&flags, "games", 1000)?,
            depth: flag_num(&flags, "depth", 8)?,
            threads: if time_control.is_some() {
                flag_num(&flags, "threads", 1usize)?.clamp(1, 8)
            } else {
                flag_num(&flags, "threads", 4usize)?.clamp(1, 8)
            },
            seed: flag_num(&flags, "seed", 5)?,
            opening_plies: flag_num(&flags, "opening", 8)?,
            tt_mb: flag_num(&flags, "tt", 32)?,
            engine: spec.clone(),
            log_path: flags.get("log").cloned(),
            probe_games: 200,
            probe_rate: 0.05,
            stop_games: 500,
            // versor switches to the exact solver here, as the opposing
            // engines do at their levels. 0 disables the switch: versor
            // then plays the endgame with the depth-limited search while
            // the opponent still solves it.
            solve_at: flag_num(&flags, "solve-at", 20u32)?,
            time_control,
            mpc: match flags.get("mpc") {
                Some(path) => {
                    versor_search::MpcParams::from_file(path)?; // validate before spawning
                    Some((path.clone(), flag_num(&flags, "mpc-t", 2.0f64)?))
                }
                None => None,
            },
        };
        // Openings either come from a file (XOT and the like) or are
        // generated. Everything else about the match is the same.
        let openings = match flags.get("openings") {
            Some(path) => Some(load_openings(&expand_home(path))?),
            None => None,
        };
        let r = match &openings {
            Some(list) => {
                let n = list.len() as u64;
                println!("openings: {} from {}", n, flags["openings"]);
                extengine::run_match(&params, |seed, pair, _plies| {
                    let mvs = &list[((seed.wrapping_add(pair)) % n) as usize];
                    let mut p = Position::initial();
                    for &sq in mvs {
                        if p.must_pass() {
                            p = p.pass();
                        }
                        p = p.play_unchecked(sq);
                    }
                    (p, mvs.clone())
                })?
            }
            None => extengine::run_match(&params, opening_with_moves)?,
        };
        if let Some((path, t)) = &params.mpc {
            println!("mpc: {} (t = {})", path, t);
        }
        let n_games = r.wins + r.draws + r.losses;
        let n_pairs = r.pair_scores.len() as f64;
        let s = r.pair_scores.iter().sum::<f64>() / n_pairs.max(1.0);
        let var = r
            .pair_scores
            .iter()
            .map(|x| (x - s) * (x - s))
            .sum::<f64>()
            / n_pairs.max(1.0);
        let se = (var / n_pairs.max(1.0)).sqrt();
        let (lo, hi) = ((s - 1.96 * se).max(0.0), (s + 1.96 * se).min(1.0));
        match params.time_control {
            Some(tc) => println!(
                "enginematch: versor vs {} level {}, {} s per player per game, \
                 {} concurrent ({} games, {} pairs, seed {}{})",
                spec.name,
                spec.level,
                tc.game_time.as_secs(),
                params.threads,
                n_games,
                r.pair_scores.len(),
                params.seed,
                if r.reference_only { ", REFERENCE ONLY (early stop)" } else { "" }
            ),
            None => println!(
                "enginematch: versor depth {} vs {} level {} ({} games, {} pairs, seed {}{})",
                params.depth,
                spec.name,
                spec.level,
                n_games,
                r.pair_scores.len(),
                params.seed,
                if r.reference_only { ", REFERENCE ONLY (early stop)" } else { "" }
            ),
        }
        println!(
            "result for versor: {} wins, {} draws, {} losses, {} voided (score {:.4})",
            r.wins, r.draws, r.losses, r.voided, s
        );
        println!(
            "elo {:+.1} (95 pct CI {:+.1} .. {:+.1})",
            elo_from_score(s),
            elo_from_score(lo),
            elo_from_score(hi)
        );
        if params.time_control.is_some() {
            let per = |ms: u64, n: u64| if n == 0 { 0.0 } else { ms as f64 / n as f64 / 1000.0 };
            println!(
                "clock: versor used {:.1} s over {} moves ({:.2} s/move), \
                 {} used {:.1} s over {} moves ({:.2} s/move)",
                r.versor_time_ms as f64 / 1000.0,
                r.versor_move_count,
                per(r.versor_time_ms, r.versor_move_count),
                spec.name,
                r.engine_time_ms as f64 / 1000.0,
                r.engine_move_count,
                per(r.engine_time_ms, r.engine_move_count),
            );
            println!(
                "time losses: versor {}, {} {} (counted as losses, not voided)",
                r.versor_flags, spec.name, r.engine_flags
            );
            // How far versor actually got in its own clock. Two runs of
            // the same games can be compared on these: a machine that
            // is busy takes depth away without showing up anywhere else.
            if r.versor_search_moves > 0 {
                let moves = r.versor_search_moves as f64;
                let search_ms = r.versor_time_ms as f64;
                println!(
                    "versor search: avg depth {:.2} over {} moves, \
                     {:.2} Mnodes/move, {:.3} Mnodes/s, \
                     solver decided {} of {} moves ({:.1} pct)",
                    r.versor_depth_sum as f64 / moves,
                    r.versor_search_moves,
                    r.versor_nodes as f64 / moves / 1e6,
                    r.versor_nodes as f64 / (search_ms / 1000.0) / 1e6,
                    r.solver_moves,
                    r.versor_move_count,
                    100.0 * r.solver_moves as f64 / r.versor_move_count.max(1) as f64,
                );
            }
            if r.versor_flags > 0 {
                println!(
                    "WARNING: versor ran out of time in {} game(s). The time manager \
                     is at fault; stop and fix it before trusting this result.",
                    r.versor_flags
                );
            }
        }
        if params.solve_at > 0 {
            println!(
                "solver: versor switched to the exact solver at {} empties, {} moves solved",
                params.solve_at, r.solver_moves
            );
        } else {
            println!(
                "solver: not used (--solve-at 0). versor plays the endgame with the \
                 depth-limited search while the opponent solves it exactly."
            );
        }
        Ok(0)
    };
    match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            1
        }
    }
}

fn cmd_serve(rest: &[String]) -> i32 {
    let (_, flags) = match split_flags(rest) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let Some(weights) = flags.get("weights").cloned() else {
        eprintln!("usage: versor-cli serve --weights FILE [--mpc PARAMS] [--mpc-t T] [--tt MB] [--hint-depth D]");
        return 2;
    };
    let solve_at: u32 = match flag_num(&flags, "solve-at", 20u32) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let solve_ms: u64 = match flag_num(&flags, "solve-ms", 3000u64) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let cfg = serve::ServeConfig {
        weights,
        mpc: flags.get("mpc").cloned(),
        mpc_t: flag_num(&flags, "mpc-t", 2.0f64).unwrap_or(2.0),
        tt_mb: flag_num(&flags, "tt", 128usize).unwrap_or(128),
        hint_depth: flag_num(&flags, "hint-depth", 6u32).unwrap_or(6),
        solve_at,
        solve_ms,
    };
    serve::run(cfg)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    let rest = if args.is_empty() { &args[..] } else { &args[1..] };
    let code = match cmd {
        "help" | "--help" | "-h" => {
            usage();
            0
        }
        "show" => cmd_show(rest),
        "play" => cmd_play(rest),
        "random" => cmd_random(rest),
        "perft" => cmd_perft(rest),
        "bench" => cmd_bench(rest),
        "search" => cmd_search(rest),
        "solve" => cmd_solve(rest),
        "ffo" => cmd_ffo(rest),
        "arena" => cmd_arena(rest),
        "enginematch" => cmd_enginematch(rest),
        "prune" => prune::cmd_prune(rest),
        "serve" => cmd_serve(rest),
        other => {
            eprintln!("unknown command: {}", other);
            usage();
            2
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod arena_tests {
    // An arena match gives the same games however many are played at
    // once and in whatever order: every game starts from a fresh engine
    // state, so a worker's earlier games leave nothing behind.

    use super::*;

    const WEIGHTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/pattern_v2.bin");
    const MPC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/mpc_params.txt");

    fn present() -> bool {
        std::fs::metadata(WEIGHTS).is_ok() && std::fs::metadata(MPC).is_ok()
    }

    fn run(threads: usize, nodes: Option<u64>, mpc: bool) -> Vec<Option<ArenaGame>> {
        let games = 12;
        arena_run(
            WEIGHTS,
            || Searcher::new(PatternEval::from_file(WEIGHTS).expect("weights"), 1),
            games,
            3,
            threads,
            7,
            8,
            1,
            games.div_ceil(2),
            nodes,
            if mpc { Some((MPC.to_string(), 2.0)) } else { None },
            false,
            false,
            12,
            None,
        )
        .0
    }

    #[test]
    fn games_do_not_depend_on_how_many_are_played_at_once() {
        if !present() {
            eprintln!("weights not present; skipping");
            return;
        }
        let one = run(1, None, false);
        assert!(one.iter().all(|g| g.is_some()));
        for threads in [2, 4, 8] {
            assert_eq!(one, run(threads, None, false), "fixed depth, {} at once", threads);
        }
        // A node budget makes every move depend on the node counts of
        // the iterations, so anything left over from an earlier game
        // would show; Multi-ProbCut on A as in the MPC comparison.
        let one = run(1, Some(3000), true);
        for threads in [2, 8] {
            assert_eq!(one, run(threads, Some(3000), true), "node budget, {} at once", threads);
        }
    }

    #[test]
    fn games_do_not_depend_on_their_order() {
        if !present() {
            eprintln!("weights not present; skipping");
            return;
        }
        let play_all = |order: &[u64]| {
            let mut a = Searcher::new(PatternEval::from_file(WEIGHTS).unwrap(), 1);
            a.set_mpc(versor_search::MpcParams::from_file(MPC).unwrap(), 2.0);
            let mut b = Searcher::new(PatternEval::from_file(WEIGHTS).unwrap(), 1);
            let mut a_solver = Solver::new(1);
            let mut b_solver = Solver::new(1);
            let mut out = std::collections::BTreeMap::new();
            for &k in order {
                let (opening, opening_moves) = opening_with_moves(7, k, 8);
                let mut moves = opening_moves.clone();
                let mut st = [ArenaStats::default(); 2];
                let score = arena_game(
                    &opening, &mut a, &mut b, &mut a_solver, &mut b_solver, 3, Some(3000), 12,
                    &mut st, &mut moves,
                );
                out.insert(k, (score, moves, st[0].nodes, st[1].nodes));
            }
            out
        };
        let forward: Vec<u64> = (0..6).collect();
        let backward: Vec<u64> = (0..6).rev().collect();
        assert_eq!(play_all(&forward), play_all(&backward));
    }
}

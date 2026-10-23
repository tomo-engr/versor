// dataprep: convert game records (WThor .wtb files and f5d6 transcript
// text files) into training records for the Python trainer.
//
// Record format (132 bytes, little endian):
//   60 x u16  canonical table-local pattern indices of every pattern
//             instance (pattern::N_INSTANCES, versor-eval order)
//   1 x i16   final score from the side to move's perspective
//             (final_score convention: empties to the winner)
//   1 x u8    phase bucket 0..14 (disc count; the trainer recomputes
//             it from the indices and does not read this byte)
//   1 x u8    source id (0 = wthor, 1 = egaroucid v0002, 2 = transcript)
//   1 x u32   low bits of the game hash (for game-granular data
//             fraction selection in the trainer)
//   1 x u8    mobility of the side to move (legal move count)
//   1 x u8    mobility of the opponent
//   1 x u8    parity (number of empty squares mod 2)
//   1 x u8    padding (zero)
//
// Games are split into train shards and a validation set BY GAME:
// a deterministic hash of (file name, game index) decides val
// membership (hash % val_mod == 0) and the train shard, so no game
// contributes positions to both sides regardless of thread count.
// Positions played during the random opening section are excluded:
// for the Egaroucid v0002 series the folder name gives the number of
// random plies N; for the 2023 transcript series N is between 10 and
// 19 per game (not recorded), so a conservative 19 is used; WThor
// games are tournament games with no random section (0).
//
// The WThor parser is self-implemented from the public format
// description: 16 byte header with a u32 game count at offset 4,
// then 68 byte records (6 bytes ids, real score, theoretical score,
// 60 move bytes, move byte = 10 * column + row, 1-based, 0 = end).
// Replay legality is the effective format check: any decoding error
// makes the replay fail and the game is dropped and counted.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use versor_core::{notation, Color, Position};
use versor_eval::pattern::{extract_indices, phase_of, N_INSTANCES};

const RECORD_BYTES: usize = N_INSTANCES * 2 + 2 + 1 + 1 + 4 + 4;

#[derive(Default, Clone)]
struct Stats {
    games: u64,
    games_val: u64,
    positions: u64,
    positions_val: u64,
    replay_errors: u64,
    score_mismatches: u64,
    by_phase: [u64; 15],
}

struct Job {
    path: PathBuf,
    source: u8,
    skip_plies: u32,
    kind: JobKind,
}

#[derive(Clone, Copy, PartialEq)]
enum JobKind {
    Wthor,
    Transcript,
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct Writers {
    shards: Vec<BufWriter<File>>,
    val: BufWriter<File>,
    /// Optional cross-check dump: flat board text, score and indices
    /// of the first N validation positions (for the Python/Rust
    /// inference parity test vectors).
    boards: Option<BufWriter<File>>,
    boards_left: u64,
}

fn emit_game(
    moves: &[u8],
    skip_plies: u32,
    source: u8,
    game_hash: u64,
    val_mod: u64,
    n_shards: usize,
    writers: &Mutex<Writers>,
    stats: &mut Stats,
    wthor_real_score: Option<i32>,
) {
    // Replay the whole game first to get the final score.
    let mut positions: Vec<Position> = Vec::with_capacity(moves.len());
    let mut p = Position::initial();
    for &sq in moves {
        if p.must_pass() {
            p = p.pass();
        }
        positions.push(p);
        match p.play(sq) {
            Ok(next) => p = next,
            Err(_) => {
                stats.replay_errors += 1;
                return;
            }
        }
    }
    if !p.is_game_over() {
        // Incomplete record (WThor has a few); drop it.
        stats.replay_errors += 1;
        return;
    }
    let score_stm_final = p.final_score();
    let score_black = match p.stm {
        Color::Black => score_stm_final,
        Color::White => -score_stm_final,
    };
    if let Some(real) = wthor_real_score {
        // WThor real score = black disc count with empties to the
        // winner = (64 + diff) / 2. Drop games whose stored score
        // disagrees with the replay (bad records).
        if (64 + score_black) / 2 != real {
            stats.score_mismatches += 1;
            return;
        }
    }

    let is_val = game_hash % val_mod == 0;
    let shard = ((game_hash >> 8) % n_shards as u64) as usize;
    stats.games += 1;
    if is_val {
        stats.games_val += 1;
    }

    let mut buf: Vec<u8> = Vec::with_capacity(positions.len() * RECORD_BYTES);
    let mut idx = [0u16; N_INSTANCES];
    for (ply, pos) in positions.iter().enumerate() {
        if (ply as u32) < skip_plies {
            continue;
        }
        let score = match pos.stm {
            Color::Black => score_black,
            Color::White => -score_black,
        };
        extract_indices(pos, &mut idx);
        let phase = phase_of(pos) as u8;
        for v in idx {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&(score as i16).to_le_bytes());
        buf.push(phase);
        buf.push(source);
        buf.extend_from_slice(&((game_hash >> 32) as u32).to_le_bytes());
        let own_mob = pos.legal_moves().count_ones() as u8;
        let opp_mob = versor_core::bitboard::legal_moves(pos.opp, pos.own).count_ones() as u8;
        buf.push(own_mob);
        buf.push(opp_mob);
        buf.push((pos.empty().count_ones() & 1) as u8);
        buf.push(0);
        stats.positions += 1;
        if is_val {
            stats.positions_val += 1;
        }
        stats.by_phase[phase as usize] += 1;
    }
    if buf.is_empty() {
        return;
    }
    let mut w = writers.lock().unwrap();
    if is_val {
        w.val.write_all(&buf).unwrap();
        if w.boards_left > 0 && w.boards.is_some() {
            let mut lines = String::new();
            for (ply, pos) in positions.iter().enumerate() {
                if (ply as u32) < skip_plies || w.boards_left == 0 {
                    continue;
                }
                w.boards_left -= 1;
                let flat: String = pos.to_text().chars().filter(|c| !c.is_whitespace()).collect();
                let score = match pos.stm {
                    Color::Black => score_black,
                    Color::White => -score_black,
                };
                extract_indices(pos, &mut idx);
                let idx_str: Vec<String> = idx.iter().map(|v| v.to_string()).collect();
                let om = pos.legal_moves().count_ones();
                let pm = versor_core::bitboard::legal_moves(pos.opp, pos.own).count_ones();
                let par = pos.empty().count_ones() & 1;
                lines.push_str(&format!(
                    "{}\t{}\t{}\t{},{},{}\n",
                    flat, score, idx_str.join(","), om, pm, par
                ));
            }
            w.boards.as_mut().unwrap().write_all(lines.as_bytes()).unwrap();
        }
    } else {
        w.shards[shard].write_all(&buf).unwrap();
    }
}

fn process_wthor(
    job: &Job,
    val_mod: u64,
    n_shards: usize,
    writers: &Mutex<Writers>,
    stats: &mut Stats,
    limit: &mut u64,
) {
    let mut raw = Vec::new();
    File::open(&job.path)
        .and_then(|mut f| f.read_to_end(&mut raw))
        .unwrap_or_else(|e| panic!("{}: {}", job.path.display(), e));
    if raw.len() < 16 {
        return;
    }
    let n_games = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    let body = &raw[16..];
    if body.len() < n_games * 68 {
        eprintln!("warning: {} truncated", job.path.display());
        return;
    }
    let file_hash = fnv1a(job.path.file_name().unwrap().to_str().unwrap());
    for g in 0..n_games {
        if *limit == 0 {
            return;
        }
        *limit -= 1;
        let rec = &body[g * 68..(g + 1) * 68];
        let real_score = rec[6] as i32;
        let mut moves = Vec::with_capacity(60);
        let mut ok = true;
        for &b in &rec[8..68] {
            if b == 0 {
                break;
            }
            let col = b / 10;
            let row = b % 10;
            if !(1..=8).contains(&col) || !(1..=8).contains(&row) {
                ok = false;
                break;
            }
            moves.push((row - 1) * 8 + (col - 1));
        }
        if !ok || moves.is_empty() {
            stats.replay_errors += 1;
            continue;
        }
        let game_hash = splitmix64(file_hash ^ (g as u64));
        emit_game(
            &moves,
            job.skip_plies,
            job.source,
            game_hash,
            val_mod,
            n_shards,
            writers,
            stats,
            Some(real_score),
        );
    }
}

fn process_transcript(
    job: &Job,
    val_mod: u64,
    n_shards: usize,
    writers: &Mutex<Writers>,
    stats: &mut Stats,
    limit: &mut u64,
) {
    let mut text = String::new();
    File::open(&job.path)
        .and_then(|mut f| f.read_to_string(&mut text))
        .unwrap_or_else(|e| panic!("{}: {}", job.path.display(), e));
    // File names repeat across folders, so include the parent folder
    // in the identity hash.
    let name = {
        let parent = job
            .path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("");
        format!("{}/{}", parent, job.path.file_name().unwrap().to_str().unwrap())
    };
    let file_hash = fnv1a(&name);
    for (g, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if *limit == 0 {
            return;
        }
        *limit -= 1;
        let moves = match notation::parse_move_seq(line) {
            Ok(m) => m,
            Err(_) => {
                stats.replay_errors += 1;
                continue;
            }
        };
        let game_hash = splitmix64(file_hash ^ (g as u64));
        emit_game(
            &moves,
            job.skip_plies,
            job.source,
            game_hash,
            val_mod,
            n_shards,
            writers,
            stats,
            None,
        );
    }
}

fn collect_jobs(args: &Args) -> Vec<Job> {
    let mut jobs = Vec::new();
    if let Some(dir) = &args.wthor {
        let mut files: Vec<PathBuf> = walk(dir)
            .into_iter()
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("wtb"))
            })
            .collect();
        files.sort();
        for path in files {
            jobs.push(Job {
                path,
                source: 0,
                skip_plies: 0,
                kind: JobKind::Wthor,
            });
        }
    }
    for dir in &args.egaroucid_v2 {
        // Folder name = number of random opening plies.
        let mut files: Vec<(u32, PathBuf)> = Vec::new();
        for path in walk(dir) {
            if path.extension().and_then(|e| e.to_str()) != Some("txt") {
                continue;
            }
            let n: Option<u32> = path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .and_then(|s| s.parse().ok());
            if let Some(n) = n {
                files.push((n, path));
            }
        }
        files.sort();
        for (n, path) in files {
            jobs.push(Job {
                path,
                source: 1,
                skip_plies: n,
                kind: JobKind::Transcript,
            });
        }
    }
    if let Some(dir) = &args.transcript {
        let mut files: Vec<PathBuf> = walk(dir)
            .into_iter()
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
            .collect();
        files.sort();
        for path in files {
            jobs.push(Job {
                path,
                source: 2,
                // N is 10..=19 per game and not recorded; skip the
                // conservative maximum.
                skip_plies: 19,
                kind: JobKind::Transcript,
            });
        }
    }
    jobs
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    out.push(p);
                }
            }
        }
    }
    out
}

struct Args {
    wthor: Option<PathBuf>,
    egaroucid_v2: Vec<PathBuf>,
    transcript: Option<PathBuf>,
    out: PathBuf,
    threads: usize,
    shards: usize,
    val_mod: u64,
    limit_games: u64,
    dump_boards: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        wthor: None,
        egaroucid_v2: Vec::new(),
        transcript: None,
        out: PathBuf::from("dataprep_out"),
        threads: 4,
        shards: 32,
        val_mod: 128,
        limit_games: u64::MAX,
        dump_boards: 0,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<&String, String> {
            argv.get(i + 1).ok_or_else(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--wthor" => args.wthor = Some(PathBuf::from(need(i)?)),
            "--egaroucid-v2" => args
                .egaroucid_v2
                .push(PathBuf::from(need(i)?)),
            "--transcript" => args.transcript = Some(PathBuf::from(need(i)?)),
            "--out" => args.out = PathBuf::from(need(i)?),
            "--threads" => args.threads = need(i)?.parse().map_err(|_| "bad --threads")?,
            "--shards" => args.shards = need(i)?.parse().map_err(|_| "bad --shards")?,
            "--val-mod" => args.val_mod = need(i)?.parse().map_err(|_| "bad --val-mod")?,
            "--limit-games" => {
                args.limit_games = need(i)?.parse().map_err(|_| "bad --limit-games")?
            }
            "--dump-boards" => {
                args.dump_boards = need(i)?.parse().map_err(|_| "bad --dump-boards")?
            }
            other => return Err(format!("unknown argument {}", other)),
        }
        i += 2;
    }
    args.threads = args.threads.clamp(1, 4);
    Ok(args)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {}", e);
            eprintln!(
                "usage: dataprep [--wthor DIR] [--egaroucid-v2 DIR]... [--transcript DIR] \
                 --out DIR [--threads N<=4] [--shards N] [--val-mod N] [--limit-games N]"
            );
            std::process::exit(2);
        }
    };
    // Check the inputs before writing anything: a missing directory or no
    // input files at all is an error, not an empty data set.
    let dirs = args
        .wthor
        .iter()
        .chain(&args.egaroucid_v2)
        .chain(&args.transcript);
    for dir in dirs {
        if !dir.is_dir() {
            eprintln!("error: input directory not found: {}", dir.display());
            std::process::exit(2);
        }
    }
    let jobs = collect_jobs(&args);
    if jobs.is_empty() {
        eprintln!(
            "error: no input files (WThor .wtb files under --wthor, game record .txt files \
             under --egaroucid-v2 or --transcript)"
        );
        std::process::exit(2);
    }
    eprintln!("{} input files", jobs.len());
    std::fs::create_dir_all(&args.out).unwrap();
    let writers = Mutex::new(Writers {
        shards: (0..args.shards)
            .map(|s| {
                BufWriter::new(
                    File::create(args.out.join(format!("train_{:02}.bin", s))).unwrap(),
                )
            })
            .collect(),
        val: BufWriter::new(File::create(args.out.join("val.bin")).unwrap()),
        boards: if args.dump_boards > 0 {
            Some(BufWriter::new(
                File::create(args.out.join("boards_val.txt")).unwrap(),
            ))
        } else {
            None
        },
        boards_left: args.dump_boards,
    });

    let job_queue = Mutex::new(jobs.into_iter());
    let global_limit = Mutex::new(args.limit_games);
    let all_stats: Mutex<Vec<Stats>> = Mutex::new(Vec::new());

    std::thread::scope(|scope| {
        for _ in 0..args.threads {
            scope.spawn(|| {
                let mut stats = Stats::default();
                loop {
                    let job = { job_queue.lock().unwrap().next() };
                    let Some(job) = job else { break };
                    // Reserve a chunk of the global game budget.
                    let mut budget = {
                        let mut l = global_limit.lock().unwrap();
                        let take = (*l).min(2_000_000);
                        *l -= take;
                        take
                    };
                    if budget == 0 {
                        break;
                    }
                    match job.kind {
                        JobKind::Wthor => process_wthor(
                            &job,
                            args.val_mod,
                            args.shards,
                            &writers,
                            &mut stats,
                            &mut budget,
                        ),
                        JobKind::Transcript => process_transcript(
                            &job,
                            args.val_mod,
                            args.shards,
                            &writers,
                            &mut stats,
                            &mut budget,
                        ),
                    }
                    // Return the unused budget.
                    let mut l = global_limit.lock().unwrap();
                    *l += budget;
                }
                all_stats.lock().unwrap().push(stats);
            });
        }
    });

    let mut w = writers.into_inner().unwrap();
    for s in &mut w.shards {
        s.flush().unwrap();
    }
    w.val.flush().unwrap();
    if let Some(b) = &mut w.boards {
        b.flush().unwrap();
    }

    let stats = all_stats.into_inner().unwrap();
    let mut total = Stats::default();
    for s in &stats {
        total.games += s.games;
        total.games_val += s.games_val;
        total.positions += s.positions;
        total.positions_val += s.positions_val;
        total.replay_errors += s.replay_errors;
        total.score_mismatches += s.score_mismatches;
        for i in 0..15 {
            total.by_phase[i] += s.by_phase[i];
        }
    }
    let mut meta = BTreeMap::new();
    meta.insert("record_bytes", RECORD_BYTES as u64);
    meta.insert("n_instances", N_INSTANCES as u64);
    meta.insert("games", total.games);
    meta.insert("games_val", total.games_val);
    meta.insert("positions", total.positions);
    meta.insert("positions_val", total.positions_val);
    meta.insert("replay_errors", total.replay_errors);
    meta.insert("score_mismatches", total.score_mismatches);
    let mut json = String::from("{\n");
    for (k, v) in &meta {
        json.push_str(&format!("  \"{}\": {},\n", k, v));
    }
    json.push_str("  \"by_phase\": [");
    json.push_str(
        &total
            .by_phase
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    );
    json.push_str("]\n}\n");
    std::fs::write(args.out.join("meta.json"), &json).unwrap();
    println!("{}", json);
}

// enginematch with an external engine that cannot be started: the
// command names the engine and its path, and exits with status 1.

use std::process::Command;

#[test]
fn enginematch_reports_an_engine_that_does_not_start() {
    let missing = "/nonexistent/versor-test/engine";
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let out = Command::new(env!("CARGO_BIN_EXE_versor-cli"))
        .current_dir(root)
        .args([
            "enginematch",
            "--engine",
            "edax",
            "--binary",
            missing,
            "--weights",
            "models/pattern_v2.bin",
            "--games",
            "2",
            "--depth",
            "1",
        ])
        .output()
        .expect("run versor-cli");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {}", stderr);
    assert!(stderr.contains("cannot start engine edax"), "{}", stderr);
    assert!(stderr.contains(missing), "{}", stderr);
}

/// Run versor-cli in the repository root and return its standard output.
fn run(args: &[&str]) -> String {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let out = Command::new(env!("CARGO_BIN_EXE_versor-cli"))
        .current_dir(root)
        .args(args)
        .output()
        .expect("run versor-cli");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The lines of a game log, sorted (games are written as they end).
fn sorted_lines(path: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    v.sort();
    v
}

/// The summary without the lines that report wall time.
fn without_times(out: &str) -> Vec<&str> {
    out.lines()
        .filter(|l| !l.starts_with("speed:") && !l.starts_with("clock:"))
        .collect()
}

#[test]
fn arena_gives_the_same_games_one_at_a_time_and_four_at_once() {
    let dir = std::env::temp_dir().join(format!("versor-test-arena-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut runs = Vec::new();
    for threads in ["1", "4"] {
        let log = dir.join(format!("games-{}.txt", threads));
        let out = run(&[
            "arena", "--weights", "models/pattern_v2.bin", "--weights-b", "models/pattern_v1.bin",
            "--games", "8", "--depth", "3", "--solve-at", "12", "--tt", "1", "--seed", "5",
            "--threads", threads, "--log", log.to_str().unwrap(),
        ]);
        runs.push((out, sorted_lines(&log)));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(runs[0].1.len(), 8);
    assert_eq!(runs[0].1, runs[1].1);
    assert_eq!(without_times(&runs[0].0), without_times(&runs[1].0));
}

/// With a real engine: set VERSOR_TEST_ENGINES to an engines.json whose
/// paths point at installed engines (and VERSOR_TEST_ENGINE to its name,
/// default edax). Skipped otherwise.
#[test]
fn enginematch_gives_the_same_games_one_at_a_time_and_four_at_once() {
    let Ok(cfg) = std::env::var("VERSOR_TEST_ENGINES") else {
        eprintln!("VERSOR_TEST_ENGINES not set; skipping");
        return;
    };
    let engine = std::env::var("VERSOR_TEST_ENGINE").unwrap_or_else(|_| "edax".to_string());
    let dir = std::env::temp_dir().join(format!("versor-test-match-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut runs = Vec::new();
    for threads in ["1", "4"] {
        let log = dir.join(format!("games-{}.txt", threads));
        let out = run(&[
            "enginematch", "--engines", &cfg, "--engine", &engine, "--level", "4",
            "--weights", "models/pattern_v2.bin", "--games", "16", "--depth", "4",
            "--tt", "4", "--seed", "9", "--threads", threads, "--log", log.to_str().unwrap(),
        ]);
        runs.push((out, sorted_lines(&log)));
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(runs[0].1.len(), 16);
    assert_eq!(runs[0].1, runs[1].1);
    assert_eq!(without_times(&runs[0].0), without_times(&runs[1].0));
}

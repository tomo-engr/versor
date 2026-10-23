// Python/Rust inference parity.
//
// python/train.py --testvec writes lines of "board_text<TAB>value"
// where value is the quantized integer evaluation computed in Python
// (emulating the Rust fixed point rounding exactly). This test loads
// the weight file, evaluates every board through the full Rust path
// (board -> pattern indices -> weights) and requires exact equality.
//
// Run explicitly:
//   VERSOR_WEIGHTS=path/to/weights.bin VERSOR_TESTVEC=path/to/testvec.txt \
//   cargo test --release -p versor-eval --test parity -- --ignored --nocapture

use versor_core::Position;
use versor_eval::PatternEval;
use versor_search::Eval;

#[test]
#[ignore = "needs VERSOR_WEIGHTS and VERSOR_TESTVEC (see file header)"]
fn python_rust_inference_parity() {
    let weights = std::env::var("VERSOR_WEIGHTS").expect("set VERSOR_WEIGHTS");
    let testvec = std::env::var("VERSOR_TESTVEC").expect("set VERSOR_TESTVEC");
    let eval = PatternEval::from_file(&weights).expect("weight file");
    let text = std::fs::read_to_string(&testvec).expect("test vector file");
    let mut n = 0u32;
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (board, expect) = line.split_once('\t').expect("board<TAB>value");
        let pos = Position::from_text(board).expect("board text");
        let expect: i32 = expect.trim().parse().expect("integer value");
        let got = eval.eval(&pos);
        assert_eq!(
            got,
            expect,
            "parity mismatch at line {}: rust {} python {}\n{}",
            lineno + 1,
            got,
            expect,
            pos.to_text()
        );
        n += 1;
    }
    assert!(n >= 100, "test vector should hold at least 100 positions");
    println!("parity ok: {} positions match exactly", n);
}

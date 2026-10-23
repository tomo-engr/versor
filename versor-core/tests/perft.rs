// Perft against the reference leaf counts for depths 0..9 from the
// initial position, under the pass convention of src/perft.rs: a forced
// pass does not consume depth. The depth 9 value is 3005320 under this
// convention (the widely quoted 3005288 corresponds to the
// pass-consumes-one-ply convention).

use versor_core::{perft, Position};

const REFERENCE: [u64; 10] = [
    1, 4, 12, 56, 244, 1396, 8200, 55092, 390216, 3005320,
];

#[test]
fn perft_shallow_matches_reference() {
    let p = Position::initial();
    for depth in 0..=7u32 {
        assert_eq!(
            perft(&p, depth),
            REFERENCE[depth as usize],
            "perft depth {}",
            depth
        );
    }
}

#[test]
#[ignore = "slow in debug builds; run with --release -- --ignored"]
fn perft_deep_matches_reference() {
    let p = Position::initial();
    for depth in 8..=9u32 {
        assert_eq!(
            perft(&p, depth),
            REFERENCE[depth as usize],
            "perft depth {}",
            depth
        );
    }
}

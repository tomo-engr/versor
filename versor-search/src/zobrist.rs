// Zobrist hashing. Keys are generated at compile time by xorshift64*
// from a fixed seed, so hashes are reproducible across builds and runs.
//
// Key layout: KEYS[0..64] are black disc keys (indexed by square),
// KEYS[64..128] are white disc keys, KEYS[128] is xored into the hash
// when white is the side to move.

use versor_core::{Color, Position};

const SEED: u64 = 0x7665_7273_6f72_0001;

const fn xorshift_step(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

const fn generate_keys() -> [u64; 129] {
    let mut keys = [0u64; 129];
    let mut state = SEED;
    let mut i = 0;
    while i < 129 {
        state = xorshift_step(state);
        keys[i] = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        i += 1;
    }
    keys
}

pub const KEYS: [u64; 129] = generate_keys();
pub const STM_WHITE_KEY: u64 = KEYS[128];

/// Full hash of a position, computed from scratch.
pub fn hash(pos: &Position) -> u64 {
    let mut h = match pos.stm {
        Color::Black => 0,
        Color::White => STM_WHITE_KEY,
    };
    let mut black = pos.black();
    while black != 0 {
        let sq = black.trailing_zeros() as usize;
        black &= black - 1;
        h ^= KEYS[sq];
    }
    let mut white = pos.white();
    while white != 0 {
        let sq = white.trailing_zeros() as usize;
        white &= white - 1;
        h ^= KEYS[64 + sq];
    }
    h
}

/// Incremental hash update: the side `mover` places a disc on `sq` and
/// flips the discs in `flipped`. Returns the hash of the resulting
/// position (side to move switched).
pub fn child_hash(h: u64, mover: Color, sq: u8, mut flipped: u64) -> u64 {
    let base = match mover {
        Color::Black => 0usize,
        Color::White => 64usize,
    };
    let mut h = h ^ KEYS[base + sq as usize] ^ STM_WHITE_KEY;
    while flipped != 0 {
        let f = flipped.trailing_zeros() as usize;
        flipped &= flipped - 1;
        h ^= KEYS[f] ^ KEYS[64 + f];
    }
    h
}

/// Hash update for a pass (only the side to move changes).
pub fn pass_hash(h: u64) -> u64 {
    h ^ STM_WHITE_KEY
}

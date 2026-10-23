// Small deterministic PRNG (xorshift64*) for random playouts and
// benchmarks. Not suitable for cryptographic use. Kept dependency-free
// on purpose; modulo bias in `below` is irrelevant for playouts.

#[derive(Clone, Debug)]
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    pub fn new(seed: u64) -> Xorshift64Star {
        Xorshift64Star {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform-ish integer in 0..n (n > 0).
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        (self.next_u64() % n as u64) as u32
    }
}

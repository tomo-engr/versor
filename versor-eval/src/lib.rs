// versor-eval: pattern-based linear evaluation.
//
// Score = sum over the pattern instances of a per-phase weight table
// entry (plus, from v2 on, three scalar features), plus a per-phase
// bias, in 1/256 disc units, rounded to whole discs and clamped to
// [-63, 63]. Cell states are relative to the side to move, so one
// weight set serves both colors. Weights are trained in Python
// (python/train.py) and loaded from a versioned binary file (see
// weights.rs); the file version decides which pattern tables, scalar
// features and phase buckets are used.

pub mod pattern;
pub mod weights;

use versor_core::{bitboard, Color, Position};
use versor_search::{Eval, Evaluator};

use pattern::{
    extract_indices, extract_raw_black, patterns, N_INSTANCES, N_PHASES, N_TABLES_V2, N_TABLES_V3,
    TOTAL_WEIGHTS, TOTAL_WEIGHTS_V2, TOTAL_WEIGHTS_V3,
};
use weights::{
    PHASE_STRIDE, PHASE_STRIDE_V2, PHASE_STRIDE_V3, PHASE_STRIDE_V5, VERSION_V2, VERSION_V3,
    VERSION_V4, VERSION_V5,
};

/// Incremental evaluation state: raw pattern indices in the fixed
/// black basis (black = 1, white = 2), maintained differentially as
/// moves are made. Independent of the side to move, so a pass leaves
/// it unchanged.
#[derive(Clone, Copy)]
pub struct IdxState {
    raw: [u16; N_INSTANCES],
}

pub struct PatternEval {
    /// [phase][pattern tables, (v2 and later: scalars,) bias], flattened.
    weights: Vec<i16>,
    /// v2 and later files add scalar features (own/opp mobility,
    /// parity) and phase-boundary smoothing; v1 files have neither.
    has_scalars: bool,
    /// Number of pattern tables the loaded file covers (10 for v1/v2,
    /// 11 for v3/v4, 15 for v5). Tables beyond that are ignored.
    active_tables: usize,
    /// Sum of the active pattern-table sizes (scalar/bias offset).
    active_total: usize,
    stride: usize,
    /// Phase bucketing of the loaded file: number of phases and the
    /// disc-count width of each bucket (v4: 30 phases, 2 discs; all
    /// other versions: 15 phases, 4 discs).
    n_phases: usize,
    bucket_discs: usize,
}

impl PatternEval {
    pub fn from_file(path: &str) -> Result<PatternEval, String> {
        let (weights, version) = weights::load(path)?;
        let (has_scalars, active_tables, active_total, stride, n_phases, bucket_discs) =
            match version {
                VERSION_V5 => (
                    true,
                    pattern::N_TABLES,
                    TOTAL_WEIGHTS,
                    PHASE_STRIDE_V5,
                    N_PHASES,
                    4,
                ),
                VERSION_V4 => (
                    true,
                    N_TABLES_V3,
                    TOTAL_WEIGHTS_V3,
                    PHASE_STRIDE_V3,
                    pattern::N_PHASES_V4,
                    2,
                ),
                VERSION_V3 => (true, N_TABLES_V3, TOTAL_WEIGHTS_V3, PHASE_STRIDE_V3, N_PHASES, 4),
                VERSION_V2 => (true, N_TABLES_V2, TOTAL_WEIGHTS_V2, PHASE_STRIDE_V2, N_PHASES, 4),
                _ => (false, N_TABLES_V2, TOTAL_WEIGHTS_V2, PHASE_STRIDE, N_PHASES, 4),
            };
        Ok(PatternEval {
            weights,
            has_scalars,
            active_tables,
            active_total,
            stride,
            n_phases,
            bucket_discs,
        })
    }

    /// Build an evaluator from a weight vector in the v1 layout (used by
    /// the tests, which evaluate with random weights).
    pub fn from_weights(weights: Vec<i16>) -> Result<PatternEval, String> {
        if weights.len() != N_PHASES * PHASE_STRIDE {
            return Err("wrong weight vector length".to_string());
        }
        Ok(PatternEval {
            weights,
            has_scalars: false,
            active_tables: N_TABLES_V2,
            active_total: TOTAL_WEIGHTS_V2,
            stride: PHASE_STRIDE,
            n_phases: N_PHASES,
            bucket_discs: 4,
        })
    }

    /// Accumulator for one phase from canonical side-to-move indices
    /// plus (v2 and later) the scalar features. Only the pattern tables
    /// present in the loaded file contribute.
    fn acc_at(&self, base: usize, idx: &[u16; N_INSTANCES], om: i32, pm: i32, par: i32) -> i32 {
        let pats = patterns();
        let mut acc = 0i32;
        for (i, inst) in pats.instances.iter().enumerate() {
            if (inst.table as usize) >= self.active_tables {
                continue;
            }
            let off = base + pats.offsets[inst.table as usize] + idx[i] as usize;
            acc += i32::from(self.weights[off]);
        }
        if self.has_scalars {
            let sb = base + self.active_total;
            acc += i32::from(self.weights[sb]) * om
                + i32::from(self.weights[sb + 1]) * pm
                + i32::from(self.weights[sb + 2]) * par;
        }
        acc + i32::from(self.weights[base + self.stride - 1])
    }

    /// Phase-smoothed accumulator (v2 and later): blend with the
    /// neighbor phase by eighths depending on the disc position inside
    /// the bucket. For 4-disc buckets the ramp is 3,1|1,3; for 2-disc
    /// buckets (v4) it is 2|2 (quarter blend toward the nearer
    /// neighbor). Mirrored exactly in python/train.py.
    fn blended_acc(&self, pos: &Position, idx: &[u16; N_INSTANCES], om: i32, pm: i32, par: i32) -> i32 {
        let discs = (pos.own | pos.opp).count_ones() as usize;
        let p = pattern::phase_of_discs(discs, self.bucket_discs, self.n_phases);
        let acc_p = self.acc_at(p * self.stride, idx, om, pm, par);
        if !self.has_scalars {
            return acc_p;
        }
        let u = discs.saturating_sub(4) % self.bucket_discs;
        let (q, w8): (isize, i32) = if self.bucket_discs == 4 {
            match u {
                0 => (p as isize - 1, 3),
                1 => (p as isize - 1, 1),
                2 => (p as isize + 1, 1),
                _ => (p as isize + 1, 3),
            }
        } else {
            // 2-disc buckets: u = 0 leans to the previous phase, u = 1
            // to the next, each by 2/8.
            match u {
                0 => (p as isize - 1, 2),
                _ => (p as isize + 1, 2),
            }
        };
        if !(0..self.n_phases as isize).contains(&q) {
            return acc_p;
        }
        let acc_q = self.acc_at(q as usize * self.stride, idx, om, pm, par);
        ((8 - w8) * acc_p + w8 * acc_q + 4) >> 3
    }

    fn scalars(&self, pos: &Position) -> (i32, i32, i32) {
        if !self.has_scalars {
            return (0, 0, 0);
        }
        (
            pos.legal_moves().count_ones() as i32,
            bitboard::legal_moves(pos.opp, pos.own).count_ones() as i32,
            (pos.empty().count_ones() & 1) as i32,
        )
    }

    /// Raw (phase-smoothed) accumulator in 1/256 disc units.
    pub fn eval_raw(&self, pos: &Position) -> i32 {
        let mut idx = [0u16; N_INSTANCES];
        extract_indices(pos, &mut idx);
        let (om, pm, par) = self.scalars(pos);
        self.blended_acc(pos, &idx, om, pm, par)
    }

    fn round_clamp(acc: i32) -> i32 {
        // Round to nearest disc (half away from zero) and clamp inside
        // the exact-score range. Mirrored exactly in python/train.py.
        let rounded = if acc >= 0 {
            (acc + 128) >> 8
        } else {
            -((-acc + 128) >> 8)
        };
        rounded.clamp(-63, 63)
    }
}

impl Eval for PatternEval {
    fn eval(&self, pos: &Position) -> i32 {
        PatternEval::round_clamp(self.eval_raw(pos))
    }
}

impl Evaluator for PatternEval {
    type State = IdxState;

    fn init_state(&self, pos: &Position) -> IdxState {
        let mut st = IdxState {
            raw: [0u16; N_INSTANCES],
        };
        extract_raw_black(pos, &mut st.raw);
        st
    }

    fn child_state(&self, pos: &Position, st: &IdxState, sq: u8, flips: u64) -> IdxState {
        let pats = patterns();
        let mut s = *st;
        // The mover places a disc on an empty square (0 -> 1 or 2)...
        let place: i32 = match pos.stm {
            Color::Black => 1,
            Color::White => 2,
        };
        for &(inst, pow) in &pats.by_square[sq as usize] {
            let r = &mut s.raw[inst as usize];
            *r = (i32::from(*r) + place * i32::from(pow)) as u16;
        }
        // ...and flips opponent discs to its own color (2 -> 1 when
        // black moves, 1 -> 2 when white moves).
        let delta: i32 = match pos.stm {
            Color::Black => -1,
            Color::White => 1,
        };
        let mut f = flips;
        while f != 0 {
            let fsq = f.trailing_zeros() as usize;
            f &= f - 1;
            for &(inst, pow) in &pats.by_square[fsq] {
                let r = &mut s.raw[inst as usize];
                *r = (i32::from(*r) + delta * i32::from(pow)) as u16;
            }
        }
        s
    }

    fn eval_with(&self, pos: &Position, st: &IdxState) -> i32 {
        let pats = patterns();
        let white_view = pos.stm == Color::White;
        let mut idx = [0u16; N_INSTANCES];
        for (i, inst) in pats.instances.iter().enumerate() {
            let t = inst.table as usize;
            let mut raw = st.raw[i] as usize;
            if white_view {
                raw = pats.opp_view[t][raw] as usize;
            }
            idx[i] = pats.canonical[t][raw];
        }
        let (om, pm, par) = self.scalars(pos);
        PatternEval::round_clamp(self.blended_acc(pos, &idx, om, pm, par))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use versor_core::rng::Xorshift64Star;

    #[test]
    fn save_load_round_trip_and_eval() {
        let mut rng = Xorshift64Star::new(7);
        let n = pattern::N_PHASES * PHASE_STRIDE;
        let weights: Vec<i16> = (0..n).map(|_| (rng.next_u64() & 0x3ff) as i16 - 512).collect();
        let dir = std::env::temp_dir().join("versor_eval_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.bin");
        let path = path.to_str().unwrap();
        weights::save(path, &weights).unwrap();
        let (loaded, version) = weights::load(path).unwrap();
        assert_eq!(version, weights::VERSION);
        assert_eq!(weights, loaded);

        let eval = PatternEval::from_weights(loaded).unwrap();
        let pos = versor_core::Position::initial();
        let v = eval.eval(&pos);
        assert!((-63..=63).contains(&v));
        // Rounding check against the raw accumulator.
        let raw = eval.eval_raw(&pos);
        let expect = ((raw as f64) / 256.0).round() as i32;
        assert_eq!(v, expect.clamp(-63, 63));
    }
}

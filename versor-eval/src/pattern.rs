// Pattern definitions and index extraction.
//
// The pattern shapes were designed for this project (the line,
// diagonal and corner shapes are the natural ones described in M.
// Buro's evaluation function papers, not copied from any
// implementation). Weight files from v1/v2 use tables 0..9, v3/v4 add
// table 10 and v5 adds tables 11..14:
//
//   table  shape          cells  instances
//   0      line1 (edge)   8      4
//   1      line2          8      4
//   2      line3          8      4
//   3      line4          8      4
//   4      diag8          8      2
//   5      diag7          7      4
//   6      diag6          6      4
//   7      diag5          5      4
//   8      corner3x3      9      4
//   9      edge + 2X      10     4
//   10     corner2x5      10     8
//   11     center2x4      8      2
//   12     inner3x3       9      4
//   13     diag4          4      4
//   14     diag3          3      4
//
// Every shape is replicated over the board's 8 symmetries; distinct
// placements share one weight table per shape (60 instances in all).
// Indices are canonicalized over each placement's self-symmetries, so
// that mirrored configurations share weights exactly (see `build`).
//
// Cell states are relative to the side to move: 0 = empty, 1 = own
// disc, 2 = opponent disc. The index of an instance is the base 3
// number sum(state(cells[j]) * 3^j).

use std::sync::OnceLock;

use versor_core::Position;

// Superset of pattern shapes. Weight files declare how many pattern
// tables they cover (10 for v1/v2, 11 for v3/v4, 15 for v5), so all
// coexist in one binary: the evaluator uses only the tables present in
// the loaded file. See N_TABLES_V2 / N_TABLES_V3.
pub const N_TABLES: usize = 15;
pub const N_INSTANCES: usize = 60;
pub const MAX_CELLS: usize = 10;

/// Pattern-table count of the v1/v2 weight files.
pub const N_TABLES_V2: usize = 10;
/// Sum of the v1/v2 table sizes.
pub const TOTAL_WEIGHTS_V2: usize = 114696;
/// Pattern-table count of the v3/v4 weight files.
pub const N_TABLES_V3: usize = 11;
/// Sum of the v3/v4 table sizes (tables 0..10).
pub const TOTAL_WEIGHTS_V3: usize = 173745;

/// Canonical instance of each shape (square indices, sq = row * 8 + col).
const CANONICAL: [&[u8]; N_TABLES] = [
    &[0, 1, 2, 3, 4, 5, 6, 7],           // line1: a1..h1
    &[8, 9, 10, 11, 12, 13, 14, 15],     // line2: a2..h2
    &[16, 17, 18, 19, 20, 21, 22, 23],   // line3: a3..h3
    &[24, 25, 26, 27, 28, 29, 30, 31],   // line4: a4..h4
    &[0, 9, 18, 27, 36, 45, 54, 63],     // diag8: a1..h8
    &[1, 10, 19, 28, 37, 46, 55],        // diag7: b1..h7
    &[2, 11, 20, 29, 38, 47],            // diag6: c1..h6
    &[3, 12, 21, 30, 39],                // diag5: d1..h5
    &[0, 1, 2, 8, 9, 10, 16, 17, 18],    // corner3x3 at a1
    &[0, 1, 2, 3, 4, 5, 6, 7, 9, 14],    // edge2x: a1..h1 + b2 + g2
    // corner2x5: ranks 1-2, files a-e (top-left 2x5 block).
    // Non-square, so its 8 board symmetries are all distinct (8
    // placements, no self-symmetry: identity canonicalization).
    &[0, 1, 2, 3, 4, 8, 9, 10, 11, 12],
    // --- v5 additions (self-designed, cover under-represented regions:
    // board centre and the corner-to-centre interior) ---
    // center2x4: ranks 4-5, files c-f (central 2x4 band). 2 placements
    // (this and its transpose, the vertical central band).
    &[26, 27, 28, 29, 34, 35, 36, 37],
    // inner3x3: ranks 2-4, files b-d (the 3x3 just inside a1's corner).
    // 4 placements (one per corner). Covers the corner-to-centre gap.
    &[9, 10, 11, 17, 18, 19, 25, 26, 27],
    // diag4: e1,f2,g3,h4 (4-cell diagonal). Completes the short-diagonal
    // family (diag5..8 already present). 4 placements.
    &[4, 13, 22, 31],
    // diag3: f1,g2,h3 (3-cell diagonal). 4 placements.
    &[5, 14, 23],
];

/// Number of weights in each shape's table (3^cells).
pub const TABLE_SIZES: [usize; N_TABLES] = [
    6561, 6561, 6561, 6561, 6561, 2187, 729, 243, 19683, 59049, 59049,
    6561, 19683, 81, 27,
];

/// Sum of all table sizes (bias not included).
pub const TOTAL_WEIGHTS: usize = 200097;

#[derive(Clone, Copy)]
pub struct Instance {
    pub table: u8,
    pub n_cells: u8,
    pub cells: [u8; MAX_CELLS],
}

pub struct Patterns {
    pub instances: Vec<Instance>,
    /// Per table: canonicalization LUT idx -> min(idx, perm(idx)).
    pub canonical: Vec<Vec<u16>>,
    /// Per table: perspective swap LUT (digits 1 and 2 exchanged),
    /// used to view black-basis indices from white's side.
    pub opp_view: Vec<Vec<u16>>,
    /// Flattened weight offset of each table.
    pub offsets: [usize; N_TABLES],
    /// For each square: (instance id, 3^cell_position) pairs of every
    /// pattern instance containing that square (differential updates).
    pub by_square: Vec<Vec<(u8, u16)>>,
}

fn transform(sq: u8, t: usize) -> u8 {
    let (r, c) = ((sq / 8) as i32, (sq % 8) as i32);
    let (r, c) = match t {
        0 => (r, c),
        1 => (c, 7 - r),         // rotate 90
        2 => (7 - r, 7 - c),     // rotate 180
        3 => (7 - c, r),         // rotate 270
        4 => (r, 7 - c),         // mirror left-right
        5 => (c, r),             // transpose
        6 => (7 - r, c),         // mirror top-bottom
        _ => (7 - c, 7 - r),     // anti-transpose
    };
    (r * 8 + c) as u8
}

fn build() -> Patterns {
    let mut instances = Vec::new();
    let mut canonical = Vec::new();
    for (table, cells) in CANONICAL.iter().enumerate() {
        let n = cells.len();
        // Generate all 8 transforms; deduplicate placements by cell
        // set, keeping the first ordering as canonical. Every transform
        // that maps the canonical placement back onto its own cell set
        // (its stabilizer) yields an index permutation; folding indices
        // over the WHOLE stabilizer (not just one symmetry) makes the
        // evaluation exactly invariant under all board symmetries,
        // including placements that map to themselves under a mirror.
        let mut placements: Vec<Vec<u8>> = Vec::new();
        let mut stab_perms: Vec<Vec<usize>> = Vec::new();
        for t in 0..8 {
            let mapped: Vec<u8> = cells.iter().map(|&sq| transform(sq, t)).collect();
            let mut sorted = mapped.clone();
            sorted.sort_unstable();
            let existing = placements.iter().position(|p| {
                let mut s = p.clone();
                s.sort_unstable();
                s == sorted
            });
            match existing {
                None => placements.push(mapped),
                Some(0) => {
                    // Permutation p with mapped[j] = canonical[p(j)].
                    let canon = &placements[0];
                    let p: Vec<usize> = mapped
                        .iter()
                        .map(|&sq| canon.iter().position(|&c| c == sq).unwrap())
                        .collect();
                    stab_perms.push(p);
                }
                Some(_) => {}
            }
        }
        for placement in &placements {
            let mut cells_arr = [0u8; MAX_CELLS];
            cells_arr[..n].copy_from_slice(placement);
            instances.push(Instance {
                table: table as u8,
                n_cells: n as u8,
                cells: cells_arr,
            });
        }
        // Canonicalization LUT from the placement self-symmetry. The
        // permutation is identical for every placement of the shape
        // because all placements are transforms of the canonical one.
        let size = TABLE_SIZES[table];
        let mut lut = vec![0u16; size];
        // Canonical index = minimum index over the stabilizer orbit.
        // stab_perms always contains the identity (t = 0); shapes with
        // no nontrivial self-symmetry (e.g. the non-square 2x5) fold to
        // themselves only.
        let mut digits = vec![0u8; n];
        for (idx, slot) in lut.iter_mut().enumerate() {
            let mut x = idx;
            for d in digits.iter_mut() {
                *d = (x % 3) as u8;
                x /= 3;
            }
            let mut best = idx;
            for p in &stab_perms {
                let mut permuted = 0usize;
                let mut pow = 1usize;
                for j in 0..n {
                    permuted += digits[p[j]] as usize * pow;
                    pow *= 3;
                }
                best = best.min(permuted);
            }
            *slot = best as u16;
        }
        canonical.push(lut);
    }
    debug_assert_eq!(instances.len(), N_INSTANCES);
    let mut offsets = [0usize; N_TABLES];
    let mut acc = 0;
    for (i, s) in TABLE_SIZES.iter().enumerate() {
        offsets[i] = acc;
        acc += s;
    }
    debug_assert_eq!(acc, TOTAL_WEIGHTS);
    // Perspective swap LUT: exchange digits 1 and 2 of the base 3 index.
    let mut opp_view = Vec::with_capacity(N_TABLES);
    for (table, cells) in CANONICAL.iter().enumerate() {
        let n = cells.len();
        let size = TABLE_SIZES[table];
        let mut lut = vec![0u16; size];
        for (idx, slot) in lut.iter_mut().enumerate() {
            let mut x = idx;
            let mut out = 0usize;
            let mut pow = 1usize;
            for _ in 0..n {
                let d = x % 3;
                x /= 3;
                out += (if d == 0 { 0 } else { 3 - d }) * pow;
                pow *= 3;
            }
            *slot = out as u16;
        }
        opp_view.push(lut);
    }
    // Per-square delta lists for differential index updates.
    let mut by_square = vec![Vec::new(); 64];
    for (i, inst) in instances.iter().enumerate() {
        let mut pow = 1u32;
        for j in 0..inst.n_cells as usize {
            by_square[inst.cells[j] as usize].push((i as u8, pow as u16));
            pow *= 3;
        }
    }
    Patterns {
        instances,
        canonical,
        opp_view,
        offsets,
        by_square,
    }
}

static PATTERNS: OnceLock<Patterns> = OnceLock::new();

pub fn patterns() -> &'static Patterns {
    PATTERNS.get_or_init(build)
}

/// Extract the canonical table-local index of every instance for the
/// side to move.
pub fn extract_indices(pos: &Position, out: &mut [u16; N_INSTANCES]) {
    let pats = patterns();
    let own = pos.own;
    let opp = pos.opp;
    for (i, inst) in pats.instances.iter().enumerate() {
        let mut idx = 0usize;
        let mut pow = 1usize;
        for j in 0..inst.n_cells as usize {
            let sq = inst.cells[j];
            let state = ((own >> sq) & 1) + 2 * ((opp >> sq) & 1);
            idx += state as usize * pow;
            pow *= 3;
        }
        out[i] = pats.canonical[inst.table as usize][idx];
    }
}

/// Raw (uncanonicalized) indices in the fixed black basis: black
/// disc = 1, white disc = 2, empty = 0. These are what the
/// differential updates maintain; `PatternEval::eval_with` converts
/// them to the canonical side-to-move index used by the weight tables.
pub fn extract_raw_black(pos: &Position, out: &mut [u16; N_INSTANCES]) {
    let pats = patterns();
    let black = pos.black();
    let white = pos.white();
    for (i, inst) in pats.instances.iter().enumerate() {
        let mut idx = 0usize;
        let mut pow = 1usize;
        for j in 0..inst.n_cells as usize {
            let sq = inst.cells[j];
            let state = ((black >> sq) & 1) + 2 * ((white >> sq) & 1);
            idx += state as usize * pow;
            pow *= 3;
        }
        out[i] = idx as u16;
    }
}

/// Phase bucket by disc count. v4 uses 2-disc buckets (30 phases);
/// every other version uses 4-disc buckets (15 phases). The evaluator
/// picks the bucket width from the loaded file version; `phase_of` is
/// the 15-phase bucketing (also the phase byte that dataprep writes).
pub const N_PHASES: usize = 15;
pub const N_PHASES_V4: usize = 30;

pub fn phase_of(pos: &Position) -> usize {
    let discs = (pos.own | pos.opp).count_ones() as usize;
    ((discs.saturating_sub(4)) / 4).min(N_PHASES - 1)
}

/// Phase bucket for a given disc count with `bucket_discs`-wide buckets
/// and `n_phases` total, clamped. Mirrored in python/train.py.
pub fn phase_of_discs(discs: usize, bucket_discs: usize, n_phases: usize) -> usize {
    (discs.saturating_sub(4) / bucket_discs).min(n_phases - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use versor_core::notation::{parse_move_seq, replay};

    #[test]
    fn instance_counts_and_sizes() {
        let p = patterns();
        assert_eq!(p.instances.len(), N_INSTANCES);
        let counts: Vec<usize> = (0..N_TABLES)
            .map(|t| p.instances.iter().filter(|i| i.table as usize == t).count())
            .collect();
        // Lines x4, diag8 x2, diag7/6/5 + corner3x3 + edge2x x4 each,
        // corner 2x5 with 8 distinct (asymmetric) placements, and the
        // v5 additions: center2x4 x2, inner3x3 x4, diag4 x4, diag3 x4.
        assert_eq!(counts, vec![4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 8, 2, 4, 4, 4]);
        for (t, lut) in p.canonical.iter().enumerate() {
            assert_eq!(lut.len(), TABLE_SIZES[t]);
        }
    }

    #[test]
    fn indices_invariant_under_symmetry() {
        // A position and its mirror must produce the same multiset of
        // canonical indices per table.
        let moves = parse_move_seq("f5d6c3d3c4f4f6g5e6d7").unwrap();
        let pos = replay(versor_core::Position::initial(), &moves).unwrap();
        let mirror = |bb: u64| -> u64 {
            // mirror left-right: reverse bits within each byte
            let mut out = 0u64;
            for r in 0..8 {
                let row = (bb >> (r * 8)) & 0xff;
                out |= u64::from((row as u8).reverse_bits()) << (r * 8);
            }
            out
        };
        let mpos = versor_core::Position {
            own: mirror(pos.own),
            opp: mirror(pos.opp),
            stm: pos.stm,
        };
        let mut a = [0u16; N_INSTANCES];
        let mut b = [0u16; N_INSTANCES];
        extract_indices(&pos, &mut a);
        extract_indices(&mpos, &mut b);
        let pats = patterns();
        for t in 0..N_TABLES {
            let mut xa: Vec<u16> = (0..N_INSTANCES)
                .filter(|&i| pats.instances[i].table as usize == t)
                .map(|i| a[i])
                .collect();
            let mut xb: Vec<u16> = (0..N_INSTANCES)
                .filter(|&i| pats.instances[i].table as usize == t)
                .map(|i| b[i])
                .collect();
            xa.sort_unstable();
            xb.sort_unstable();
            assert_eq!(xa, xb, "table {} indices differ under mirror", t);
        }
    }

    #[test]
    fn phase_buckets() {
        let p = versor_core::Position::initial();
        assert_eq!(phase_of(&p), 0);
    }
}

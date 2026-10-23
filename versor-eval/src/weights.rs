// Weight file I/O. Binary format (little endian):
//
//   offset  size      field
//   0       4         magic "VEVL"
//   4       4         version (u32): 1 to 5
//   8       4         scale (u32) = 256   (weight unit = 1/scale disc)
//   12      4         n_phases (u32) = 15 (30 for v4)
//   16      4         n_header_tables (u32)
//   20      4*n       cells per header table (u32); a scalar table is
//                     marked with the sentinel 200, the bias with 0
//   ...     2*N       weights (i16), N = n_phases * phase_stride
//
// Weight order per phase: pattern tables (by index), then (v2 and
// later) the three scalar feature weights, then the bias. Versions:
//   v1: 10 pattern tables + bias
//   v2: 10 pattern tables + 3 scalars + bias
//   v3: 11 pattern tables (corner 2x5 added) + 3 scalars + bias
//   v4: the v3 tables with 30 phases (2-disc buckets)
//   v5: 15 pattern tables (four more added) + 3 scalars + bias
// New tables are appended, so older files keep the same offsets for
// their tables and all versions load in the same binary. The Python
// trainer writes the v5 layout (python/train.py).

use std::io::{Read, Write};

use crate::pattern::{N_PHASES, N_PHASES_V4, TOTAL_WEIGHTS, TOTAL_WEIGHTS_V2, TOTAL_WEIGHTS_V3};

pub const MAGIC: &[u8; 4] = b"VEVL";
pub const VERSION: u32 = 1;
pub const VERSION_V2: u32 = 2;
pub const VERSION_V3: u32 = 3;
pub const VERSION_V4: u32 = 4;
pub const VERSION_V5: u32 = 5;
pub const SCALE: u32 = 256;

/// Scalar feature weights per phase (v2 and later): own/opp mobility,
/// parity.
pub const N_SCALARS: usize = 3;
/// Sentinel cell count marking the scalar table in a v2-v5 header.
pub const SCALAR_SENTINEL: u32 = 200;

/// v1: patterns (10 tables) + bias.
pub const PHASE_STRIDE: usize = TOTAL_WEIGHTS_V2 + 1;
/// v2: patterns (10 tables) + scalars + bias.
pub const PHASE_STRIDE_V2: usize = TOTAL_WEIGHTS_V2 + N_SCALARS + 1;
/// v3/v4: patterns (11 tables) + scalars + bias.
pub const PHASE_STRIDE_V3: usize = TOTAL_WEIGHTS_V3 + N_SCALARS + 1;
/// v5: patterns (15 tables) + scalars + bias.
pub const PHASE_STRIDE_V5: usize = TOTAL_WEIGHTS + N_SCALARS + 1;

const CELLS_V1: [u32; 11] = [8, 8, 8, 8, 8, 7, 6, 5, 9, 10, 0];
const CELLS_V2: [u32; 12] = [8, 8, 8, 8, 8, 7, 6, 5, 9, 10, SCALAR_SENTINEL, 0];
const CELLS_V3: [u32; 13] = [8, 8, 8, 8, 8, 7, 6, 5, 9, 10, 10, SCALAR_SENTINEL, 0];
// v5 appends four tables (center2x4=8, inner3x3=9, diag4=4, diag3=3).
const CELLS_V5: [u32; 17] =
    [8, 8, 8, 8, 8, 7, 6, 5, 9, 10, 10, 8, 9, 4, 3, SCALAR_SENTINEL, 0];

// Per version: (header table-cell list, per-phase stride, phase count).
fn layout(version: u32) -> Option<(&'static [u32], usize, usize)> {
    match version {
        VERSION => Some((&CELLS_V1, PHASE_STRIDE, N_PHASES)),
        VERSION_V2 => Some((&CELLS_V2, PHASE_STRIDE_V2, N_PHASES)),
        VERSION_V3 => Some((&CELLS_V3, PHASE_STRIDE_V3, N_PHASES)),
        // v4: same 11 pattern tables + scalars + bias as v3, but the
        // finer 30-phase (2-disc) bucketing.
        VERSION_V4 => Some((&CELLS_V3, PHASE_STRIDE_V3, N_PHASES_V4)),
        // v5: 15 pattern tables, 15 phases.
        VERSION_V5 => Some((&CELLS_V5, PHASE_STRIDE_V5, N_PHASES)),
        _ => None,
    }
}

/// Loads a weight file; returns the weights and the file version.
pub fn load(path: &str) -> Result<(Vec<i16>, u32), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut header = [0u8; 20];
    f.read_exact(&mut header)
        .map_err(|e| format!("{}: not a versor weight file (short header: {})", path, e))?;
    if &header[0..4] != MAGIC {
        return Err(format!("{}: not a versor weight file (bad magic)", path));
    }
    let word = |h: &[u8], i: usize| u32::from_le_bytes(h[i..i + 4].try_into().unwrap());
    let version = word(&header, 4);
    let (expected_cells, stride, n_phases) =
        layout(version).ok_or_else(|| format!("{}: unsupported version {}", path, version))?;
    if word(&header, 8) != SCALE
        || word(&header, 12) != n_phases as u32
        || word(&header, 16) != expected_cells.len() as u32
    {
        return Err(format!("{}: header mismatch", path));
    }
    let mut cells_raw = vec![0u8; expected_cells.len() * 4];
    f.read_exact(&mut cells_raw)
        .map_err(|e| format!("{}: short table list: {}", path, e))?;
    for (i, &cells) in expected_cells.iter().enumerate() {
        if word(&cells_raw, i * 4) != cells {
            return Err(format!("{}: table {} cell count mismatch", path, i));
        }
    }
    let n = n_phases * stride;
    let mut raw = vec![0u8; n * 2];
    f.read_exact(&mut raw)
        .map_err(|e| format!("{}: short weight data: {}", path, e))?;
    let mut rest = Vec::new();
    f.read_to_end(&mut rest).ok();
    if !rest.is_empty() {
        return Err(format!("{}: trailing bytes", path));
    }
    Ok((
        raw.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect(),
        version,
    ))
}

/// Write a v1 weight vector (used only by the round-trip unit test; the
/// Python trainer writes the trained files).
pub fn save(path: &str, weights: &[i16]) -> Result<(), String> {
    if weights.len() != N_PHASES * PHASE_STRIDE {
        return Err(format!(
            "expected {} weights, got {}",
            N_PHASES * PHASE_STRIDE,
            weights.len()
        ));
    }
    let mut header = Vec::new();
    header.extend_from_slice(MAGIC);
    for v in [VERSION, SCALE, N_PHASES as u32, CELLS_V1.len() as u32] {
        header.extend_from_slice(&v.to_le_bytes());
    }
    for cells in CELLS_V1 {
        header.extend_from_slice(&cells.to_le_bytes());
    }
    let mut f = std::fs::File::create(path).map_err(|e| format!("{}: {}", path, e))?;
    f.write_all(&header).map_err(|e| e.to_string())?;
    let mut raw = Vec::with_capacity(weights.len() * 2);
    for w in weights {
        raw.extend_from_slice(&w.to_le_bytes());
    }
    f.write_all(&raw).map_err(|e| e.to_string())?;
    Ok(())
}

// Multi-ProbCut parameters. Own implementation based on the
// descriptions in M. Buro's Multi-ProbCut papers: at non-PV
// nodes a shallow search predicts the deep result through a linear
// model v_deep ~ a * v_shallow + b with residual sigma; when the
// prediction clears beta (or alpha) with confidence t the node is cut.
// Parameters are fitted offline (python/mpc_fit.py) and loaded from a
// text file: lines "phase d dprime a b sigma".
//
// Design decision: MPC applies only to the midgame search. The endgame
// solver must stay exact (its FFO scores are checked against the
// published values), so it never uses MPC.

#[derive(Clone, Copy)]
pub struct MpcEntry {
    pub dprime: u32,
    pub a: f64,
    pub b: f64,
    pub sigma: f64,
}

pub struct MpcParams {
    // [phase 0..15][depth 0..=12]
    entries: Vec<Option<MpcEntry>>,
}

const MAX_DEPTH: usize = 13;

impl MpcParams {
    pub fn from_file(path: &str) -> Result<MpcParams, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        let mut entries = vec![None; 15 * MAX_DEPTH];
        let mut count = 0;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 6 {
                return Err(format!("{}: bad line {:?}", path, line));
            }
            let phase: usize = f[0].parse().map_err(|_| "bad phase")?;
            let d: usize = f[1].parse().map_err(|_| "bad d")?;
            let e = MpcEntry {
                dprime: f[2].parse().map_err(|_| "bad dprime")?,
                a: f[3].parse().map_err(|_| "bad a")?,
                b: f[4].parse().map_err(|_| "bad b")?,
                sigma: f[5].parse().map_err(|_| "bad sigma")?,
            };
            if phase >= 15 || d >= MAX_DEPTH || e.a.abs() < 1e-6 {
                return Err(format!("{}: out of range line {:?}", path, line));
            }
            entries[phase * MAX_DEPTH + d] = Some(e);
            count += 1;
        }
        if count == 0 {
            return Err(format!("{}: no entries", path));
        }
        Ok(MpcParams { entries })
    }

    pub fn get(&self, phase: usize, depth: u32) -> Option<MpcEntry> {
        if phase >= 15 || depth as usize >= MAX_DEPTH {
            return None;
        }
        self.entries[phase * MAX_DEPTH + depth as usize]
    }
}

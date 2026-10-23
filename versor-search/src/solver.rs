// Endgame solver: fail-soft alpha-beta on the exact final score
// (disc difference with empty squares awarded to the winner), with a
// transposition table, opponent-mobility move ordering and optional
// quadrant parity ordering in the low-empties region.
//
// Pass convention (the same as perft): a forced pass switches the side
// to move; two consecutive passes end the game.

use std::time::{Duration, Instant};

use versor_core::{bitboard, Position};

use crate::eval::{terminal_score, SCORE_INF};
use crate::tt::{Table, BOUND_EXACT, BOUND_LOWER, BOUND_UPPER, NO_MOVE};
use crate::zobrist;

/// Exact scores are in [-64, 64]; this window bounds them strictly.
pub const SOLVE_INF: i32 = 65;

/// Empties at or below which the solver skips the transposition table
/// and heavy move ordering (shallow fast path).
const SHALLOW_EMPTIES: u32 = 6;
/// Empties at or above which heavy (opponent mobility) ordering is used.
const HEAVY_ORDER_EMPTIES: u32 = 9;
/// Empties at or below which quadrant parity ordering is applied.
const PARITY_EMPTIES: u32 = 16;
/// Empties below which the stability cutoff is not attempted.
///
/// Measured, not guessed: instructions retired (which, unlike wall-clock
/// time, is a property of this process alone and so is unaffected by
/// what else the machine is running) falls monotonically as this
/// threshold is lowered, all the way to 1. Computing stable discs is
/// pure register work, while a solver node costs a transposition table
/// probe and a flip computation, so the cutoff pays for itself even at
/// the very end of the game. `versor-cli prune solveset` runs one
/// configuration per process for that measurement.
const STABILITY_MIN_EMPTIES: u32 = 1;

#[derive(Clone, Copy, Debug)]
pub struct SolverOptions {
    pub use_tt: bool,
    pub use_ordering: bool,
    pub use_parity: bool,
    /// Stability cutoff: bound the final score from above with
    /// the opponent's stable discs and fail low when that bound is
    /// already at or below alpha. Backward pruning, so exact.
    pub use_stability: bool,
    /// Empties at or below which the stability cutoff is attempted.
    /// Above this the stable set is usually empty and the computation is
    /// wasted.
    pub stability_empties: u32,
    /// Empties below which the stability cutoff is not attempted: near
    /// the leaves the subtree saved is smaller than the cost of the
    /// stable-disc computation.
    pub stability_min_empties: u32,
    /// Enhanced transposition cutoff: probe the transposition
    /// table for each child before searching any of them, and fail high
    /// as soon as one child's stored bound already proves a cutoff.
    /// Backward pruning, so exact.
    pub use_etc: bool,
    /// Empties at or above which ETC is attempted. Near the leaves the
    /// extra probes cost more than the subtree they would save.
    pub etc_empties: u32,
}

impl Default for SolverOptions {
    fn default() -> Self {
        SolverOptions {
            use_tt: true,
            use_ordering: true,
            use_parity: true,
            use_stability: true,
            stability_empties: 24,
            stability_min_empties: STABILITY_MIN_EMPTIES,
            use_etc: true,
            // Below 8 empties the solver skips the transposition table
            // entirely (SHALLOW_EMPTIES), so probing for children there
            // finds nothing and only costs work.
            etc_empties: 8,
        }
    }
}

/// Counters for tuning the two exact cutoffs. Call counts and cutoff
/// counts only; nothing here depends on wall-clock time.
#[derive(Clone, Copy, Debug, Default)]
pub struct CutoffStats {
    pub stability_calls: u64,
    pub stability_cuts: u64,
    pub etc_calls: u64,
    pub etc_cuts: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wld {
    Loss,
    Draw,
    Win,
}

#[derive(Clone, Debug)]
pub struct SolveResult {
    /// Exact score for `solve_exact`; for `solve_wld` only the sign is
    /// meaningful (fail-soft bound from a narrow window).
    pub score: i32,
    /// None means the game is already over or the side to move passes.
    pub best_move: Option<u8>,
    pub nodes: u64,
    pub elapsed: Duration,
}

pub struct Solver {
    pub opts: SolverOptions,
    tt: Table,
    nodes: u64,
    deadline: Option<Instant>,
    /// Optional node budget. Unlike a deadline this is independent of
    /// machine load, so a run bounded by it is reproducible even on a
    /// busy host.
    node_limit: Option<u64>,
    aborted: bool,
    /// Optional static evaluator for move ordering in the deep region.
    /// Replaces the mobility heuristic when set; parity stays as a
    /// tiebreak. Ordering only affects node counts.
    eval_order: Option<Box<dyn crate::eval::Eval>>,
    stats: CutoffStats,
}

const MAX_MOVES: usize = 64;

// Board quadrants: files a-d / e-h crossed with ranks 1-4 / 5-8.
const QUADRANTS: [u64; 4] = [
    0x0000_0000_0f0f_0f0f,
    0x0000_0000_f0f0_f0f0,
    0x0f0f_0f0f_0000_0000,
    0xf0f0_f0f0_0000_0000,
];

fn quadrant_of(sq: u8) -> usize {
    ((sq >> 2) & 1) as usize + (((sq >> 5) & 1) as usize) * 2
}

impl Solver {
    pub fn new(tt_mb: usize) -> Solver {
        Solver {
            opts: SolverOptions::default(),
            tt: Table::new(tt_mb),
            nodes: 0,
            deadline: None,
            node_limit: None,
            aborted: false,
            eval_order: None,
            stats: CutoffStats::default(),
        }
    }

    /// Install a static evaluator for deep-region move ordering.
    pub fn set_eval_order(&mut self, eval: Box<dyn crate::eval::Eval>) {
        self.eval_order = Some(eval);
    }

    /// Cutoff counters accumulated by the last solve.
    pub fn stats(&self) -> CutoffStats {
        self.stats
    }

    /// Abort a solve once it has visited this many nodes. `None` removes
    /// the budget. A solve that hits the budget returns `None`, exactly
    /// as one that hits a deadline does.
    pub fn set_node_limit(&mut self, limit: Option<u64>) {
        self.node_limit = limit;
    }

    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    /// Forget everything from earlier solves: the transposition table
    /// (and its generation) and the counters. The options, the node
    /// limit and the ordering evaluation stay. Called at the start of
    /// every game and before every position of a benchmark, for the
    /// same reason as `Searcher::new_game`.
    pub fn new_game(&mut self) {
        self.tt.clear();
        self.nodes = 0;
        self.stats = CutoffStats::default();
        self.deadline = None;
        self.aborted = false;
    }

    /// Solve for the exact final score under optimal play by both
    /// sides. Returns None if the optional time limit was exceeded.
    pub fn solve_exact(
        &mut self,
        pos: &Position,
        limit: Option<Duration>,
    ) -> Option<SolveResult> {
        self.solve_with_window(pos, limit, -SOLVE_INF, SOLVE_INF)
            .map(|r| r.1)
    }

    /// Win/loss/draw for the side to move. Cheaper than `solve_exact`
    /// because it searches a unit window around zero.
    pub fn solve_wld(
        &mut self,
        pos: &Position,
        limit: Option<Duration>,
    ) -> Option<(Wld, SolveResult)> {
        self.solve_with_window(pos, limit, -1, 1).map(|(v, r)| {
            let wld = if v > 0 {
                Wld::Win
            } else if v < 0 {
                Wld::Loss
            } else {
                Wld::Draw
            };
            (wld, r)
        })
    }

    fn solve_with_window(
        &mut self,
        pos: &Position,
        limit: Option<Duration>,
        alpha: i32,
        beta: i32,
    ) -> Option<(i32, SolveResult)> {
        let start = Instant::now();
        self.nodes = 0;
        self.aborted = false;
        self.stats = CutoffStats::default();
        self.deadline = limit.map(|d| start + d);
        self.tt.new_generation();
        let (score, best_move) = self.solve_root(pos, alpha, beta);
        if self.aborted {
            return None;
        }
        let result = SolveResult {
            score,
            best_move,
            nodes: self.nodes,
            elapsed: start.elapsed(),
        };
        Some((score, result))
    }

    fn solve_root(&mut self, pos: &Position, mut alpha: i32, beta: i32) -> (i32, Option<u8>) {
        self.nodes += 1;
        let empties = pos.empty().count_ones();
        let moves = pos.legal_moves();
        if moves == 0 {
            if bitboard::legal_moves(pos.opp, pos.own) == 0 {
                return (terminal_score(pos), None);
            }
            let hash = zobrist::hash(&pos.pass());
            let v = -self.end(&pos.pass(), hash, -beta, -alpha, empties);
            return (v, None);
        }
        let hash = zobrist::hash(pos);
        let mut list = [(0u8, 0u64, 0i32); MAX_MOVES];
        let n = self.build_move_list(pos, moves, NO_MOVE, empties, &mut list);
        let mut best = -SCORE_INF;
        let mut best_move = list[0].0;
        for item in list.iter().take(n) {
            let (sq, flips, _) = *item;
            let child = pos.play_with_flips(sq, flips);
            let child_hash = zobrist::child_hash(hash, pos.stm, sq, flips);
            let v = -self.end(&child, child_hash, -beta, -alpha, empties - 1);
            if self.aborted {
                return (0, None);
            }
            if v > best {
                best = v;
                best_move = sq;
            }
            if best > alpha {
                alpha = best;
                if alpha >= beta {
                    break;
                }
            }
        }
        (best, Some(best_move))
    }

    fn end(&mut self, pos: &Position, hash: u64, mut alpha: i32, beta: i32, empties: u32) -> i32 {
        self.nodes += 1;
        if let Some(limit) = self.node_limit {
            if self.nodes >= limit {
                self.aborted = true;
            }
        }
        if self.nodes & 0x0fff == 0 {
            if let Some(d) = self.deadline {
                if Instant::now() >= d {
                    self.aborted = true;
                }
            }
        }
        if self.aborted {
            return 0;
        }
        let moves = pos.legal_moves();
        if moves == 0 {
            if bitboard::legal_moves(pos.opp, pos.own) == 0 {
                return terminal_score(pos);
            }
            return -self.end(&pos.pass(), zobrist::pass_hash(hash), -beta, -alpha, empties);
        }

        let use_tt = self.opts.use_tt && empties > SHALLOW_EMPTIES;
        let alpha_orig = alpha;
        let mut tt_move = NO_MOVE;
        if use_tt {
            if let Some(e) = self.tt.probe(hash) {
                let v = i32::from(e.value);
                match e.bound {
                    BOUND_EXACT => return v,
                    BOUND_LOWER if v >= beta => return v,
                    BOUND_UPPER if v <= alpha => return v,
                    _ => {}
                }
                tt_move = e.mv;
            }
        }

        // Stability cutoff. The opponent's stable discs put a ceiling on
        // the final score for the side to move; if that ceiling is
        // already at or below alpha, no continuation can raise the score
        // into the window. Fail-soft: the bound itself is returned.
        if self.opts.use_stability
            && empties <= self.opts.stability_empties
            && empties >= self.opts.stability_min_empties
        {
            self.stats.stability_calls += 1;
            let upper = versor_core::score_upper_bound(pos.own, pos.opp);
            if upper <= alpha {
                self.stats.stability_cuts += 1;
                return upper;
            }
        }

        let mut list = [(0u8, 0u64, 0i32); MAX_MOVES];
        let n = self.build_move_list(pos, moves, tt_move, empties, &mut list);

        // Enhanced transposition cutoff. Every child hash is probed
        // before any child is searched. A stored upper bound v on a
        // child means our score through that move is at least -v, so
        // -v >= beta is a cutoff. The solver always searches to the end
        // of the game, so every stored entry is full depth and no depth
        // check is needed.
        if self.opts.use_etc && use_tt && empties >= self.opts.etc_empties {
            self.stats.etc_calls += 1;
            for item in list.iter().take(n) {
                let (sq, flips, _) = *item;
                let child_hash = zobrist::child_hash(hash, pos.stm, sq, flips);
                if let Some(e) = self.tt.probe(child_hash) {
                    if e.bound == BOUND_EXACT || e.bound == BOUND_UPPER {
                        let v = -i32::from(e.value);
                        if v >= beta {
                            self.stats.etc_cuts += 1;
                            return v;
                        }
                    }
                }
            }
        }

        let mut best = -SCORE_INF;
        let mut best_move = NO_MOVE;
        for item in list.iter().take(n) {
            let (sq, flips, _) = *item;
            let child = pos.play_with_flips(sq, flips);
            let child_hash = zobrist::child_hash(hash, pos.stm, sq, flips);
            let v = -self.end(&child, child_hash, -beta, -alpha, empties - 1);
            if v > best {
                best = v;
                best_move = sq;
            }
            if best > alpha {
                alpha = best;
                if alpha >= beta {
                    break;
                }
            }
        }
        if use_tt && !self.aborted {
            let bound = if best <= alpha_orig {
                BOUND_UPPER
            } else if best >= beta {
                BOUND_LOWER
            } else {
                BOUND_EXACT
            };
            self.tt.store(hash, best, empties, bound, best_move);
        }
        best
    }

    fn build_move_list(
        &self,
        pos: &Position,
        moves: u64,
        tt_move: u8,
        empties: u32,
        list: &mut [(u8, u64, i32); MAX_MOVES],
    ) -> usize {
        const CORNERS: u64 = 0x8100_0000_0000_0081;
        let heavy = self.opts.use_ordering && empties >= HEAVY_ORDER_EMPTIES;
        let parity = self.opts.use_parity && empties <= PARITY_EMPTIES;
        let mut odd_quadrant = [false; 4];
        if parity {
            let empty = pos.empty();
            for (i, q) in QUADRANTS.iter().enumerate() {
                odd_quadrant[i] = (empty & q).count_ones() & 1 == 1;
            }
        }
        let mut n = 0;
        let mut m = moves;
        while m != 0 {
            let sq = m.trailing_zeros() as u8;
            m &= m - 1;
            let flips = bitboard::flips(pos.own, pos.opp, sq);
            let mut score = 0i32;
            if sq == tt_move {
                score = i32::MAX;
            } else {
                if heavy {
                    let child = pos.play_with_flips(sq, flips);
                    if let Some(eval) = &self.eval_order {
                        score += -4 * eval.eval(&child);
                    } else {
                        let opp_mob = child.legal_moves().count_ones() as i32;
                        let corner = if CORNERS >> sq & 1 == 1 { 1000 } else { 0 };
                        score += corner - 10 * opp_mob;
                    }
                }
                if parity && odd_quadrant[quadrant_of(sq)] {
                    score += 1;
                }
            }
            list[n] = (sq, flips, score);
            n += 1;
        }
        for i in 1..n {
            let item = list[i];
            let mut j = i;
            while j > 0 && list[j - 1].2 < item.2 {
                list[j] = list[j - 1];
                j -= 1;
            }
            list[j] = item;
        }
        n
    }
}

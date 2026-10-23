// Midgame search: fail-soft negamax with alpha-beta pruning, iterative
// deepening, PVS, a transposition table, move ordering, Multi-ProbCut
// and the two exact cutoffs. The heuristics and cutoffs can be switched
// off through `SearchOptions` for measurement; Multi-ProbCut is used
// only after `set_mpc`.
//
// Pass convention (the same as perft): a forced pass switches the side
// to move without consuming depth. When neither side can move the
// position is terminal and scored exactly.

use std::time::{Duration, Instant};

use versor_core::{bitboard, Position};

use crate::eval::{terminal_score, Evaluator, SCORE_INF};
use crate::mpc::MpcParams;
use crate::tt::{Table, BOUND_EXACT, BOUND_LOWER, BOUND_UPPER, NO_MOVE};
use crate::zobrist;

#[derive(Clone, Copy, Debug)]
pub struct SearchOptions {
    pub use_tt: bool,
    pub use_pvs: bool,
    pub use_ordering: bool,
    /// Order moves by a static evaluation of the child position
    /// instead of the opponent-mobility heuristic (useful with a
    /// trained evaluator). Only affects node counts, not values.
    pub use_eval_ordering: bool,
    /// Killer-move ordering: two moves per draft that caused a beta
    /// cutoff are tried early (node counts only).
    pub use_killers: bool,
    /// History-heuristic ordering: moves that caused cutoffs get a
    /// per-square bonus (node counts only).
    pub use_history: bool,
    /// Aspiration windows around the previous iteration's value in the
    /// root iterative deepening (node counts only).
    pub use_aspiration: bool,
    /// Enhanced transposition cutoff: probe the transposition
    /// table for every child before searching any of them, and fail high
    /// as soon as a stored child bound already proves the cutoff. Uses
    /// exactly the same entry-acceptance rule as the ordinary probe at
    /// the child, so the value is the one the child search would have
    /// returned. Backward pruning, so exact.
    pub use_etc: bool,
    /// Draft at or above which ETC is attempted. Near the leaves the
    /// extra hashing and probing costs more than the subtree it saves.
    pub etc_min_depth: u32,
    /// Stability cutoff in the midgame searcher.
    ///
    /// The bound derived from stable discs is a bound on the FINAL score
    /// of the game, so it may only be used where the search value is a
    /// final score rather than a heuristic estimate. That is exactly the
    /// region `depth >= empties`: a move consumes one draft and fills
    /// one square, and a pass consumes neither, so `depth - empties` is
    /// invariant down the tree and draft 0 can only be reached with the
    /// board full, which the terminal test catches first. Every leaf
    /// below such a node is therefore terminal and exactly scored.
    /// Outside that region the cutoff is skipped, which is why a plain
    /// fixed-depth midgame search sees no effect from it.
    pub use_stability: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            use_tt: true,
            use_pvs: true,
            use_ordering: true,
            use_eval_ordering: false,
            use_killers: true,
            use_history: true,
            use_aspiration: true,
            use_etc: true,
            etc_min_depth: 5,
            use_stability: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Value from the side to move's perspective, disc scale.
    pub value: i32,
    /// None means the side to move has no legal move (pass) or the
    /// game is over.
    pub best_move: Option<u8>,
    pub depth: u32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: Vec<u8>,
}

/// Killer/history tables are indexed by draft (remaining depth), capped
/// at this bound; deeper drafts share the top slot.
const MAX_DRAFT: usize = 64;

pub struct Searcher<E: Evaluator> {
    pub opts: SearchOptions,
    eval: E,
    tt: Table,
    nodes: u64,
    /// Multi-ProbCut parameters and confidence threshold t (midgame
    /// only; never used by the endgame solver).
    mpc: Option<(MpcParams, f64)>,
    /// Two killer moves per draft (NO_MOVE when empty).
    killers: [[u8; 2]; MAX_DRAFT],
    /// Per-square history score of cutoff-causing moves.
    history: [u32; 64],
    /// Cutoff counters of the last search (counts only, no timing).
    etc_calls: u64,
    etc_cuts: u64,
    stability_cuts: u64,
    /// Wall-clock deadline for the search in progress, set by
    /// `search_timed` and cleared when it returns. `None` for the
    /// depth-limited and node-limited searches, which stay exactly
    /// reproducible.
    deadline: Option<Instant>,
    /// Set once the deadline passes. Every node returns immediately
    /// while it is set and the caller throws the iteration away, so the
    /// value that unwinds is meaningless by design. Nothing is written
    /// to the transposition table while it is set.
    aborted: bool,
}

/// How often the deadline is checked, in nodes. `Instant::now` is far
/// more expensive than a node, so it is read once per this many.
const ABORT_CHECK_MASK: u64 = 0xFFF; // every 4096 nodes

/// Wall-clock budget for one move.
#[derive(Clone, Copy, Debug)]
pub struct TimeBudget {
    /// Target for this move. Iterations are started or skipped against
    /// this (see `search_timed`).
    pub alloc: Duration,
    /// Absolute cap. An iteration that overruns it is abandoned, so the
    /// search returns shortly after this no matter how badly the
    /// per-iteration estimate was off.
    pub hard: Duration,
}

// Upper bound on the move list length. Reachable positions never have
// more than 33 legal moves, but text-loaded positions are arbitrary, so
// the full 64 is used to stay safe.
const MAX_MOVES: usize = 64;

impl<E: Evaluator> Searcher<E> {
    pub fn new(eval: E, tt_mb: usize) -> Searcher<E> {
        Searcher {
            opts: SearchOptions::default(),
            eval,
            tt: Table::new(tt_mb),
            nodes: 0,
            mpc: None,
            killers: [[NO_MOVE; 2]; MAX_DRAFT],
            history: [0; 64],
            etc_calls: 0,
            etc_cuts: 0,
            stability_cuts: 0,
            deadline: None,
            aborted: false,
        }
    }

    /// (ETC attempts, ETC cutoffs) in the last search.
    pub fn etc_stats(&self) -> (u64, u64) {
        (self.etc_calls, self.etc_cuts)
    }

    pub fn stability_cuts(&self) -> u64 {
        self.stability_cuts
    }

    fn reset_heuristics(&mut self) {
        self.killers = [[NO_MOVE; 2]; MAX_DRAFT];
        // Decay history rather than clearing, so ordering knowledge
        // carries across iterative-deepening iterations but fades.
        for h in self.history.iter_mut() {
            *h /= 2;
        }
    }

    fn record_cutoff(&mut self, sq: u8, depth: u32) {
        let d = (depth as usize).min(MAX_DRAFT - 1);
        if self.opts.use_killers && self.killers[d][0] != sq {
            self.killers[d][1] = self.killers[d][0];
            self.killers[d][0] = sq;
        }
        if self.opts.use_history {
            self.history[sq as usize] =
                self.history[sq as usize].saturating_add(depth * depth);
        }
    }

    pub fn set_mpc(&mut self, params: MpcParams, t: f64) {
        self.mpc = Some((params, t));
    }

    /// Iterative deepening until the node budget is exhausted: runs
    /// whole iterations and stops before starting one once `budget`
    /// nodes have been spent (the last iteration may overshoot; both
    /// sides of a fixed-node match use the same rule).
    pub fn search_nodes(&mut self, pos: &Position, budget: u64) -> SearchResult {
        let start = Instant::now();
        self.nodes = 0;
        self.etc_calls = 0;
        self.etc_cuts = 0;
        self.stability_cuts = 0;
        self.tt.new_generation();
        self.reset_heuristics();
        let hash = zobrist::hash(pos);
        let st = self.eval.init_state(pos);
        let mut value = 0;
        let mut best_move = None;
        let mut depth = 0;
        for d in 1..=60u32 {
            if d > 1 && self.nodes >= budget {
                break;
            }
            let (v, m) = self.ab_root_aspiration(pos, &st, hash, d, value);
            value = v;
            best_move = m;
            depth = d;
        }
        let pv = self.extract_pv(pos, best_move, depth);
        SearchResult {
            value,
            best_move,
            depth,
            nodes: self.nodes,
            elapsed: start.elapsed(),
            pv,
        }
    }

    /// Iterative deepening under a wall-clock budget, for play with a
    /// game clock.
    ///
    /// Two mechanisms, doing different jobs:
    ///
    /// 1. Before each iteration, decide whether to start it at all.
    ///    The rule follows the one described for BILL: under 40 % of the
    ///    allocation spent, start the next iteration; between 40 % and
    ///    100 %, start it only if the best move changed between the last
    ///    two completed iterations (a move that has stopped moving is
    ///    unlikely to move again); at or over 100 %, stop.
    /// 2. A hard deadline inside the search itself, so that an
    ///    iteration which was started and then turned out to be far more
    ///    expensive than the last one cannot eat the clock. This is the
    ///    backstop for when rule 1 guesses wrong.
    ///
    /// Depth 1 is searched with no deadline at all, so that a move is
    /// always returned even when the budget is already exhausted.
    pub fn search_timed(&mut self, pos: &Position, budget: TimeBudget) -> SearchResult {
        let start = Instant::now();
        self.nodes = 0;
        self.etc_calls = 0;
        self.etc_cuts = 0;
        self.stability_cuts = 0;
        self.tt.new_generation();
        self.reset_heuristics();
        let hash = zobrist::hash(pos);
        let st = self.eval.init_state(pos);
        self.aborted = false;
        self.deadline = None;

        // Depth 1 first, without a deadline: cheap, and it guarantees a
        // legal move even if there is no time left at all.
        let (mut value, mut best_move) = self.ab_root_aspiration(pos, &st, hash, 1, 0);
        let mut depth = 1;
        let mut prev_best: Option<u8> = None;

        self.deadline = Some(start + budget.hard);
        for d in 2..=60u32 {
            let used = start.elapsed();
            if used >= budget.alloc {
                break;
            }
            // 40 % of the allocation, in integer arithmetic.
            if used * 5 >= budget.alloc * 2 && best_move == prev_best {
                break;
            }
            let (v, m) = self.ab_root_aspiration(pos, &st, hash, d, value);
            if self.aborted {
                break;
            }
            prev_best = best_move;
            value = v;
            best_move = m;
            depth = d;
        }
        self.deadline = None;
        self.aborted = false;

        let pv = self.extract_pv(pos, best_move, depth);
        SearchResult {
            value,
            best_move,
            depth,
            nodes: self.nodes,
            elapsed: start.elapsed(),
            pv,
        }
    }

    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    /// Forget everything learned from earlier searches: the
    /// transposition table (and its generation), the history and killer
    /// tables, and the counters. The options, the evaluation and the
    /// Multi-ProbCut parameters stay.
    ///
    /// Called at the start of every game, and before every position of
    /// a benchmark. Within a game the history carries from one move to
    /// the next (decayed by `reset_heuristics`); across games it must
    /// not, or a game's moves depend on the games the same searcher
    /// played before it, and a match with several games at once gives
    /// different games depending on which worker played what.
    pub fn new_game(&mut self) {
        self.tt.clear();
        self.killers = [[NO_MOVE; 2]; MAX_DRAFT];
        self.history = [0; 64];
        self.nodes = 0;
        self.etc_calls = 0;
        self.etc_cuts = 0;
        self.stability_cuts = 0;
        self.deadline = None;
        self.aborted = false;
    }

    /// Depth-limited search with iterative deepening from depth 1.
    pub fn search(&mut self, pos: &Position, depth: u32) -> SearchResult {
        let start = Instant::now();
        let depth = depth.max(1);
        self.nodes = 0;
        self.etc_calls = 0;
        self.etc_cuts = 0;
        self.stability_cuts = 0;
        self.tt.new_generation();
        self.reset_heuristics();
        let hash = zobrist::hash(pos);
        let st = self.eval.init_state(pos);
        let mut value = 0;
        let mut best_move = None;
        for d in 1..=depth {
            let (v, m) = self.ab_root_aspiration(pos, &st, hash, d, value);
            value = v;
            best_move = m;
        }
        let pv = self.extract_pv(pos, best_move, depth);
        SearchResult {
            value,
            best_move,
            depth,
            nodes: self.nodes,
            elapsed: start.elapsed(),
            pv,
        }
    }

    // Root search with an aspiration window around the previous
    // iteration's value; widens and re-searches on fail-high/low so the
    // returned value is always exact.
    fn ab_root_aspiration(
        &mut self,
        pos: &Position,
        st: &E::State,
        hash: u64,
        depth: u32,
        prev: i32,
    ) -> (i32, Option<u8>) {
        if !self.opts.use_aspiration || depth < 3 {
            return self.ab_root(pos, st, hash, depth, -SCORE_INF, SCORE_INF);
        }
        let mut window = 3;
        let mut alpha = (prev - window).max(-SCORE_INF);
        let mut beta = (prev + window).min(SCORE_INF);
        loop {
            let (v, m) = self.ab_root(pos, st, hash, depth, alpha, beta);
            if self.aborted {
                return (v, m);
            }
            if v <= alpha && alpha > -SCORE_INF {
                window = (window * 3).min(2 * SCORE_INF);
                alpha = (v - window).max(-SCORE_INF);
            } else if v >= beta && beta < SCORE_INF {
                window = (window * 3).min(2 * SCORE_INF);
                beta = (v + window).min(SCORE_INF);
            } else {
                return (v, m);
            }
        }
    }

    fn ab_root(
        &mut self,
        pos: &Position,
        st: &E::State,
        hash: u64,
        depth: u32,
        alpha0: i32,
        beta: i32,
    ) -> (i32, Option<u8>) {
        self.nodes += 1;
        let moves = pos.legal_moves();
        if moves == 0 {
            if bitboard::legal_moves(pos.opp, pos.own) == 0 {
                return (terminal_score(pos), None);
            }
            let (v, _) = self.ab_root(&pos.pass(), st, zobrist::pass_hash(hash), depth, -beta, -alpha0);
            return (-v, None);
        }
        let tt_move = if self.opts.use_tt {
            self.tt.probe(hash).map_or(NO_MOVE, |e| e.mv)
        } else {
            NO_MOVE
        };
        let mut list = [(0u8, 0u64, 0i32); MAX_MOVES];
        let n = self.build_move_list(pos, st, moves, tt_move, depth, &mut list);
        let mut alpha = alpha0;
        let mut best = -SCORE_INF;
        let mut best_move = list[0].0;
        for (i, item) in list.iter().enumerate().take(n) {
            let (sq, flips, _) = *item;
            let child = pos.play_with_flips(sq, flips);
            let child_st = self.eval.child_state(pos, st, sq, flips);
            let child_hash = zobrist::child_hash(hash, pos.stm, sq, flips);
            let v = if i == 0 || !self.opts.use_pvs {
                -self.ab(&child, &child_st, child_hash, depth - 1, -beta, -alpha)
            } else {
                let v = -self.ab(&child, &child_st, child_hash, depth - 1, -alpha - 1, -alpha);
                if v > alpha && v < beta {
                    -self.ab(&child, &child_st, child_hash, depth - 1, -beta, -alpha)
                } else {
                    v
                }
            };
            if v > best {
                best = v;
                best_move = sq;
            }
            if best > alpha {
                alpha = best;
                if alpha >= beta {
                    break; // fail-high against the aspiration window
                }
            }
        }
        // As in `ab`: nothing is stored from an abandoned iteration.
        if self.aborted {
            return (0, Some(best_move));
        }
        // Store as an exact bound only when the value is inside the
        // window; a fail-high/low is a lower/upper bound.
        if self.opts.use_tt {
            let bound = if best <= alpha0 {
                BOUND_UPPER
            } else if best >= beta {
                BOUND_LOWER
            } else {
                BOUND_EXACT
            };
            self.tt.store(hash, best, depth, bound, best_move);
        }
        (best, Some(best_move))
    }

    /// Poll the deadline (only every ABORT_CHECK_MASK+1 nodes) and
    /// report whether the search is being abandoned.
    #[inline]
    fn check_abort(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if let Some(d) = self.deadline {
            if self.nodes & ABORT_CHECK_MASK == 0 && Instant::now() >= d {
                self.aborted = true;
                return true;
            }
        }
        false
    }

    fn ab(
        &mut self,
        pos: &Position,
        st: &E::State,
        hash: u64,
        depth: u32,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        self.nodes += 1;
        if self.check_abort() {
            return 0;
        }
        let moves = pos.legal_moves();
        if moves == 0 {
            if bitboard::legal_moves(pos.opp, pos.own) == 0 {
                return terminal_score(pos);
            }
            return -self.ab(&pos.pass(), st, zobrist::pass_hash(hash), depth, -beta, -alpha);
        }
        if depth == 0 {
            return self.eval.eval_with(pos, st);
        }

        let alpha_orig = alpha;
        let mut tt_move = NO_MOVE;
        if self.opts.use_tt {
            if let Some(e) = self.tt.probe(hash) {
                if u32::from(e.depth) >= depth {
                    let v = i32::from(e.value);
                    match e.bound {
                        BOUND_EXACT => return v,
                        BOUND_LOWER if v >= beta => return v,
                        BOUND_UPPER if v <= alpha => return v,
                        _ => {}
                    }
                }
                tt_move = e.mv;
            }
        }

        // Multi-ProbCut at non-PV nodes (null windows only).
        if beta - alpha == 1 && depth >= 4 {
            if let Some((params, t)) = &self.mpc {
                let discs = (pos.own | pos.opp).count_ones() as usize;
                let phase = (discs.saturating_sub(4) / 4).min(14);
                if let Some(e) = params.get(phase, depth) {
                    let t = *t;
                    let hi = ((f64::from(beta) + t * e.sigma - e.b) / e.a).ceil() as i32;
                    if hi < SCORE_INF {
                        let v = self.ab(pos, st, hash, e.dprime, hi - 1, hi);
                        if v >= hi {
                            return beta;
                        }
                    }
                    let lo = ((f64::from(alpha) - t * e.sigma - e.b) / e.a).floor() as i32;
                    if lo > -SCORE_INF {
                        let v = self.ab(pos, st, hash, e.dprime, lo, lo + 1);
                        if v <= lo {
                            return alpha;
                        }
                    }
                }
            }
        }

        // Stability cutoff, valid only where every leaf below is
        // terminal so that the search value is the exact final score
        // (see `SearchOptions::use_stability`). The opponent's stable
        // discs cap the final score for the side to move; if that cap is
        // already at or below alpha, nothing below can reach the window.
        if self.opts.use_stability && depth >= pos.empty().count_ones() {
            let upper = versor_core::score_upper_bound(pos.own, pos.opp);
            if upper <= alpha {
                self.stability_cuts += 1;
                return upper;
            }
        }

        let mut list = [(0u8, 0u64, 0i32); MAX_MOVES];
        let n = self.build_move_list(pos, st, moves, tt_move, depth, &mut list);

        // Enhanced transposition cutoff. A stored upper bound v on a
        // child means our score through that move is at least -v, so
        // -v >= beta is a cutoff. The entry-acceptance rule below is the
        // same one the child's own probe would apply (depth >= draft,
        // then an upper bound at or below the child's alpha, which is
        // -beta), so this returns the value the child search would have
        // returned.
        if self.opts.use_etc && self.opts.use_tt && depth >= self.opts.etc_min_depth {
            self.etc_calls += 1;
            for item in list.iter().take(n) {
                let (sq, flips, _) = *item;
                let child_hash = zobrist::child_hash(hash, pos.stm, sq, flips);
                if let Some(e) = self.tt.probe(child_hash) {
                    if u32::from(e.depth) >= depth - 1
                        && (e.bound == BOUND_EXACT || e.bound == BOUND_UPPER)
                    {
                        let v = -i32::from(e.value);
                        if v >= beta {
                            self.etc_cuts += 1;
                            return v;
                        }
                    }
                }
            }
        }

        let mut best = -SCORE_INF;
        let mut best_move = NO_MOVE;
        for (i, item) in list.iter().enumerate().take(n) {
            let (sq, flips, _) = *item;
            let child = pos.play_with_flips(sq, flips);
            let child_st = self.eval.child_state(pos, st, sq, flips);
            let child_hash = zobrist::child_hash(hash, pos.stm, sq, flips);
            let v = if i == 0 || !self.opts.use_pvs {
                -self.ab(&child, &child_st, child_hash, depth - 1, -beta, -alpha)
            } else {
                let v = -self.ab(&child, &child_st, child_hash, depth - 1, -alpha - 1, -alpha);
                if v > alpha && v < beta {
                    -self.ab(&child, &child_st, child_hash, depth - 1, -beta, -alpha)
                } else {
                    v
                }
            };
            if v > best {
                best = v;
                best_move = sq;
            }
            if best > alpha {
                alpha = best;
                if alpha >= beta {
                    self.record_cutoff(sq, depth);
                    break;
                }
            }
        }
        // An abandoned search has explored only part of the move list,
        // so `best` is not a bound on anything. Never store it.
        if self.aborted {
            return 0;
        }
        if self.opts.use_tt {
            let bound = if best <= alpha_orig {
                BOUND_UPPER
            } else if best >= beta {
                BOUND_LOWER
            } else {
                BOUND_EXACT
            };
            self.tt.store(hash, best, depth, bound, best_move);
        }
        best
    }

    /// Fill `list` with (square, flips, order_score) sorted best-first.
    /// The transposition table move gets the highest score; the rest are
    /// ordered by corner preference and low opponent mobility when
    /// ordering is enabled (and the node is not just above the leaves).
    fn build_move_list(
        &self,
        pos: &Position,
        st: &E::State,
        moves: u64,
        tt_move: u8,
        depth: u32,
        list: &mut [(u8, u64, i32); MAX_MOVES],
    ) -> usize {
        const CORNERS: u64 = 0x8100_0000_0000_0081;
        // Ordering score bands (high to low): TT move, killer moves,
        // then positional heuristic (scaled) refined by history. This
        // keeps killers above every heuristic value while history only
        // breaks ties within a heuristic bucket.
        const KILLER0: i32 = 1 << 28;
        const KILLER1: i32 = (1 << 28) - 1;
        let d = (depth as usize).min(MAX_DRAFT - 1);
        let (k0, k1) = if self.opts.use_killers {
            (self.killers[d][0], self.killers[d][1])
        } else {
            (NO_MOVE, NO_MOVE)
        };
        let mut n = 0;
        let mut m = moves;
        let heavy = self.opts.use_ordering && depth >= 2;
        while m != 0 {
            let sq = m.trailing_zeros() as u8;
            m &= m - 1;
            let flips = bitboard::flips(pos.own, pos.opp, sq);
            let score = if sq == tt_move {
                i32::MAX
            } else if sq == k0 {
                KILLER0
            } else if sq == k1 {
                KILLER1
            } else {
                let heur = if heavy {
                    let child = pos.play_with_flips(sq, flips);
                    if self.opts.use_eval_ordering {
                        // The child is evaluated from the opponent's
                        // perspective; lower is better for the mover.
                        let child_st = self.eval.child_state(pos, st, sq, flips);
                        -self.eval.eval_with(&child, &child_st)
                    } else {
                        let opp_mob = child.legal_moves().count_ones() as i32;
                        let corner = if CORNERS >> sq & 1 == 1 { 100 } else { 0 };
                        corner - opp_mob
                    }
                } else {
                    0
                };
                let hist = if self.opts.use_history {
                    (self.history[sq as usize]).min(4095) as i32
                } else {
                    0
                };
                heur * 4096 + hist
            };
            list[n] = (sq, flips, score);
            n += 1;
        }
        // Insertion sort by descending score; stable, so equal scores
        // keep the deterministic square order.
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

    /// Reconstruct the principal variation by walking transposition
    /// table best moves. Passes are traversed transparently.
    fn extract_pv(&self, pos: &Position, first: Option<u8>, max_len: u32) -> Vec<u8> {
        let mut pv = Vec::new();
        let Some(first) = first else {
            return pv;
        };
        if !self.opts.use_tt {
            pv.push(first);
            return pv;
        }
        let mut p = *pos;
        let mut h = zobrist::hash(&p);
        for _ in 0..max_len * 2 {
            if pv.len() as u32 >= max_len {
                break;
            }
            if p.must_pass() {
                p = p.pass();
                h = zobrist::pass_hash(h);
                continue;
            }
            let mv = if pv.is_empty() {
                first
            } else {
                match self.tt.probe(h) {
                    Some(e) if e.mv != NO_MOVE => e.mv,
                    _ => break,
                }
            };
            let flips = p.flips_for(mv);
            if flips == 0 {
                break;
            }
            h = zobrist::child_hash(h, p.stm, mv, flips);
            p = p.play_with_flips(mv, flips);
            pv.push(mv);
            if p.is_game_over() {
                break;
            }
        }
        pv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::PlaceholderEval;

    fn midgame_position() -> Position {
        // A fixed opening line, deep enough to have a real move list.
        let mut pos = Position::initial();
        for sq in [44u8, 43, 34, 26, 20, 21, 29, 12] {
            if let Ok(next) = pos.play(sq) {
                pos = next;
            }
        }
        pos
    }

    #[test]
    fn an_abandoned_iteration_stores_nothing() {
        // The value that unwinds out of an abandoned search is
        // meaningless: only part of the move list was searched. If it
        // reached the transposition table it would corrupt every later
        // search that probed those positions, and the corruption would
        // be timing-dependent and so nearly impossible to reproduce.
        //
        // Give the search a long allocation (so it keeps starting new
        // iterations) but a hard cap that expires almost at once, then
        // check that nothing in the table claims a depth from an
        // iteration that never finished.
        let pos = midgame_position();
        let mut s = Searcher::new(PlaceholderEval, 32);
        // Aspiration windows are turned off for this check. With them
        // on, one iteration can call the root search several times, and
        // a re-search that completes legitimately stores a bound at its
        // depth even if a later re-search of the same iteration is then
        // abandoned. That entry is sound, but it would make "no entry
        // deeper than the last accepted iteration" the wrong test. With
        // one root call per iteration, any entry deeper than the last
        // accepted depth could only have come from an abandoned search.
        s.opts.use_aspiration = false;
        let r = s.search_timed(
            &pos,
            TimeBudget {
                alloc: Duration::from_secs(10),
                hard: Duration::from_millis(2),
            },
        );
        let hash = zobrist::hash(&pos);
        if let Some(e) = s.tt.probe(hash) {
            assert!(
                u32::from(e.depth) <= r.depth,
                "table holds depth {} at the root but only depth {} completed",
                e.depth,
                r.depth
            );
        }
    }

    #[test]
    fn a_generous_budget_leaves_the_result_untouched() {
        // Same depth, with and without the deadline machinery armed.
        let pos = midgame_position();
        let mut a = Searcher::new(PlaceholderEval, 32);
        let timed = a.search_timed(
            &pos,
            TimeBudget {
                alloc: Duration::from_millis(50),
                hard: Duration::from_millis(75),
            },
        );
        let mut b = Searcher::new(PlaceholderEval, 32);
        let plain = b.search(&pos, timed.depth);
        assert_eq!(timed.best_move, plain.best_move);
        assert_eq!(timed.value, plain.value);
    }
}

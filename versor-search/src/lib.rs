// versor-search: search for the versor reversi engine.
//
// Midgame search: fail-soft negamax with alpha-beta, iterative
// deepening, PVS, a transposition table, move ordering, Multi-ProbCut
// and two exact cutoffs (stable discs, enhanced transposition cutoff).
// Endgame: an exact solver on the final score. The evaluation sits
// behind the Evaluator trait; the pattern evaluation lives in
// versor-eval, and eval.rs has a small hand-written one.
//
// Score scale: disc difference from the side to move's perspective.
// Terminal positions score the disc difference with empty squares
// awarded to the winner, so heuristic search and the exact solver
// share one axis.

pub mod eval;
pub mod mpc;
pub mod reference;
pub mod search;
pub mod solver;
pub mod timectl;
pub mod tt;
pub mod zobrist;

pub use eval::{terminal_score, Eval, Evaluator, PlaceholderEval, SCORE_INF};
pub use mpc::MpcParams;
pub use search::{SearchOptions, SearchResult, Searcher, TimeBudget};
pub use solver::{CutoffStats, SolveResult, Solver, SolverOptions, Wld};
pub use timectl::{budget_for_move, budget_for_move_all_moves, moves_left, solver_budget};

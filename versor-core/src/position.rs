// Game position: two disjoint disc sets plus the side to move.

use crate::bitboard;

/// Disc color. Black moves first in the initial position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    Black,
    White,
}

impl Color {
    pub fn opponent(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

/// Final result of a finished game, decided by disc count.
/// Empty squares are not awarded to either side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    BlackWin,
    WhiteWin,
    Draw,
}

/// Error returned by `Position::play` for an illegal move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IllegalMove {
    pub sq: u8,
}

impl std::fmt::Display for IllegalMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "illegal move on square index {}", self.sq)
    }
}

impl std::error::Error for IllegalMove {}

/// A reversi position: discs of the side to move (`own`), discs of the
/// opponent (`opp`) and the side to move (`stm`). `own` and `opp` must
/// stay disjoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Position {
    pub own: u64,
    pub opp: u64,
    pub stm: Color,
}

impl Position {
    /// Standard initial position: white on d4 and e5, black on d5 and
    /// e4, black to move.
    pub fn initial() -> Position {
        const D4: u64 = 1 << 27;
        const E4: u64 = 1 << 28;
        const D5: u64 = 1 << 35;
        const E5: u64 = 1 << 36;
        Position {
            own: E4 | D5,
            opp: D4 | E5,
            stm: Color::Black,
        }
    }

    /// Discs of the given color.
    pub fn discs(&self, color: Color) -> u64 {
        if color == self.stm {
            self.own
        } else {
            self.opp
        }
    }

    pub fn black(&self) -> u64 {
        self.discs(Color::Black)
    }

    pub fn white(&self) -> u64 {
        self.discs(Color::White)
    }

    pub fn empty(&self) -> u64 {
        !(self.own | self.opp)
    }

    /// Bitboard of legal moves for the side to move.
    pub fn legal_moves(&self) -> u64 {
        bitboard::legal_moves(self.own, self.opp)
    }

    /// Discs flipped by playing on `sq`, or 0 if the move is illegal
    /// (occupied square or no flip).
    pub fn flips_for(&self, sq: u8) -> u64 {
        if sq >= 64 || (self.own | self.opp) >> sq & 1 == 1 {
            return 0;
        }
        bitboard::flips(self.own, self.opp, sq)
    }

    /// Play a move for the side to move, checking legality.
    pub fn play(&self, sq: u8) -> Result<Position, IllegalMove> {
        let flipped = self.flips_for(sq);
        if flipped == 0 {
            return Err(IllegalMove { sq });
        }
        Ok(self.apply(1u64 << sq, flipped))
    }

    /// Play a move that is known to be legal (checked only in debug builds).
    pub fn play_unchecked(&self, sq: u8) -> Position {
        debug_assert!(self.legal_moves() >> sq & 1 == 1);
        let flipped = bitboard::flips(self.own, self.opp, sq);
        self.apply(1u64 << sq, flipped)
    }

    /// Play a move whose flip set has already been computed (with
    /// `flips_for` or `bitboard::flips`), avoiding a recomputation.
    /// `flipped` must be the exact non-empty flip set for `sq` on this
    /// position (checked only in debug builds).
    pub fn play_with_flips(&self, sq: u8, flipped: u64) -> Position {
        debug_assert!(flipped != 0 && flipped == self.flips_for(sq));
        self.apply(1u64 << sq, flipped)
    }

    fn apply(&self, mv: u64, flipped: u64) -> Position {
        Position {
            own: self.opp ^ flipped,
            opp: self.own | mv | flipped,
            stm: self.stm.opponent(),
        }
    }

    /// Pass the turn. Only meaningful when the side to move has no legal
    /// move (checked in debug builds).
    pub fn pass(&self) -> Position {
        debug_assert_eq!(self.legal_moves(), 0);
        Position {
            own: self.opp,
            opp: self.own,
            stm: self.stm.opponent(),
        }
    }

    /// True if the side to move has no legal move but the game is not over.
    pub fn must_pass(&self) -> bool {
        self.legal_moves() == 0 && bitboard::legal_moves(self.opp, self.own) != 0
    }

    /// The game is over when neither side has a legal move (this includes
    /// a full board and wiped-out positions).
    pub fn is_game_over(&self) -> bool {
        self.legal_moves() == 0 && bitboard::legal_moves(self.opp, self.own) == 0
    }

    /// Result of a finished game by plain disc count, or None if the game
    /// is not over. Empty squares are not awarded to either side.
    pub fn result(&self) -> Option<GameResult> {
        if !self.is_game_over() {
            return None;
        }
        let diff = self.disc_diff();
        Some(if diff > 0 {
            GameResult::BlackWin
        } else if diff < 0 {
            GameResult::WhiteWin
        } else {
            GameResult::Draw
        })
    }

    /// Disc difference: black count minus white count, counting only
    /// the discs actually on the board. Use this for raw counts and
    /// display; for search, solver and training labels use
    /// `final_score`, which awards empty squares to the winner.
    pub fn disc_diff(&self) -> i32 {
        self.black().count_ones() as i32 - self.white().count_ones() as i32
    }

    /// Score of a finished game from the side to move's perspective:
    /// disc difference with empty squares awarded to the winner, the
    /// convention used by FFO, WThor and common training data (a
    /// wipeout scores +64 or -64, a draw 0). This is the single
    /// implementation of the convention; search, the endgame solver
    /// and training labels all use it. Meaningful only when the game
    /// is over; calling it earlier scores the position as if the game
    /// ended now.
    pub fn final_score(&self) -> i32 {
        let own = self.own.count_ones() as i32;
        let opp = self.opp.count_ones() as i32;
        let empty = 64 - own - opp;
        if own > opp {
            own - opp + empty
        } else if opp > own {
            own - opp - empty
        } else {
            0
        }
    }
}

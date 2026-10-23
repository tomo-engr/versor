// Bitboard primitives: directional shifts, legal move generation and
// flip computation. All functions treat two disjoint u64 disc sets
// ("own" = side to move, "opp" = opponent).
//
// Legal move generation uses the standard shift-and-propagate scheme:
// for each of the 8 directions, opponent discs adjacent to own discs are
// propagated up to 6 times (the maximum run length between two discs on
// an 8x8 board), then shifted once more onto empty squares.

/// Squares not on file a (used to mask off wrap-around when shifting west).
pub const NOT_FILE_A: u64 = 0xfefe_fefe_fefe_fefe;
/// Squares not on file h (used to mask off wrap-around when shifting east).
pub const NOT_FILE_H: u64 = 0x7f7f_7f7f_7f7f_7f7f;

// Rank 1 is the top row and lives in the low byte, so "north" (towards
// rank 1) is a right shift and "south" (towards rank 8) is a left shift.
#[inline(always)]
pub(crate) fn shift_n(x: u64) -> u64 {
    x >> 8
}
#[inline(always)]
pub(crate) fn shift_s(x: u64) -> u64 {
    x << 8
}
#[inline(always)]
pub(crate) fn shift_e(x: u64) -> u64 {
    (x & NOT_FILE_H) << 1
}
#[inline(always)]
pub(crate) fn shift_w(x: u64) -> u64 {
    (x & NOT_FILE_A) >> 1
}
#[inline(always)]
pub(crate) fn shift_ne(x: u64) -> u64 {
    (x & NOT_FILE_H) >> 7
}
#[inline(always)]
pub(crate) fn shift_nw(x: u64) -> u64 {
    (x & NOT_FILE_A) >> 9
}
#[inline(always)]
pub(crate) fn shift_se(x: u64) -> u64 {
    (x & NOT_FILE_H) << 9
}
#[inline(always)]
pub(crate) fn shift_sw(x: u64) -> u64 {
    (x & NOT_FILE_A) << 7
}

/// Bitboard of all legal moves for the side to move.
pub fn legal_moves(own: u64, opp: u64) -> u64 {
    debug_assert_eq!(own & opp, 0);
    let empty = !(own | opp);
    let mut moves = 0u64;
    macro_rules! dir {
        ($shift:expr) => {{
            let mut t = $shift(own) & opp;
            t |= $shift(t) & opp;
            t |= $shift(t) & opp;
            t |= $shift(t) & opp;
            t |= $shift(t) & opp;
            t |= $shift(t) & opp;
            moves |= $shift(t) & empty;
        }};
    }
    dir!(shift_n);
    dir!(shift_s);
    dir!(shift_e);
    dir!(shift_w);
    dir!(shift_ne);
    dir!(shift_nw);
    dir!(shift_se);
    dir!(shift_sw);
    moves
}

/// Bitboard of opponent discs flipped by playing on `sq`.
/// `sq` must be an empty square; returns 0 if the move flips nothing
/// (i.e. the move is illegal).
pub fn flips(own: u64, opp: u64, sq: u8) -> u64 {
    debug_assert!(sq < 64);
    debug_assert_eq!(own & opp, 0);
    let mv = 1u64 << sq;
    debug_assert_eq!(mv & (own | opp), 0);
    let mut flipped = 0u64;
    macro_rules! dir {
        ($shift:expr) => {{
            let mut x = $shift(mv);
            let mut cap = 0u64;
            while x & opp != 0 {
                cap |= x;
                x = $shift(x);
            }
            if x & own != 0 {
                flipped |= cap;
            }
        }};
    }
    dir!(shift_n);
    dir!(shift_s);
    dir!(shift_e);
    dir!(shift_w);
    dir!(shift_ne);
    dir!(shift_nw);
    dir!(shift_se);
    dir!(shift_sw);
    flipped
}

/// Index of the n-th (0-based) set bit of `bb`. `n` must be less than
/// `bb.count_ones()`.
pub fn nth_set_bit(mut bb: u64, mut n: u32) -> u8 {
    debug_assert!(n < bb.count_ones());
    loop {
        let sq = bb.trailing_zeros();
        if n == 0 {
            return sq as u8;
        }
        bb &= bb - 1;
        n -= 1;
    }
}

// Text I/O: square names (a1..h8), a simple board text format, and move
// sequence strings such as "f5d6" (concatenated 2-character moves, as
// used by WThor derived game records).

use crate::position::{Color, Position};

/// Parse a square name like "d3" (case-insensitive) into a bit index.
pub fn square_from_str(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    square_from_bytes(b[0], b[1])
}

fn square_from_bytes(file: u8, rank: u8) -> Option<u8> {
    let col = match file {
        b'a'..=b'h' => file - b'a',
        b'A'..=b'H' => file - b'A',
        _ => return None,
    };
    let row = match rank {
        b'1'..=b'8' => rank - b'1',
        _ => return None,
    };
    Some(row * 8 + col)
}

/// Format a bit index as a lowercase square name like "d3".
pub fn square_to_string(sq: u8) -> String {
    debug_assert!(sq < 64);
    let file = (b'a' + sq % 8) as char;
    let rank = (b'1' + sq / 8) as char;
    format!("{}{}", file, rank)
}

/// Parse a move sequence string such as "f5d6c3". Whitespace is ignored;
/// the remaining characters must form consecutive 2-character moves.
pub fn parse_move_seq(s: &str) -> Result<Vec<u8>, String> {
    let b: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    if !b.len().is_multiple_of(2) {
        return Err(format!("odd number of coordinate characters in {:?}", s));
    }
    let mut moves = Vec::with_capacity(b.len() / 2);
    for pair in b.chunks_exact(2) {
        match square_from_bytes(pair[0], pair[1]) {
            Some(sq) => moves.push(sq),
            None => {
                return Err(format!(
                    "invalid square {:?} in move sequence",
                    String::from_utf8_lossy(pair)
                ))
            }
        }
    }
    Ok(moves)
}

/// Format a move sequence as a concatenated lowercase string like "f5d6c3".
pub fn move_seq_to_string(moves: &[u8]) -> String {
    let mut s = String::with_capacity(moves.len() * 2);
    for &sq in moves {
        s.push_str(&square_to_string(sq));
    }
    s
}

/// Replay a move sequence from `pos`. A pass is inserted automatically
/// whenever the side to move has no legal move, so move strings never
/// contain explicit passes. Returns the final position.
pub fn replay(pos: Position, moves: &[u8]) -> Result<Position, String> {
    let mut p = pos;
    for (i, &sq) in moves.iter().enumerate() {
        if p.must_pass() {
            p = p.pass();
        }
        p = p.play(sq).map_err(|_| {
            format!(
                "illegal move {} at index {} of move sequence",
                square_to_string(sq),
                i
            )
        })?;
    }
    Ok(p)
}

impl Position {
    /// Serialize to the canonical text format: 8 lines of 8 characters
    /// (rank 1 first; 'X' = black, 'O' = white, '-' = empty) followed by
    /// one line holding the side to move ('X' or 'O').
    pub fn to_text(&self) -> String {
        let black = self.black();
        let white = self.white();
        let mut s = String::with_capacity(9 * 9);
        for row in 0..8u8 {
            for col in 0..8u8 {
                let sq = row * 8 + col;
                s.push(if black >> sq & 1 == 1 {
                    'X'
                } else if white >> sq & 1 == 1 {
                    'O'
                } else {
                    '-'
                });
            }
            s.push('\n');
        }
        s.push(match self.stm {
            Color::Black => 'X',
            Color::White => 'O',
        });
        s.push('\n');
        s
    }

    /// Parse the text format produced by `to_text`. All whitespace is
    /// ignored, so the 64 board characters may be laid out freely; the
    /// 65th non-whitespace character is the side to move.
    pub fn from_text(s: &str) -> Result<Position, String> {
        let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
        if chars.len() != 65 {
            return Err(format!(
                "expected 65 non-whitespace characters (64 squares + side to move), got {}",
                chars.len()
            ));
        }
        let mut black = 0u64;
        let mut white = 0u64;
        for (i, &c) in chars[..64].iter().enumerate() {
            match c {
                'X' | 'x' | '*' => black |= 1u64 << i,
                'O' | 'o' => white |= 1u64 << i,
                '-' | '.' => {}
                _ => return Err(format!("invalid board character {:?} at square {}", c, i)),
            }
        }
        let stm = match chars[64] {
            'X' | 'x' | '*' => Color::Black,
            'O' | 'o' => Color::White,
            c => return Err(format!("invalid side-to-move character {:?}", c)),
        };
        let (own, opp) = match stm {
            Color::Black => (black, white),
            Color::White => (white, black),
        };
        Ok(Position { own, opp, stm })
    }
}

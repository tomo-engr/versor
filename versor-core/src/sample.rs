// Seeded random positions for the tests, the examples and the
// measurement commands. Every position is reached by random play from
// the initial position, so it is legal and reachable, and the same seed
// always gives the same positions.

use crate::bitboard;
use crate::position::Position;
use crate::rng::Xorshift64Star;

/// Play `plies` random moves from the initial position. A forced pass is
/// played automatically before the move and does not count as a ply. If
/// the game ends on the way, or right after the last move, start again
/// from the initial position (the generator is not reset). Returns the
/// position and the moves that led to it.
pub fn random_opening(rng: &mut Xorshift64Star, plies: u32) -> (Position, Vec<u8>) {
    'retry: loop {
        let mut p = Position::initial();
        let mut moves = Vec::with_capacity(plies as usize);
        for _ in 0..plies {
            if p.is_game_over() {
                continue 'retry;
            }
            if p.must_pass() {
                p = p.pass();
            }
            let legal = p.legal_moves();
            let sq = bitboard::nth_set_bit(legal, rng.below(legal.count_ones()));
            moves.push(sq);
            p = p.play_unchecked(sq);
        }
        if !p.is_game_over() {
            return (p, moves);
        }
    }
}

/// The position of `random_opening` without the moves.
pub fn random_position(rng: &mut Xorshift64Star, plies: u32) -> Position {
    random_opening(rng, plies).0
}

/// Random play from the initial position until exactly `empties` squares
/// are empty; games that end first are discarded. The side to move may
/// have to pass in the returned position. `empties` must be at least 1:
/// a full board is a finished game, which is always discarded, so the
/// loop would never end.
pub fn random_position_with_empties(rng: &mut Xorshift64Star, empties: u32) -> Position {
    loop {
        let mut p = Position::initial();
        loop {
            if p.is_game_over() {
                break;
            }
            if p.empty().count_ones() == empties {
                return p;
            }
            if p.must_pass() {
                p = p.pass();
                continue;
            }
            let moves = p.legal_moves();
            let sq = bitboard::nth_set_bit(moves, rng.below(moves.count_ones()));
            p = p.play_unchecked(sq);
        }
    }
}

/// One walk of random play from the initial position to `empties` empty
/// squares, with the moves played (so the position can be replayed with
/// `versor-cli search <depth> <moves>`). None when the game ends first,
/// or when the side to move has to pass at the target.
pub fn walk_to_empties(rng: &mut Xorshift64Star, empties: u32) -> Option<(Position, Vec<u8>)> {
    let mut p = Position::initial();
    let mut played = Vec::new();
    loop {
        if p.empty().count_ones() == empties {
            if p.legal_moves() == 0 {
                return None;
            }
            return Some((p, played));
        }
        let moves = p.legal_moves();
        if moves == 0 {
            if p.is_game_over() {
                return None;
            }
            p = p.pass();
            continue;
        }
        let sq = bitboard::nth_set_bit(moves, rng.below(moves.count_ones()));
        played.push(sq);
        p = p.play_unchecked(sq);
    }
}

/// `n` positions with `empties` empty squares and a legal move for the
/// side to move, from successive `walk_to_empties` walks with one
/// generator seeded by `seed`. Gives up (returning fewer positions) after
/// ten million walks.
pub fn positions_at_empties(seed: u64, empties: u32, n: usize) -> Vec<(Position, Vec<u8>)> {
    let mut rng = Xorshift64Star::new(seed);
    let mut out = Vec::with_capacity(n);
    let mut guard = 0u64;
    while out.len() < n {
        guard += 1;
        if guard > 10_000_000 {
            break;
        }
        if let Some(p) = walk_to_empties(&mut rng, empties) {
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::replay;

    #[test]
    fn openings_replay_to_the_returned_position() {
        let mut rng = Xorshift64Star::new(5);
        for plies in [0u32, 1, 8, 20, 40] {
            for _ in 0..50 {
                let (p, moves) = random_opening(&mut rng, plies);
                assert_eq!(moves.len(), plies as usize);
                assert!(!p.is_game_over());
                assert_eq!(replay(Position::initial(), &moves).unwrap(), p);
            }
        }
    }

    #[test]
    fn positions_have_the_requested_empty_count() {
        let mut rng = Xorshift64Star::new(6);
        for empties in [8u32, 12, 17, 30] {
            for _ in 0..50 {
                assert_eq!(random_position_with_empties(&mut rng, empties).empty().count_ones(), empties);
            }
        }
        for empties in [10u32, 16, 34] {
            let set = positions_at_empties(7, empties, 40);
            assert_eq!(set.len(), 40);
            for (p, moves) in &set {
                assert_eq!(p.empty().count_ones(), empties);
                assert_ne!(p.legal_moves(), 0);
                assert_eq!(replay(Position::initial(), moves).unwrap(), *p);
            }
            assert_eq!(set, positions_at_empties(7, empties, 40), "same seed, same positions");
        }
    }
}

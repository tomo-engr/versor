// Transposition table: single-slot buckets with a depth-and-generation
// replacement policy. Entry values are on the disc difference scale and
// always fit in an i16.

pub const BOUND_NONE: u8 = 0;
pub const BOUND_EXACT: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_UPPER: u8 = 3;

/// Sentinel for "no best move stored".
pub const NO_MOVE: u8 = 64;

#[derive(Clone, Copy)]
pub struct Entry {
    pub key: u64,
    pub value: i16,
    pub depth: u8,
    pub bound: u8,
    pub mv: u8,
    pub gen: u8,
}

const EMPTY_ENTRY: Entry = Entry {
    key: 0,
    value: 0,
    depth: 0,
    bound: BOUND_NONE,
    mv: NO_MOVE,
    gen: 0,
};

pub struct Table {
    entries: Vec<Entry>,
    mask: usize,
    gen: u8,
}

impl Table {
    /// Create a table of approximately `mb` megabytes (rounded down to
    /// a power-of-two entry count, at least one entry).
    pub fn new(mb: usize) -> Table {
        let bytes = mb.max(1) << 20;
        let count = (bytes / std::mem::size_of::<Entry>()).max(2);
        let count = if count.is_power_of_two() {
            count
        } else {
            count.next_power_of_two() >> 1
        };
        Table {
            entries: vec![EMPTY_ENTRY; count],
            mask: count - 1,
            gen: 0,
        }
    }

    pub fn clear(&mut self) {
        self.entries.fill(EMPTY_ENTRY);
        self.gen = 0;
    }

    /// Start a new search: old entries become replaceable.
    pub fn new_generation(&mut self) {
        self.gen = self.gen.wrapping_add(1);
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        let e = self.entries[key as usize & self.mask];
        if e.bound != BOUND_NONE && e.key == key {
            Some(e)
        } else {
            None
        }
    }

    pub fn store(&mut self, key: u64, value: i32, depth: u32, bound: u8, mv: u8) {
        debug_assert!(value > i32::from(i16::MIN) && value < i32::from(i16::MAX));
        let e = &mut self.entries[key as usize & self.mask];
        let depth = depth.min(255) as u8;
        // Replace empty slots, the same position, entries from older
        // searches, and shallower entries.
        if e.bound == BOUND_NONE || e.key == key || e.gen != self.gen || depth >= e.depth {
            *e = Entry {
                key,
                value: value as i16,
                depth,
                bound,
                mv,
                gen: self.gen,
            };
        }
    }
}

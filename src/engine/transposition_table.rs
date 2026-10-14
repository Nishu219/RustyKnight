use chess::{ChessMove, Piece, Square};
use std::sync::atomic::{AtomicU64, Ordering};

#[inline(always)]
pub fn pack_move(mv: ChessMove) -> u16 {
    let source = mv.get_source().to_index() as u16;
    let dest = mv.get_dest().to_index() as u16;
    let promotion = match mv.get_promotion() {
        None => 0,
        Some(Piece::Knight) => 1,
        Some(Piece::Bishop) => 2,
        Some(Piece::Rook) => 3,
        Some(Piece::Queen) => 4,
        _ => 0,
    };
    source | (dest << 6) | (promotion << 12)
}

#[inline(always)]
pub fn unpack_move(packed: u16) -> ChessMove {
    let source = unsafe { Square::new((packed & 0x3F) as u8) };
    let dest = unsafe { Square::new(((packed >> 6) & 0x3F) as u8) };
    let promotion = match (packed >> 12) & 0x7 {
        1 => Some(Piece::Knight),
        2 => Some(Piece::Bishop),
        3 => Some(Piece::Rook),
        4 => Some(Piece::Queen),
        _ => None,
    };
    ChessMove::new(source, dest, promotion)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TTFlag {
    None = 0,
    Exact = 1,
    Lower = 2,
    Upper = 3,
}

impl TTFlag {
    #[inline(always)]
    pub fn from_u8(val: u8) -> Self {
        match val & 0x3 {
            1 => TTFlag::Exact,
            2 => TTFlag::Lower,
            3 => TTFlag::Upper,
            _ => TTFlag::None,
        }
    }
}

/// 16-byte lockless Transposition Table entry using Hyatt XOR data verification.
///
/// `data` layout (64 bits):
/// - bits  0..15 (16 bits): packed move
/// - bits 16..31 (16 bits): search score (i16)
/// - bits 32..47 (16 bits): static eval (i16)
/// - bits 48..55 ( 8 bits): depth (u8)
/// - bits 56..57 ( 2 bits): flag (TTFlag)
/// - bits 58..63 ( 6 bits): age (u8, 0..63)
///
/// `key_xor_data`: stores `key ^ data`.
/// On probe: if `(key_xor_data ^ data) == key`, the data is guaranteed consistent and belongs to `key`.
#[repr(C)]
pub struct TTEntry {
    pub key_xor_data: AtomicU64,
    pub data: AtomicU64,
}

impl Default for TTEntry {
    fn default() -> Self {
        Self {
            key_xor_data: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

#[inline(always)]
fn pack_data(mv: u16, score: i16, static_eval: i16, depth: u8, flag: TTFlag, age: u8) -> u64 {
    (mv as u64)
        | ((score as u16 as u64) << 16)
        | ((static_eval as u16 as u64) << 32)
        | ((depth as u64) << 48)
        | (((flag as u8 as u64) & 0x3) << 56)
        | (((age as u64) & 0x3F) << 58)
}

#[inline(always)]
fn unpack_data(data: u64) -> (u16, i16, i16, u8, TTFlag, u8) {
    let mv = (data & 0xFFFF) as u16;
    let score = ((data >> 16) & 0xFFFF) as i16;
    let static_eval = ((data >> 32) & 0xFFFF) as i16;
    let depth = ((data >> 48) & 0xFF) as u8;
    let flag = TTFlag::from_u8(((data >> 56) & 0x3) as u8);
    let age = ((data >> 58) & 0x3F) as u8;
    (mv, score, static_eval, depth, flag, age)
}

impl TTEntry {
    #[inline(always)]
    pub fn read(&self, key: u64) -> Option<(u16, i16, i16, u8, TTFlag, u8)> {
        let data = self.data.load(Ordering::Relaxed);
        let key_xor_data = self.key_xor_data.load(Ordering::Relaxed);
        if (key_xor_data ^ data) == key {
            let unpacked = unpack_data(data);
            if unpacked.4 != TTFlag::None {
                return Some(unpacked);
            }
        }
        None
    }

    #[inline(always)]
    pub fn read_any(&self) -> (u64, u16, i16, i16, u8, TTFlag, u8) {
        let data = self.data.load(Ordering::Relaxed);
        let key_xor_data = self.key_xor_data.load(Ordering::Relaxed);
        let key = key_xor_data ^ data;
        let (mv, score, static_eval, depth, flag, age) = unpack_data(data);
        (key, mv, score, static_eval, depth, flag, age)
    }

    #[inline(always)]
    pub fn write(&self, key: u64, mv: u16, score: i16, static_eval: i16, depth: u8, flag: TTFlag, age: u8) {
        let data = pack_data(mv, score, static_eval, depth, flag, age);
        self.data.store(data, Ordering::Relaxed);
        self.key_xor_data.store(key ^ data, Ordering::Relaxed);
    }
}

/// 4-entry bucket aligned to 64 bytes (exact CPU L1/L2 cache line size).
#[repr(C, align(64))]
pub struct TTBucket {
    pub entries: [TTEntry; 4],
}

impl Default for TTBucket {
    fn default() -> Self {
        Self {
            entries: [
                TTEntry::default(),
                TTEntry::default(),
                TTEntry::default(),
                TTEntry::default(),
            ],
        }
    }
}

/// Lemire Fastrange64: maps a 64-bit key to an index in [0, max) using a single 128-bit multiplication.
#[inline(always)]
pub fn fastrange64(val: u64, max: usize) -> usize {
    ((val as u128 * max as u128) >> 64) as usize
}

pub struct TranspositionTable {
    buckets: Vec<TTBucket>,
    num_buckets: usize,
    pub age: u8,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let bucket_size = std::mem::size_of::<TTBucket>(); // 64 bytes
        let num_buckets = (size_mb * 1024 * 1024) / bucket_size;
        let num_buckets = num_buckets.max(1);

        let mut buckets = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            buckets.push(TTBucket::default());
        }

        Self {
            buckets,
            num_buckets,
            age: 0,
        }
    }

    pub fn clear(&mut self) {
        for bucket in &self.buckets {
            for entry in &bucket.entries {
                entry.data.store(0, Ordering::Relaxed);
                entry.key_xor_data.store(0, Ordering::Relaxed);
            }
        }
        self.age = (self.age + 1) & 0x3F;
    }

    #[inline(always)]
    pub fn get(&self, key: u64, depth: usize, alpha: i32, beta: i32) -> Option<i32> {
        let (score, _) = self.get_with_move(key, depth, alpha, beta);
        score
    }

    /// Combined probe: returns (Option<score>, Option<best_move>).
    #[inline(always)]
    pub fn get_with_move(&self, key: u64, depth: usize, alpha: i32, beta: i32) -> (Option<i32>, Option<ChessMove>) {
        let bucket_idx = fastrange64(key, self.num_buckets);
        let bucket = &self.buckets[bucket_idx];

        let mut score: Option<i32> = None;
        let mut best_move: Option<ChessMove> = None;
        let mut best_move_depth: u8 = 0;

        for entry in &bucket.entries {
            if let Some((mv_packed, val, _static_eval, d, flag, _entry_age)) = entry.read(key) {
                if mv_packed != 0 && d >= best_move_depth {
                    best_move_depth = d;
                    best_move = Some(unpack_move(mv_packed));
                }

                if score.is_none() && (d as usize) >= depth {
                    let s = val as i32;
                    match flag {
                        TTFlag::Exact => score = Some(s),
                        TTFlag::Lower if s >= beta => score = Some(s),
                        TTFlag::Upper if s <= alpha => score = Some(s),
                        _ => {}
                    }
                }
            }
        }

        (score, best_move)
    }

    #[inline(always)]
    pub fn store(
        &self,
        key: u64,
        depth: usize,
        value: i32,
        flag: TTFlag,
        move_: Option<ChessMove>,
    ) {
        let bucket_idx = fastrange64(key, self.num_buckets);
        let bucket = &self.buckets[bucket_idx];

        let depth_u8 = depth.min(63) as u8;
        let value_i16 = value.clamp(-32000, 32000) as i16;
        let move_packed = move_.map(pack_move).unwrap_or(0);

        let mut replace_idx = 0;
        let mut replace_score = i32::MAX;

        for (i, entry) in bucket.entries.iter().enumerate() {
            let (entry_key, entry_mv, _val, _static_eval, entry_depth, entry_flag, entry_age) = entry.read_any();

            if entry_key == key && entry_flag != TTFlag::None {
                // Same position: preserve existing move if none provided in new store
                let final_mv = if move_packed != 0 { move_packed } else { entry_mv };
                entry.write(key, final_mv, value_i16, 0, depth_u8, flag, self.age);
                return;
            }

            let score = if entry_flag == TTFlag::None {
                -1000000
            } else {
                let age_diff = (self.age.wrapping_sub(entry_age) & 0x3F) as i32;
                let depth_diff = depth_u8 as i32 - entry_depth as i32;
                let type_bonus = if entry_flag == TTFlag::Exact { 2 } else { 0 };
                -age_diff * 4 - depth_diff * 2 + type_bonus
            };

            if score < replace_score {
                replace_score = score;
                replace_idx = i;
            }
        }

        bucket.entries[replace_idx].write(key, move_packed, value_i16, 0, depth_u8, flag, self.age);
    }

    #[inline(always)]
    pub fn get_move(&self, key: u64) -> Option<ChessMove> {
        let bucket_idx = fastrange64(key, self.num_buckets);
        let bucket = &self.buckets[bucket_idx];
        let mut best_move = None;
        let mut best_depth = 0;

        for entry in &bucket.entries {
            if let Some((mv_packed, _, _, d, _, _)) = entry.read(key) {
                if mv_packed != 0 && d >= best_depth {
                    best_depth = d;
                    best_move = Some(unpack_move(mv_packed));
                }
            }
        }
        best_move
    }

    pub fn hashfull(&self) -> usize {
        let sample_size = 1000.min(self.num_buckets);
        let mut filled = 0;

        for i in 0..sample_size {
            let bucket = &self.buckets[i];
            for entry in &bucket.entries {
                let (_, _, _, _, _, flag, age) = entry.read_any();
                if age == self.age && flag != TTFlag::None {
                    filled += 1;
                    break;
                }
            }
        }
        (filled * 1000) / sample_size
    }

    #[inline(always)]
    pub fn get_depth(&self, key: u64) -> Option<usize> {
        let bucket_idx = fastrange64(key, self.num_buckets);
        let bucket = &self.buckets[bucket_idx];

        let mut best_depth = 0;
        let mut found = false;

        for entry in &bucket.entries {
            if let Some((_, _, _, d, _, _)) = entry.read(key) {
                if (d as usize) > best_depth {
                    best_depth = d as usize;
                    found = true;
                }
            }
        }

        if found {
            Some(best_depth)
        } else {
            None
        }
    }
}

use chess::{ChessMove, Piece, Square};

// Pack ChessMove into 16 bits: source(6) + dest(6) + promotion(4)
#[inline(always)]
pub fn pack_move(mv: ChessMove) -> u32 {
    let source = mv.get_source().to_index() as u32;
    let dest = mv.get_dest().to_index() as u32;
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
pub fn unpack_move(packed: u32) -> ChessMove {
    let source = unsafe { Square::new((packed & 0x3F) as u8) };
    let dest = unsafe { Square::new(((packed >> 6) & 0x3F) as u8) };
    let promotion = match (packed >> 12) & 0xF {
        1 => Some(Piece::Knight),
        2 => Some(Piece::Bishop),
        3 => Some(Piece::Rook),
        4 => Some(Piece::Queen),
        _ => None,
    };
    ChessMove::new(source, dest, promotion)
}

#[derive(Clone, Copy, PartialEq)]
pub enum TTFlag {
    None,
    Exact,
    Lower,
    Upper,
}

// Optimized entry: 12 bytes (best achievable with alignment)
// Rust pads to 4-byte alignment, so 10 bytes becomes 12
// Still better than original 16 bytes (25% memory savings)
// Layout: key(4) + move(4) + value(2) + packed(2) + padding(2)
#[derive(Clone, Copy)]
pub struct TTEntry {
    key: u32,      // Upper 32 bits of zobrist hash
    move_: u32,    // Packed move
    value: i16,    // Evaluation score
    packed: u16,   // depth:6, flag:2, age:8
}

impl Default for TTEntry {
    fn default() -> Self {
        Self {
            key: 0,
            move_: 0,
            value: 0,
            packed: 0,
        }
    }
}

impl TTEntry {
    // Extract depth (0-63) from bits 0-5
    #[inline(always)]
    pub fn depth(&self) -> u8 {
        (self.packed & 0x3F) as u8
    }
    
    // Store depth, clamped to 6 bits (max 63)
    #[inline(always)]
    pub fn set_depth(&mut self, depth: u8) {
        let depth_clamped = depth.min(63);
        self.packed = (self.packed & 0xFFC0) | (depth_clamped as u16);
    }
    
    // Extract flag from bits 6-7
    #[inline(always)]
    pub fn flag(&self) -> TTFlag {
        match (self.packed >> 6) & 0x3 {
            0 => TTFlag::None,
            1 => TTFlag::Exact,
            2 => TTFlag::Lower,
            3 => TTFlag::Upper,
            _ => unreachable!(),
        }
    }
    
    #[inline(always)]
    pub fn set_flag(&mut self, flag: TTFlag) {
        let flag_bits = match flag {
            TTFlag::None => 0,
            TTFlag::Exact => 1,
            TTFlag::Lower => 2,
            TTFlag::Upper => 3,
        };
        self.packed = (self.packed & 0xFF3F) | ((flag_bits as u16) << 6);
    }
    
    // Extract age from bits 8-15
    #[inline(always)]
    pub fn age(&self) -> u8 {
        (self.packed >> 8) as u8
    }
    
    #[inline(always)]
    pub fn set_age(&mut self, age: u8) {
        self.packed = (self.packed & 0x00FF) | ((age as u16) << 8);
    }
}

// 4-way set associative bucket
#[derive(Clone)]
struct TTBucket {
    entries: [TTEntry; 4],
}

impl Default for TTBucket {
    fn default() -> Self {
        Self {
            entries: [TTEntry::default(); 4],
        }
    }
}

pub struct TranspositionTable {
    table: Vec<TTBucket>,
    pub hits: usize,
    age: u8,
    size: usize,
}

impl TranspositionTable {

    pub fn new(size_mb: usize) -> Self {
        let num_buckets = (size_mb * 1024 * 1024) / std::mem::size_of::<TTBucket>();
        
        Self {
            table: vec![TTBucket::default(); num_buckets],
            hits: 0,
            age: 0,
            size: num_buckets,
        }
    }

    // Clear table and increment age
    pub fn clear(&mut self) {
        self.table.fill(TTBucket::default());
        self.hits = 0;
        self.age = self.age.wrapping_add(1);
    }

    // Probe TT for stored evaluation
    // Returns Some(value) if entry is usable for cutoff, None otherwise
    #[inline(always)]
    pub fn get(&mut self, key: u64, depth: usize, alpha: i32, beta: i32) -> Option<i32> {
        let index = (key as usize) % self.size;
        let key32 = (key >> 32) as u32;
        let bucket = &self.table[index];
        
        for entry in &bucket.entries {
            if entry.key == key32 && entry.flag() != TTFlag::None {
                if entry.depth() as usize >= depth {
                    self.hits += 1;
                    let val = entry.value as i32;
                    
                    match entry.flag() {
                        TTFlag::Exact => return Some(val),
                        TTFlag::Lower if val >= beta => return Some(val),
                        TTFlag::Upper if val <= alpha => return Some(val),
                        _ => {}
                    }
                }
            }
        }
        None
    }

    // Store position evaluation and best move
    // Uses replacement scheme: same position > old entries > lower depth
    #[inline(always)]
    pub fn store(
        &mut self,
        key: u64,
        depth: usize,
        value: i32,
        flag: TTFlag,
        move_: Option<ChessMove>,
    ) {
        let index = (key as usize) % self.size;
        let key32 = (key >> 32) as u32;
        let bucket = &mut self.table[index];
        
        let depth_u8 = depth.min(63) as u8;
        let value_i16 = value.clamp(-32000, 32000) as i16;
        let move_packed = move_.map(|m| pack_move(m)).unwrap_or(0);
        
        // Find best slot: prioritize exact position match, then old/shallow entries
        let mut replace_idx = 0;
        let mut replace_score = i32::MAX;
        
        for (i, entry) in bucket.entries.iter().enumerate() {
            // Always update exact position match
            if entry.key == key32 {
                let mut updated_entry = *entry;
                updated_entry.set_depth(depth_u8);
                updated_entry.value = value_i16;
                updated_entry.set_flag(flag);
                if move_packed != 0 {
                    updated_entry.move_ = move_packed;
                }
                updated_entry.set_age(self.age);
                bucket.entries[i] = updated_entry;
                return;
            }
            
            // Calculate replacement score (lower = better to replace)
            let score = if entry.flag() == TTFlag::None {
                -1000000  // Empty slots are best to replace
            } else {
                let age_diff = self.age.wrapping_sub(entry.age()) as i32;
                let depth_diff = depth_u8 as i32 - entry.depth() as i32;
                
                // Prefer replacing: old entries, shallow entries, non-exact bounds
                let type_bonus = match entry.flag() {
                    TTFlag::Exact => 2,
                    _ => 0,
                };
                
                -age_diff * 4 - depth_diff * 2 + type_bonus
            };
            
            if score < replace_score {
                replace_score = score;
                replace_idx = i;
            }
        }
        
        // Create and store new entry
        let mut new_entry = TTEntry::default();
        new_entry.key = key32;
        new_entry.set_depth(depth_u8);
        new_entry.value = value_i16;
        new_entry.set_flag(flag);
        new_entry.move_ = move_packed;
        new_entry.set_age(self.age);
        bucket.entries[replace_idx] = new_entry;
    }

    // Retrieve best move for position (for move ordering)
    #[inline(always)]
    pub fn get_move(&self, key: u64) -> Option<ChessMove> {
        let index = (key as usize) % self.size;
        let key32 = (key >> 32) as u32;
        let bucket = &self.table[index];
        let mut best_move = None;
        let mut best_depth = 0;
        
        for entry in &bucket.entries {
            if entry.key == key32 && entry.move_ != 0 {
                if entry.depth() > best_depth {
                    best_depth = entry.depth();
                    best_move = Some(unpack_move(entry.move_));
                }
            }
        }
        best_move
    }
    
    // Calculate hash table fill rate (per mille: 0-1000)
    // Samples first 1000 buckets for efficiency
    pub fn hashfull(&self) -> usize {
        let sample_size = 1000.min(self.size);
        let mut filled = 0;
        
        for i in 0..sample_size {
            let bucket = &self.table[i];
            for entry in &bucket.entries {
                if entry.age() == self.age && entry.key != 0 {
                    filled += 1;
                    break;
                }
            }
        }
        (filled * 1000) / sample_size
    }

    // Get stored depth for singular extension checks
    #[inline(always)]
    pub fn get_depth(&self, key: u64) -> Option<usize> {
        let index = (key as usize) % self.size;
        let key32 = (key >> 32) as u32;
        let bucket = &self.table[index];
        
        let mut best_depth = 0;
        let mut found = false;
        
        for entry in &bucket.entries {
            if entry.key == key32 && entry.flag() != TTFlag::None {
                if entry.depth() as usize > best_depth {
                    best_depth = entry.depth() as usize;
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

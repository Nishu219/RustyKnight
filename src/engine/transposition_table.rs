use chess::{ChessMove, Piece, Square};
use std::mem;

// Helper functions to pack/unpack ChessMove into u32
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

#[derive(Clone, Copy)]
pub struct TTEntry {
    key: u32,      // Store only upper 32 bits (lower bits used for indexing)
    depth: u8,
    value: i16,
    flag: TTFlag,
    move_: u32,
    age: u8,
}

impl Default for TTEntry {
    fn default() -> Self {
        Self {
            key: 0,
            depth: 0,
            value: 0,
            flag: TTFlag::None,
            move_: 0,
            age: 0,
        }
    }
}

#[derive(Clone)]
struct TTBucket {
    entries: [TTEntry; 4],  // 4-bucket for better collision handling 
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
        let num_buckets = (size_mb * 1024 * 1024) / mem::size_of::<TTBucket>();
        
        Self {
            table: vec![TTBucket::default(); num_buckets],
            hits: 0,
            age: 0,
            size: num_buckets,
        }
    }

    pub fn clear(&mut self) {
        self.table.fill(TTBucket::default());
        self.hits = 0;
        self.age = self.age.wrapping_add(1);
    }

    #[inline(always)]
    pub fn get(&mut self, key: u64, depth: usize, alpha: i32, beta: i32) -> Option<i32> {
        let index = (key as usize) % self.size;
        let key32 = (key >> 32) as u32;  // Upper 32 bits for verification
        let bucket = &self.table[index];
        
        // Check all 4 entries in the bucket
        for entry in &bucket.entries {
            if entry.key == key32 && entry.flag != TTFlag::None {
                // Key matches - verify depth
                if entry.depth >= depth as u8 {
                    self.hits += 1;
                    let val = entry.value as i32;
                    
                    match entry.flag {
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
        
        let depth_u8 = depth.min(255) as u8;
        let value_i16 = value.clamp(-32000, 32000) as i16;
        let move_packed = move_.map(|m| pack_move(m)).unwrap_or(0);
        
        // Find best slot to replace 
        let mut replace_idx = 0;
        let mut replace_score = i32::MAX;
        
        for (i, entry) in bucket.entries.iter().enumerate() {
            // If we find the same position, always update it
            if entry.key == key32 {
                bucket.entries[i] = TTEntry {
                    key: key32,
                    depth: depth_u8,
                    value: value_i16,
                    flag,
                    move_: if move_packed != 0 { move_packed } else { entry.move_ }, // Preserve move if new one is empty
                    age: self.age,
                };
                return;
            }
            let score = if entry.flag == TTFlag::None {
                // Empty slot - use immediately
                -1000000
            } else {
                let age_diff = self.age.wrapping_sub(entry.age) as i32;
                let depth_diff = depth_u8 as i32 - entry.depth as i32;
                let age_weight = age_diff * 4;  
                let depth_weight = depth_diff * 2;  
                
                let type_bonus = match entry.flag {
                    TTFlag::Exact => 2, 
                    TTFlag::Lower | TTFlag::Upper => 0,  
                    TTFlag::None => -100,  
                };

                -age_weight - depth_weight + type_bonus
            };
            
            if score < replace_score {
                replace_score = score;
                replace_idx = i;
            }
        }
        bucket.entries[replace_idx] = TTEntry {
            key: key32,
            depth: depth_u8,
            value: value_i16,
            flag,
            move_: move_packed,
            age: self.age,
        };
    }

    #[inline(always)]
    pub fn get_move(&self, key: u64) -> Option<ChessMove> {
        let index = (key as usize) % self.size;
        let key32 = (key >> 32) as u32;
        let bucket = &self.table[index];
        let mut best_move = None;
        let mut best_depth = 0;
        
        for entry in &bucket.entries {
            if entry.key == key32 && entry.move_ != 0 {
                if entry.depth > best_depth {
                    best_depth = entry.depth;
                    best_move = Some(unpack_move(entry.move_));
                }
            }
        }
        best_move
    }
    
    pub fn hashfull(&self) -> usize {
        // Sample first 1000 buckets to estimate fill rate
        let sample_size = 1000.min(self.size);
        let mut filled = 0;
        
        for i in 0..sample_size {
            let bucket = &self.table[i];
            // Count entries from current search (age matches)
            for entry in &bucket.entries {
                if entry.age == self.age && entry.key != 0 {
                    filled += 1;
                    break; // Count each bucket only once
                }
            }
        }
        (filled * 1000) / sample_size
    }
}
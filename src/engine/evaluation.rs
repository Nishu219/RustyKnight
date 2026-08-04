use crate::engine::constants::*;
use crate::engine::move_ordering::PIECE_VALUES;
use chess::{BitBoard, Board, Color, MoveGen, Piece, Rank, File, Square};
use lazy_static::lazy_static;
use std::cell::RefCell;

thread_local! {
    pub static PAWN_HASH_TABLE: RefCell<PawnHashTable> = RefCell::new(PawnHashTable::new(16));
}

// Piece-Square Tables
// Indexed as [piece_index][phase_index][square_index]
// piece_index: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
// phase_index: 0=Middle Game, 1=End Game
// square_index: 0-63 (a1=0, h1=7, a8=56, h8=63)
pub static PST: [[[i32; 64]; 2]; 6] = [
    // Pawn (index 0)
    [
        // Middle game
        [
            0, 0, 0, 0, 0, 0, 0, 0,
            98, 134, 61, 95, 68, 126, 34, -11,
            -6, 7, 26, 31, 65, 56, 25, -20,
            -14, 13, 6, 21, 23, 12, 17, -23,
            -27, -2, -5, 12, 17, 6, 10, -25,
            -26, -4, -4, -10, 3, 3, 33, -12,
            -35, -1, -20, -23, -15, 24, 38, -22,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        // End game
        [
            0, 0, 0, 0, 0, 0, 0, 0,
            178, 173, 158, 134, 147, 132, 165, 187,
            94, 100, 85, 67, 56, 53, 82, 84,
            32, 24, 13, 5, -2, 4, 17, 17,
            13, 9, -3, -7, -7, -8, 3, -1,
            4, 7, -6, 1, 0, -5, -1, -8,
            13, 8, 8, 10, 13, 0, 2, -7,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
    ],
    // Knight (index 1)
    [
        // Middle game
        [
            -167, -89, -34, -49, 61, -97, -15, -107,
            -73, -41, 72, 36, 23, 62, 7, -17,
            -47, 60, 37, 65, 84, 129, 73, 44,
            -9, 17, 19, 53, 37, 69, 18, 22,
            -13, 4, 16, 13, 28, 19, 21, -8,
            -23, -9, 12, 10, 19, 17, 25, -16,
            -29, -53, -12, -3, -1, 18, -14, -19,
            -105, -21, -58, -33, -17, -28, -19, -23,
        ],
        // End game
        [
            -58, -38, -13, -28, -31, -27, -63, -99,
            -25, -8, -25, -2, -9, -25, -24, -52,
            -24, -20, 10, 9, -1, -9, -19, -41,
            -17, 3, 22, 22, 22, 11, 8, -18,
            -18, -6, 16, 25, 16, 17, 4, -18,
            -23, -3, -1, 15, 10, -3, -20, -22,
            -42, -20, -10, -5, -2, -20, -23, -44,
            -29, -51, -23, -15, -22, -18, -50, -64,
        ],
    ],
    // Bishop (index 2)
    [
        // Middle game
        [
            -29, 4, -82, -37, -25, -42, 7, -8,
            -26, 16, -18, -13, 30, 59, 18, -47,
            -16, 37, 43, 40, 35, 50, 37, -2,
            -4, 5, 19, 50, 37, 37, 7, -2,
            -6, 13, 13, 26, 34, 12, 10, 4,
            0, 15, 15, 15, 14, 27, 18, 10,
            4, 15, 16, 0, 7, 21, 33, 1,
            -33, -3, -14, -21, -13, -12, -39, -21,
        ],
        // End game
        [
            -14, -21, -11, -8, -7, -9, -17, -24,
            -8, -4, 7, -12, -3, -13, -4, -14,
            2, -8, 0, -1, -2, 6, 0, 4,
            -3, 9, 12, 9, 14, 10, 3, 2,
            -6, 3, 13, 19, 7, 10, -3, -9,
            -12, -3, 8, 10, 13, 3, -7, -15,
            -14, -18, -7, -1, 4, -9, -15, -27,
            -23, -9, -23, -5, -9, -16, -5, -17,
        ],
    ],
    // Rook (index 3)
    [
        // Middle game
        [
            32, 42, 32, 51, 63, 9, 31, 43,
            27, 32, 58, 62, 80, 67, 26, 44,
            -5, 19, 26, 36, 17, 45, 61, 16,
            -24, -11, 7, 26, 24, 35, -8, -20,
            -36, -26, -12, -1, 9, -7, 6, -23,
            -45, -25, -16, -17, 3, 0, -5, -33,
            -44, -16, -20, -9, -1, 11, -6, -71,
            -19, -13, 1, 17, 16, 7, -37, -26,
        ],
        // End game
        [
            13, 10, 18, 15, 12, 12, 8, 5,
            11, 13, 13, 11, -3, 3, 8, 3,
            7, 7, 7, 5, 4, -3, -5, -3,
            4, 3, 13, 1, 2, 1, -1, 2,
            3, 5, 8, 4, -5, -6, -8, -11,
            -4, 0, -5, -1, -7, -12, -8, -16,
            -6, -6, 0, 2, -9, -9, -11, -3,
            -9, 2, 3, -1, -5, -13, 4, -20,
        ],
    ],
    // Queen (index 4)
    [
        // Middle game
        [
            -28, 0, 29, 12, 59, 44, 43, 45,
            -24, -39, -5, 1, -16, 57, 28, 54,
            -13, -17, 7, 8, 29, 56, 47, 57,
            -27, -27, -16, -16, -1, 17, -2, 1,
            -9, -26, -9, -10, -2, -4, 3, -3,
            -14, 2, -11, -2, -5, 2, 14, 5,
            -35, -8, 11, 2, 8, 15, -3, 1,
            -18, -9, 10, -15, -25, -31, -50, 0,
        ],
        // End game
        [
            -9, 22, 22, 27, 27, 19, 10, 20,
            -17, 20, 32, 41, 58, 25, 30, 0,
            -20, 6, 9, 49, 47, 35, 19, 9,
            3, 22, 24, 45, 57, 40, 57, 36,
            -18, 28, 19, 47, 31, 34, 39, 23,
            -16, -27, 15, 6, 9, 17, 10, 5,
            -22, -23, -30, -16, -16, -23, -36, -32,
            -33, -28, -22, -43, -5, -32, -20, -41,
        ],
    ],
    // King (index 5)
    [
        // Middle game
        [
            -65, 23, 16, -15, -56, -34, 2, 13,
            29, -1, -20, -7, -8, -4, -38, -29,
            -9, 24, 2, -16, -20, 6, 22, -22,
            -17, -20, -12, -27, -30, -25, -14, -36,
            -49, -1, -27, -39, -46, -44, -33, -51,
            -14, -14, -22, -46, -44, -30, -15, -27,
            1, 7, -8, -64, -43, -16, 9, 8,
            -15, 36, 12, -54, 8, -28, 24, 14,
        ],
        // End game
        [
            -74, -35, -18, -18, -11, 15, 4, -17,
            -12, 17, 14, 17, 17, 38, 23, 11,
            10, 17, 23, 15, 20, 45, 44, 13,
            -8, 22, 24, 27, 26, 33, 26, 3,
            -18, -4, 21, 24, 27, 23, 9, -11,
            -19, -3, 11, 21, 23, 16, 7, -9,
            -27, -11, 4, 13, 14, 4, -5, -17,
            -53, -34, -21, -11, -28, -14, -24, -43,
        ],
    ],
];

// Pre-computed piece indices for faster access
pub const PAWN_INDEX: usize = 0;
pub const KNIGHT_INDEX: usize = 1;
pub const BISHOP_INDEX: usize = 2;
pub const ROOK_INDEX: usize = 3;
pub const QUEEN_INDEX: usize = 4;
pub const KING_INDEX: usize = 5;

// Helper function to convert Piece to array index
#[inline]
pub fn piece_to_index(piece: Piece) -> usize {
    match piece {
        Piece::Pawn => PAWN_INDEX,
        Piece::Knight => KNIGHT_INDEX,
        Piece::Bishop => BISHOP_INDEX,
        Piece::Rook => ROOK_INDEX,
        Piece::Queen => QUEEN_INDEX,
        Piece::King => KING_INDEX,
    }
}

lazy_static! {
    pub static ref FILE_MASKS: [BitBoard; 8] = [
        BitBoard::new(0x0101010101010101),
        BitBoard::new(0x0202020202020202),
        BitBoard::new(0x0404040404040404),
        BitBoard::new(0x0808080808080808),
        BitBoard::new(0x1010101010101010),
        BitBoard::new(0x2020202020202020),
        BitBoard::new(0x4040404040404040),
        BitBoard::new(0x8080808080808080),
    ];
    pub static ref ADJACENT_FILES: [BitBoard; 8] = [
        BitBoard::new(0x0202020202020202),
        BitBoard::new(0x0505050505050505),
        BitBoard::new(0x0A0A0A0A0A0A0A0A),
        BitBoard::new(0x1414141414141414),
        BitBoard::new(0x2828282828282828),
        BitBoard::new(0x5050505050505050),
        BitBoard::new(0xA0A0A0A0A0A0A0A0),
        BitBoard::new(0x4040404040404040),
    ];
    pub static ref WHITE_PASSED_MASKS: [BitBoard; 64] = {
        let mut masks = [BitBoard::new(0); 64];
        for sq in 0..64 {
            let square = unsafe { Square::new(sq as u8) };
            let file = square.get_file().to_index();
            let rank = square.get_rank().to_index();

            if rank < 7 {
                let mut mask = BitBoard::new(0);
                for r in (rank + 1)..8 {
                    mask |= BitBoard::new(1u64 << (r * 8 + file));
                    if file > 0 {
                        mask |= BitBoard::new(1u64 << (r * 8 + file - 1));
                    }
                    if file < 7 {
                        mask |= BitBoard::new(1u64 << (r * 8 + file + 1));
                    }
                }
                masks[sq as usize] = mask;
            }
        }
        masks
    };
    pub static ref BLACK_PASSED_MASKS: [BitBoard; 64] = {
        let mut masks = [BitBoard::new(0); 64];
        for sq in 0..64 {
            let square = unsafe { Square::new(sq as u8) };
            let file = square.get_file().to_index();
            let rank = square.get_rank().to_index();

            if rank > 0 {
                let mut mask = BitBoard::new(0);
                for r in 0..rank {
                    mask |= BitBoard::new(1u64 << (r * 8 + file));
                    if file > 0 {
                        mask |= BitBoard::new(1u64 << (r * 8 + file - 1));
                    }
                    if file < 7 {
                        mask |= BitBoard::new(1u64 << (r * 8 + file + 1));
                    }
                }
                masks[sq as usize] = mask;
            }
        }
        masks
    };
    pub static ref WHITE_PAWN_PUSHES: [BitBoard; 64] = {
        let mut pushes = [BitBoard::new(0); 64];
        for sq in 0..64 {
            let rank = sq / 8;
            if rank < 7 {
                pushes[sq] = BitBoard::new(1u64 << (sq + 8));
            }
        }
        pushes
    };
    
    pub static ref BLACK_PAWN_PUSHES: [BitBoard; 64] = {
        let mut pushes = [BitBoard::new(0); 64];
        for sq in 0..64 {
            let rank = sq / 8;
            if rank > 0 {
                pushes[sq] = BitBoard::new(1u64 << (sq - 8));
            }
        }
        pushes
    };
    
    pub static ref WHITE_PAWN_ATTACKS: [BitBoard; 64] = {
        let mut attacks = [BitBoard::new(0); 64];
        for sq in 0..64 {
            let file = sq % 8;
            let rank = sq / 8;
            let mut attack = BitBoard::new(0);
            
            if rank < 7 {
                if file > 0 {
                    attack |= BitBoard::new(1u64 << (sq + 7));
                }
                if file < 7 {
                    attack |= BitBoard::new(1u64 << (sq + 9));
                }
            }
            attacks[sq] = attack;
        }
        attacks
    };
    
    pub static ref BLACK_PAWN_ATTACKS: [BitBoard; 64] = {
        let mut attacks = [BitBoard::new(0); 64];
        for sq in 0..64 {
            let file = sq % 8;
            let rank = sq / 8;
            let mut attack = BitBoard::new(0);
            
            if rank > 0 {
                if file > 0 {
                    attack |= BitBoard::new(1u64 << (sq - 9));
                }
                if file < 7 {
                    attack |= BitBoard::new(1u64 << (sq - 7));
                }
            }
            attacks[sq] = attack;
        }
        attacks
    };
    
    pub static ref WHITE_PAWN_SUPPORT_SQUARES: [BitBoard; 64] = {
        let mut support = [BitBoard::new(0); 64];
        for sq in 0..64 {
            let file = sq % 8;
            let rank = sq / 8;
            let mut supporters = BitBoard::new(0);
            
            if rank > 0 {
                if file > 0 {
                    supporters |= BitBoard::new(1u64 << (sq - 9));
                }
                if file < 7 {
                    supporters |= BitBoard::new(1u64 << (sq - 7));
                }
            }
            support[sq] = supporters;
        }
        support
    };
    
    pub static ref BLACK_PAWN_SUPPORT_SQUARES: [BitBoard; 64] = {
        let mut support = [BitBoard::new(0); 64];
        for sq in 0..64 {
            let file = sq % 8;
            let rank = sq / 8;
            let mut supporters = BitBoard::new(0);
            
            if rank < 7 {
                if file > 0 {
                    supporters |= BitBoard::new(1u64 << (sq + 7));
                }
                if file < 7 {
                    supporters |= BitBoard::new(1u64 << (sq + 9));
                }
            }
            support[sq] = supporters;
        }
        support
    };
    
    pub static ref PHALANX_SQUARES: [BitBoard; 64] = {
        let mut phalanx = [BitBoard::new(0); 64];
        for sq in 0..64 {
            let file = sq % 8;
            let mut phalanx_bb = BitBoard::new(0);
            
            if file > 0 {
                phalanx_bb |= BitBoard::new(1u64 << (sq - 1));
            }
            if file < 7 {
                phalanx_bb |= BitBoard::new(1u64 << (sq + 1));
            }
            phalanx[sq] = phalanx_bb;
        }
        phalanx
    };

    pub static ref CHEBYSHEV_DISTANCE: [[i32; 64]; 64] = {
        let mut distances = [[0; 64]; 64];
        for sq1 in 0..64 {
            for sq2 in 0..64 {
                let file1 = sq1 % 8;
                let rank1 = sq1 / 8;
                let file2 = sq2 % 8;
                let rank2 = sq2 / 8;
                distances[sq1][sq2] = ((file1 as i32) - (file2 as i32)).abs().max(((rank1 as i32) - (rank2 as i32)).abs());
            }
        }
        distances
    };

    pub static ref KING_RING: [BitBoard; 64] = {
        let mut rings = [BitBoard::new(0); 64];
        for sq_idx in 0..64 {
            let file = (sq_idx % 8) as i32;
            let rank = (sq_idx / 8) as i32;
            let mut ring_bb = BitBoard::new(0);
            for file_offset in -1..=1 {
                for rank_offset in -1..=1 {
                    if file_offset == 0 && rank_offset == 0 { continue; }
                    let new_file = file + file_offset;
                    let new_rank = rank + rank_offset;
                    if new_file >= 0 && new_file < 8 && new_rank >= 0 && new_rank < 8 {
                        let new_sq_idx = (new_rank * 8 + new_file) as u8;
                        ring_bb |= BitBoard(1u64 << new_sq_idx);
                    }
                }
            }
            rings[sq_idx] = ring_bb;
        }
        rings
    };
}

pub struct CacheTable<T> {
    table: Vec<[(u64, T, usize); 2]>,
    age: usize,
    size: usize,
}

impl<T: Copy + Default> CacheTable<T> {
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<[(u64, T, usize); 2]>();
        let mut size = (size_mb * 1024 * 1024) / entry_size;
        size = 1 << (63 - (size as u64).leading_zeros());
        Self {
            table: vec![[(0, T::default(), 0); 2]; size],
            age: 0,
            size,
        }
    }
    
    pub fn lookup(&self, key: u64) -> Option<T> {
        let idx = key as usize & (self.size - 1);
        let bucket = &self.table[idx];
        if bucket[0].0 == key {
            return Some(bucket[0].1);
        }
        if bucket[1].0 == key {
            return Some(bucket[1].1);
        }
        None
    }
    
    pub fn store(&mut self, key: u64, value: T) {
        self.age += 1;
        let idx = key as usize & (self.size - 1);
        let bucket = &mut self.table[idx];
        if bucket[0].0 == key {
            bucket[0] = (key, value, self.age);
            return;
        }
        if bucket[1].0 == key {
            bucket[1] = (key, value, self.age);
            return;
        }
        if bucket[0].2 < bucket[1].2 {
            bucket[0] = (key, value, self.age);
        } else {
            bucket[1] = (key, value, self.age);
        }
    }
    
    pub fn clear(&mut self) {
        self.table.fill([(0, T::default(), 0); 2]);
        self.age = 0;
    }
}

pub type PawnHashTable = CacheTable<i32>;

pub fn compute_material_eval(board: &Board) -> i32 {
    let white = board.color_combined(Color::White);
    let black = board.color_combined(Color::Black);

    let mut material = 0;
    material += ((board.pieces(Piece::Pawn) & white).popcnt() as i32 - (board.pieces(Piece::Pawn) & black).popcnt() as i32) * PIECE_VALUES[0];
    material += ((board.pieces(Piece::Knight) & white).popcnt() as i32 - (board.pieces(Piece::Knight) & black).popcnt() as i32) * PIECE_VALUES[1];
    material += ((board.pieces(Piece::Bishop) & white).popcnt() as i32 - (board.pieces(Piece::Bishop) & black).popcnt() as i32) * PIECE_VALUES[2];
    material += ((board.pieces(Piece::Rook) & white).popcnt() as i32 - (board.pieces(Piece::Rook) & black).popcnt() as i32) * PIECE_VALUES[3];
    material += ((board.pieces(Piece::Queen) & white).popcnt() as i32 - (board.pieces(Piece::Queen) & black).popcnt() as i32) * PIECE_VALUES[4];
    material += ((board.pieces(Piece::King) & white).popcnt() as i32 - (board.pieces(Piece::King) & black).popcnt() as i32) * PIECE_VALUES[5];
    material
}

fn evaluate_rooks(board: &Board) -> i32 {
    let mut score = 0;

    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);

    let white_rooks = board.pieces(Piece::Rook) & board.color_combined(Color::White);
    let black_rooks = board.pieces(Piece::Rook) & board.color_combined(Color::Black);

    for sq in white_rooks {
        let file = sq.get_file();
        let file_mask = BitBoard::new(0x0101010101010101u64 << file.to_index());

        let white_pawns_on_file = (white_pawns & file_mask).popcnt();
        let black_pawns_on_file = (black_pawns & file_mask).popcnt();

        if white_pawns_on_file == 0 && black_pawns_on_file == 0 {
            score += 25;
        } else if white_pawns_on_file == 0 {
            score += 15;
        }

        if sq.get_rank() == Rank::Seventh {
            score += 20;
        }
    }

    for sq in black_rooks {
        let file = sq.get_file();
        let file_mask = BitBoard::new(0x0101010101010101u64 << file.to_index());

        let white_pawns_on_file = (white_pawns & file_mask).popcnt();
        let black_pawns_on_file = (black_pawns & file_mask).popcnt();

        if white_pawns_on_file == 0 && black_pawns_on_file == 0 {
            score -= 25;
        } else if black_pawns_on_file == 0 {
            score -= 15;
        }

        if sq.get_rank() == Rank::Second {
            score -= 20;
        }
    }

    score
}
fn evaluate_pawns(board: &Board, pawn_hash: u64, white_pawn_attacks: BitBoard, black_pawn_attacks: BitBoard) -> i32 {
    let cached_score = PAWN_HASH_TABLE.with(|cache| {
        cache.borrow().lookup(pawn_hash)
    });
    if let Some(score) = cached_score {
        return score;
    }

    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);

    let white_king_sq = (board.pieces(Piece::King) & board.color_combined(Color::White))
        .into_iter()
        .next()
        .unwrap();
    let black_king_sq = (board.pieces(Piece::King) & board.color_combined(Color::Black))
        .into_iter()
        .next()
        .unwrap();
    
    let white_king_idx = white_king_sq.to_index();
    let black_king_idx = black_king_sq.to_index();

    let mut score = 0;
    
    let not_a_file = BitBoard::new(0xFEFEFEFEFEFEFEFE);
    let not_h_file = BitBoard::new(0x7F7F7F7F7F7F7F7F);
    
    let white_pawn_support = BitBoard::new(((white_pawns & not_a_file).0 >> 9) | 
                                           ((white_pawns & not_h_file).0 >> 7));
    let black_pawn_support = BitBoard::new(((black_pawns & not_a_file).0 << 7) | 
                                           ((black_pawns & not_h_file).0 << 9));

    let mut white_file_counts = [0u8; 8];
    let mut black_file_counts = [0u8; 8];

    for square in white_pawns {
        white_file_counts[square.get_file().to_index()] += 1;
    }
    for square in black_pawns {
        black_file_counts[square.get_file().to_index()] += 1;
    }

    // Doubled and isolated pawn penalties
    for file_idx in 0..8 {
        let w_count = white_file_counts[file_idx];
        let b_count = black_file_counts[file_idx];
        
        if w_count >= 2 {
            score -= 15 * (w_count - 1) as i32;
        }
        if b_count >= 2 {
            score += 15 * (b_count - 1) as i32;
        }
        
        if w_count > 0 {
            let has_adjacent = (file_idx > 0 && white_file_counts[file_idx - 1] > 0) ||
                               (file_idx < 7 && white_file_counts[file_idx + 1] > 0);
            if !has_adjacent {
                score -= 12;
            }
        }
        if b_count > 0 {
            let has_adjacent = (file_idx > 0 && black_file_counts[file_idx - 1] > 0) ||
                               (file_idx < 7 && black_file_counts[file_idx + 1] > 0);
            if !has_adjacent {
                score += 12;
            }
        }
    }

    // White pawns
    for square in white_pawns {
        let sq_idx = square.to_index();
        let rank = square.get_rank().to_index();
        let sq_bb = BitBoard::from_square(square);

        // Passed pawn
        if (WHITE_PASSED_MASKS[sq_idx] & black_pawns).0 == 0 {
            score += [0, 5, 10, 20, 35, 55, 80, 0][rank];
            
            let friendly_dist = CHEBYSHEV_DISTANCE[white_king_idx][sq_idx];
            let enemy_dist = CHEBYSHEV_DISTANCE[black_king_idx][sq_idx];
            let rank_factor = [0, 1, 2, 3, 4, 5, 6, 0][rank];
            
            score += (enemy_dist - 2) * rank_factor;
            score -= (friendly_dist - 1) * rank_factor;
        }
        
        // Phalanx pawn
        if (PHALANX_SQUARES[sq_idx] & white_pawns).0 != 0 {
            score += WHITE_PHALANX_BONUS[rank];
        }
        
        // Connected / Backward pawn
        if (WHITE_PAWN_SUPPORT_SQUARES[sq_idx] & white_pawns).0 != 0 {
            score += 3;
        } else {
            if (sq_bb & white_pawn_support).0 == 0 {
                let push_sq = WHITE_PAWN_PUSHES[sq_idx];
                if push_sq.0 != 0 {
                    if (push_sq & (black_pawns | black_pawn_attacks)).0 != 0 {
                        score -= 8;
                    }
                }
            }
        }
    }

    // Black pawns
    for square in black_pawns {
        let sq_idx = square.to_index();
        let rank = square.get_rank().to_index();
        let sq_bb = BitBoard::from_square(square);

        // Passed pawn
        if (BLACK_PASSED_MASKS[sq_idx] & white_pawns).0 == 0 {
            score -= [0, 80, 55, 35, 20, 10, 5, 0][rank];
            
            let friendly_dist = CHEBYSHEV_DISTANCE[black_king_idx][sq_idx];
            let enemy_dist = CHEBYSHEV_DISTANCE[white_king_idx][sq_idx];
            let rank_factor = [0, 6, 5, 4, 3, 2, 1, 0][rank];
            
            score -= (enemy_dist - 2) * rank_factor;
            score += (friendly_dist - 1) * rank_factor;
        }
        
        // Phalanx pawn
        if (PHALANX_SQUARES[sq_idx] & black_pawns).0 != 0 {
            score -= BLACK_PHALANX_BONUS[rank];
        }
        
        // Connected / Backward pawn
        if (BLACK_PAWN_SUPPORT_SQUARES[sq_idx] & black_pawns).0 != 0 {
            score -= 3;
        } else {
            if (sq_bb & black_pawn_support).0 == 0 {
                let push_sq = BLACK_PAWN_PUSHES[sq_idx];
                if push_sq.0 != 0 {
                    if (push_sq & (white_pawns | white_pawn_attacks)).0 != 0 {
                        score += 8;
                    }
                }
            }
        }
    }

    PAWN_HASH_TABLE.with(|cache| {
        cache.borrow_mut().store(pawn_hash, score);
    });

    score
}
fn compute_tropism_score(
    piece_bb: BitBoard,
    target_file: i32,
    target_rank: i32,
    piece_idx: usize,
) -> (i32, i32) {
    let (mg_weight, eg_weight) = TROPISM_WEIGHTS[piece_idx];
    let mut mg_total = 0;
    let mut eg_total = 0;

    for square in piece_bb {
        let file_dist = (square.get_file().to_index() as i32 - target_file).abs();
        let rank_dist = (square.get_rank().to_index() as i32 - target_rank).abs();

        let distance = if piece_idx == 3 {
            (file_dist + rank_dist + file_dist.max(rank_dist)) / 2
        } else {
            file_dist.max(rank_dist)
        };

        let max_dist = if piece_idx == 3 { 5 } else { 7 };
        let tropism = (if piece_idx == 3 { 6 } else { 8 }) - distance.min(max_dist);

        mg_total += tropism * mg_weight;
        eg_total += tropism * eg_weight
            / if piece_idx == 2 {
                2
            } else {
                if piece_idx == 3 {
                    3
                } else {
                    1
                }
            };
    }

    (mg_total, eg_total)
}

fn evaluate_king_tropism(board: &Board, phase: i32) -> i32 {
    let white_king_sq = (board.pieces(Piece::King) & board.color_combined(Color::White))
        .into_iter()
        .next()
        .unwrap();
    let black_king_sq = (board.pieces(Piece::King) & board.color_combined(Color::Black))
        .into_iter()
        .next()
        .unwrap();

    let white_king_file = white_king_sq.get_file().to_index() as i32;
    let white_king_rank = white_king_sq.get_rank().to_index() as i32;
    let black_king_file = black_king_sq.get_file().to_index() as i32;
    let black_king_rank = black_king_sq.get_rank().to_index() as i32;

    let mut mg_score = 0;
    let mut eg_score = 0;

    for (i, &piece_type) in PIECE_ORDER.iter().enumerate().take(4) {
        let white_pieces = board.pieces(piece_type) & board.color_combined(Color::White);
        let black_pieces = board.pieces(piece_type) & board.color_combined(Color::Black);

        let (white_mg, white_eg) =
            compute_tropism_score(white_pieces, black_king_file, black_king_rank, i);
        let (black_mg, black_eg) =
            compute_tropism_score(black_pieces, white_king_file, white_king_rank, i);

        mg_score += white_mg - black_mg;
        eg_score += white_eg - black_eg;
    }

    ((mg_score * phase) + (eg_score * (24 - phase))) / 24
}
#[inline(always)]
fn get_king_ring(king_sq: Square) -> BitBoard {
    KING_RING[king_sq.to_index()]
}

fn evaluate_pieces_and_king_safety(
    board: &Board,
    phase: i32,
    white_pawn_attacks: BitBoard,
    black_pawn_attacks: BitBoard
) -> i32 {
    let mut score_mg = 0;
    let mut score_eg = 0;

    let white_pieces = board.color_combined(Color::White);
    let black_pieces = board.color_combined(Color::Black);
    let occupied = *board.combined();

    let white_king_sq = (board.pieces(Piece::King) & white_pieces).into_iter().next().unwrap();
    let black_king_sq = (board.pieces(Piece::King) & black_pieces).into_iter().next().unwrap();
    
    let white_king_ring = get_king_ring(white_king_sq);
    let black_king_ring = get_king_ring(black_king_sq);

    // White and Black Knights
    let (k_atk_mg, k_atk_eg) = ATTACK_WEIGHTS[3];
    for sq in board.pieces(Piece::Knight) & white_pieces {
        let attacks = chess::get_knight_moves(sq);
        let safe = attacks & !white_pieces & !black_pawn_attacks;
        let mob = safe.popcnt() as usize;
        score_mg += KNIGHT_MOBILITY_MG[mob.min(8)];
        score_eg += KNIGHT_MOBILITY_EG[mob.min(8)];
        let ring_hits = (attacks & black_king_ring).popcnt() as i32;
        score_mg += ring_hits * k_atk_mg;
        score_eg += ring_hits * k_atk_eg;
    }
    for sq in board.pieces(Piece::Knight) & black_pieces {
        let attacks = chess::get_knight_moves(sq);
        let safe = attacks & !black_pieces & !white_pawn_attacks;
        let mob = safe.popcnt() as usize;
        score_mg -= KNIGHT_MOBILITY_MG[mob.min(8)];
        score_eg -= KNIGHT_MOBILITY_EG[mob.min(8)];
        let ring_hits = (attacks & white_king_ring).popcnt() as i32;
        score_mg -= ring_hits * k_atk_mg;
        score_eg -= ring_hits * k_atk_eg;
    }

    // White and Black Bishops
    let (b_atk_mg, b_atk_eg) = ATTACK_WEIGHTS[2];
    for sq in board.pieces(Piece::Bishop) & white_pieces {
        let attacks = chess::get_bishop_moves(sq, occupied);
        let safe = attacks & !white_pieces & !black_pawn_attacks;
        let mob = safe.popcnt() as usize;
        score_mg += BISHOP_MOBILITY_MG[mob.min(13)];
        score_eg += BISHOP_MOBILITY_EG[mob.min(13)];
        
        let ring_hits = (attacks & black_king_ring).popcnt() as i32;
        score_mg += ring_hits * b_atk_mg;
        score_eg += ring_hits * b_atk_eg;
    }
    for sq in board.pieces(Piece::Bishop) & black_pieces {
        let attacks = chess::get_bishop_moves(sq, occupied);
        let safe = attacks & !black_pieces & !white_pawn_attacks;
        let mob = safe.popcnt() as usize;
        score_mg -= BISHOP_MOBILITY_MG[mob.min(13)];
        score_eg -= BISHOP_MOBILITY_EG[mob.min(13)];
        
        let ring_hits = (attacks & white_king_ring).popcnt() as i32;
        score_mg -= ring_hits * b_atk_mg;
        score_eg -= ring_hits * b_atk_eg;
    }

    // White and Black Rooks
    let (r_atk_mg, r_atk_eg) = ATTACK_WEIGHTS[1];
    for sq in board.pieces(Piece::Rook) & white_pieces {
        let attacks = chess::get_rook_moves(sq, occupied);
        let safe = attacks & !white_pieces & !black_pawn_attacks;
        let mob = safe.popcnt() as usize;
        score_mg += ROOK_MOBILITY_MG[mob.min(14)];
        score_eg += ROOK_MOBILITY_EG[mob.min(14)];
        
        let ring_hits = (attacks & black_king_ring).popcnt() as i32;
        score_mg += ring_hits * r_atk_mg;
        score_eg += ring_hits * r_atk_eg;
    }
    for sq in board.pieces(Piece::Rook) & black_pieces {
        let attacks = chess::get_rook_moves(sq, occupied);
        let safe = attacks & !black_pieces & !white_pawn_attacks;
        let mob = safe.popcnt() as usize;
        score_mg -= ROOK_MOBILITY_MG[mob.min(14)];
        score_eg -= ROOK_MOBILITY_EG[mob.min(14)];
        
        let ring_hits = (attacks & white_king_ring).popcnt() as i32;
        score_mg -= ring_hits * r_atk_mg;
        score_eg -= ring_hits * r_atk_eg;
    }

    // White and Black Queens
    let (q_atk_mg, q_atk_eg) = ATTACK_WEIGHTS[0];
    for sq in board.pieces(Piece::Queen) & white_pieces {
        let attacks = chess::get_bishop_moves(sq, occupied) | chess::get_rook_moves(sq, occupied);
        let safe = attacks & !white_pieces & !black_pawn_attacks;
        let mob = safe.popcnt() as usize;
        score_mg += QUEEN_MOBILITY_MG[mob.min(27)];
        score_eg += QUEEN_MOBILITY_EG[mob.min(27)];
        
        let ring_hits = (attacks & black_king_ring).popcnt() as i32;
        score_mg += ring_hits * q_atk_mg;
        score_eg += ring_hits * q_atk_eg;
    }
    for sq in board.pieces(Piece::Queen) & black_pieces {
        let attacks = chess::get_bishop_moves(sq, occupied) | chess::get_rook_moves(sq, occupied);
        let safe = attacks & !black_pieces & !white_pawn_attacks;
        let mob = safe.popcnt() as usize;
        score_mg -= QUEEN_MOBILITY_MG[mob.min(27)];
        score_eg -= QUEEN_MOBILITY_EG[mob.min(27)];
        
        let ring_hits = (attacks & white_king_ring).popcnt() as i32;
        score_mg -= ring_hits * q_atk_mg;
        score_eg -= ring_hits * q_atk_eg;
    }

    // Pawn attacks on the King Ring
    let (p_atk_mg, p_atk_eg) = ATTACK_WEIGHTS[4];
    
    let white_pawn_ring_hits = (white_pawn_attacks & black_king_ring).popcnt() as i32;
    score_mg += white_pawn_ring_hits * p_atk_mg;
    score_eg += white_pawn_ring_hits * p_atk_eg;
    
    let black_pawn_ring_hits = (black_pawn_attacks & white_king_ring).popcnt() as i32;
    score_mg -= black_pawn_ring_hits * p_atk_mg;
    score_eg -= black_pawn_ring_hits * p_atk_eg;

    ((score_mg * phase) + (score_eg * (24 - phase))) / 24
}
fn evaluate_space(board: &Board, phase: i32) -> i32 {
    if phase < 8 {
        return 0;
    }
    
    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
    let space_mask = BitBoard::new(0x00003C3C3C3C0000);
    
    // Calculate pawn control using bitboard shifts
    let white_controlled = {
        let not_a_file = BitBoard::new(0xFEFEFEFEFEFEFEFE);
        let not_h_file = BitBoard::new(0x7F7F7F7F7F7F7F7F);
        let left_attacks = BitBoard::new((white_pawns & not_a_file).0 << 7);
        let right_attacks = BitBoard::new((white_pawns & not_h_file).0 << 9);
        left_attacks | right_attacks
    };
    
    let black_controlled = {
        let not_a_file = BitBoard::new(0xFEFEFEFEFEFEFEFE);
        let not_h_file = BitBoard::new(0x7F7F7F7F7F7F7F7F);
        let left_attacks = BitBoard::new((black_pawns & not_a_file).0 >> 9);
        let right_attacks = BitBoard::new((black_pawns & not_h_file).0 >> 7);
        left_attacks | right_attacks
    };
    
    // Calculate squares behind pawns using a fill algorithm for better performance
    let white_behind = {
        let mut behind = white_pawns.0;
        behind |= behind >> 8;
        behind |= behind >> 16;
        behind |= behind >> 32;
        BitBoard(behind >> 8)
    };

    let black_behind = {
        let mut behind = black_pawns.0;
        behind |= behind << 8;
        behind |= behind << 16;
        behind |= behind << 32;
        BitBoard(behind << 8)
    };
    
    // Calculate space with bonus for controlled squares behind pawns
    let white_space_mask = space_mask & !black_controlled;
    let black_space_mask = space_mask & !white_controlled;
    
    let white_space = (white_controlled & white_space_mask).popcnt() as i32
        + (white_behind & white_controlled & white_space_mask).popcnt() as i32;
    
    let black_space = (black_controlled & black_space_mask).popcnt() as i32
        + (black_behind & black_controlled & black_space_mask).popcnt() as i32;
    
    ((white_space - black_space) * phase * 4) / 24
}

fn evaluate_king_pawn_shield(board: &Board, phase: i32) -> i32 {
    let mut score = 0;

    let white_king_sq = (board.pieces(Piece::King) & board.color_combined(Color::White))
        .into_iter()
        .next()
        .unwrap();
    let black_king_sq = (board.pieces(Piece::King) & board.color_combined(Color::Black))
        .into_iter()
        .next()
        .unwrap();

    if phase < 12 {
        return 0;
    }
    if white_king_sq.get_file() == File::E {
        score -= 15;
    }
    if black_king_sq.get_file() == File::E {
        score += 15;
    }

    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);

    score += get_king_shield_score(white_king_sq, white_pawns, Color::White);
    score -= get_king_shield_score(black_king_sq, black_pawns, Color::Black);

    (score * phase) / 24
}

fn get_king_shield_score(king_sq: Square, friendly_pawns: BitBoard, color: Color) -> i32 {
    let mut score = 0;
    let king_file = king_sq.get_file().to_index();
    let king_rank = king_sq.get_rank().to_index();

    let (expected_king_rank, shield_rank) = if color == Color::White {
        (0, 1)
    } else {
        (7, 6)
    };

    if king_rank != expected_king_rank {
        return 0;
    }

    let shield_files: &[usize] = if king_file <= 2 {
        &[0, 1, 2]
    } else if king_file >= 5 {
        &[5, 6, 7]
    } else {
        return 0;
    };

    for &file in shield_files {
        let file_mask = FILE_MASKS[file];
        let pawns_on_file = friendly_pawns & file_mask;

        if pawns_on_file == BitBoard::new(0) {
            score -= 30;
        } else {
            let pawn_rank = if color == Color::White {
                pawns_on_file
                    .into_iter()
                    .map(|s| s.get_rank().to_index())
                    .min()
                    .unwrap_or(7)
            } else {
                pawns_on_file
                    .into_iter()
                    .map(|s| s.get_rank().to_index())
                    .max()
                    .unwrap_or(0)
            };

            let rank_diff = (pawn_rank as i32 - shield_rank as i32).abs();
            if rank_diff > 0 {
                score -= 15 * rank_diff;
            }
            if pawns_on_file.popcnt() > 1 {
                score -= 20;
            }
        }
    }
    score
}
fn evaluate_bad_bishops(board: &Board, phase: i32) -> i32 {
    if phase < 8 {
        return 0;
    }
    const DARK_SQUARES: BitBoard = BitBoard(0xAA55AA55AA55AA55);

    let mut score = 0;

    let white_bishops = board.pieces(Piece::Bishop) & board.color_combined(Color::White);
    if white_bishops != BitBoard::new(0) {
        let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
        let white_pawns_on_dark = (white_pawns & DARK_SQUARES).popcnt();
        let white_pawns_on_light = white_pawns.popcnt() - white_pawns_on_dark;

        for bishop_sq in white_bishops {
            if (BitBoard::from_square(bishop_sq) & DARK_SQUARES) != BitBoard::new(0) {
                if white_pawns_on_dark > 3 {
                    score -= 10;
                }
            } else {
                if white_pawns_on_light > 3 {
                    score -= 10;
                }
            }
        }
    }

    let black_bishops = board.pieces(Piece::Bishop) & board.color_combined(Color::Black);
    if black_bishops != BitBoard::new(0) {
        let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
        let black_pawns_on_dark = (black_pawns & DARK_SQUARES).popcnt();
        let black_pawns_on_light = black_pawns.popcnt() - black_pawns_on_dark;

        for bishop_sq in black_bishops {
            if (BitBoard::from_square(bishop_sq) & DARK_SQUARES) != BitBoard::new(0) {
                if black_pawns_on_dark > 3 {
                    score += 10;
                }
            } else {
                if black_pawns_on_light > 3 {
                    score += 10;
                }
            }
        }
    }

    (score * phase) / 24
}
fn evaluate_pawn_threats(
    board: &Board,
    white_pawn_attacks: BitBoard,
    black_pawn_attacks: BitBoard,
    phase: i32,
) -> i32 {
    let mut score = 0;
    
    let black_pieces = board.color_combined(Color::Black);
    
    for piece_idx in 0..PAWN_ATTACK_THREAT.len() {
        let piece = match piece_idx {
            0 => Piece::Knight,
            1 => Piece::Bishop,
            2 => Piece::Rook,
            3 => Piece::Queen,
            _ => Piece::Pawn,
        };
        let black_piece_bb = board.pieces(piece) & black_pieces;
        let attacked_pieces = black_piece_bb & white_pawn_attacks;
        
        if attacked_pieces != BitBoard::new(0) {
            let count = attacked_pieces.popcnt() as i32;
            let (mg_penalty, eg_penalty) = PAWN_ATTACK_THREAT[piece_idx];
            score += count * ((mg_penalty * phase) + (eg_penalty * (24 - phase))) / 24;
        }
    }
    
    let white_pieces = board.color_combined(Color::White);
    
    for piece_idx in 0..PAWN_ATTACK_THREAT.len() {
        let piece = match piece_idx {
            0 => Piece::Knight,
            1 => Piece::Bishop,
            2 => Piece::Rook,
            3 => Piece::Queen,
            _ => Piece::Pawn,
        };
        
        let white_piece_bb = board.pieces(piece) & white_pieces;
        let attacked_pieces = white_piece_bb & black_pawn_attacks;
        
        if attacked_pieces != BitBoard::new(0) {
            let count = attacked_pieces.popcnt() as i32;
            let (mg_penalty, eg_penalty) = PAWN_ATTACK_THREAT[piece_idx];
            score -= count * ((mg_penalty * phase) + (eg_penalty * (24 - phase))) / 24;
        }
    }
    
    score
}
pub fn evaluate(board: &Board, material: i32, pawn_hash: u64, contempt: i32, beta: Option<i32>) -> i32 {
    let in_check = *board.checkers() != BitBoard(0);
    let has_legal_moves = MoveGen::new_legal(board).next().is_some();

    if !has_legal_moves {
        return if in_check {
            if board.side_to_move() == Color::White { -30000 } else { 30000 }
        } else {
            if board.side_to_move() == Color::White { contempt } else { -contempt }
        };
    }
    let mut score = material;
    let mut phase = 0;
    phase += (board.pieces(Piece::Knight).popcnt() + board.pieces(Piece::Bishop).popcnt()) as i32;
    phase += (board.pieces(Piece::Rook).popcnt() * 2) as i32;
    phase += (board.pieces(Piece::Queen).popcnt() * 4) as i32;
    phase = phase.min(24);

    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
    
    // PST Evaluation
    let white_pieces = board.color_combined(Color::White);
    let black_pieces = board.color_combined(Color::Black);

    for piece in chess::ALL_PIECES {
        let piece_idx = piece_to_index(piece);
        
        // White pieces
        for square in board.pieces(piece) & white_pieces {
            let sq_idx = square.to_index();
            let pst_index = sq_idx ^ 56;
            let mg_pst = PST[piece_idx][0][pst_index];
            let eg_pst = PST[piece_idx][1][pst_index];
            score += ((mg_pst * phase) + (eg_pst * (24 - phase))) / 24;

            // Bonus for minor pieces with a pawn directly in front and penalty for undefended knight
            if (piece == Piece::Knight || piece == Piece::Bishop) && square.get_rank().to_index() < 7 {
                let ahead_sq = unsafe { Square::new((sq_idx + 8) as u8) };
                if (white_pawns & BitBoard::from_square(ahead_sq)) != BitBoard::new(0) {
                    score += 15;
                }
                let defended = (WHITE_PAWN_SUPPORT_SQUARES[sq_idx] & white_pawns) != BitBoard::new(0);
                if !defended && piece == Piece::Knight {
                    score -= ((13 * phase) + (8 * (24 - phase))) / 24;
                }
            }
        }
        
        // Black pieces
        for square in board.pieces(piece) & black_pieces {
            let sq_idx = square.to_index();
            let pst_index = sq_idx;
            let mg_pst = PST[piece_idx][0][pst_index];
            let eg_pst = PST[piece_idx][1][pst_index];
            score -= ((mg_pst * phase) + (eg_pst * (24 - phase))) / 24;

            if (piece == Piece::Knight || piece == Piece::Bishop) && square.get_rank().to_index() > 0 {
                let ahead_sq = unsafe { Square::new((sq_idx - 8) as u8) };
                if (black_pawns & BitBoard::from_square(ahead_sq)) != BitBoard::new(0) {
                    score -= 15;
                }
                let defended = (BLACK_PAWN_SUPPORT_SQUARES[sq_idx] & black_pawns) != BitBoard::new(0);
                if !defended && piece == Piece::Knight {
                    score += ((13 * phase) + (8 * (24 - phase))) / 24;
                }
            }
        }
    }

    let current_lazy_score = if board.side_to_move() == Color::White { score } else { -score };
    
    if let Some(beta_val) = beta {
        if current_lazy_score - LAZY_EVAL_MARGIN >= beta_val {
            return current_lazy_score;
        }
    }
    
    let not_a_file = BitBoard::new(0xFEFEFEFEFEFEFEFE);
    let not_h_file = BitBoard::new(0x7F7F7F7F7F7F7F7F);
    let white_pawn_attacks = BitBoard::new(((white_pawns & not_a_file).0 << 7) | 
                                           ((white_pawns & not_h_file).0 << 9));
    let black_pawn_attacks = BitBoard::new(((black_pawns & not_a_file).0 >> 9) | 
                                           ((black_pawns & not_h_file).0 >> 7));
    
    score += evaluate_rooks(board);
    score += evaluate_pawns(board, pawn_hash, white_pawn_attacks, black_pawn_attacks);
    score += evaluate_king_tropism(board, phase);
    score += evaluate_king_pawn_shield(board, phase);
    score += evaluate_pieces_and_king_safety(board, phase, white_pawn_attacks, black_pawn_attacks);
    score += evaluate_space(board, phase);
    score += evaluate_bad_bishops(board, phase);
    score += evaluate_pawn_threats(board, white_pawn_attacks, black_pawn_attacks, phase);   
    if (board.pieces(Piece::Bishop) & board.color_combined(Color::White)).popcnt() >= 2 {
        score += ((25 * phase) + (40 * (24 - phase))) / 24;
    }
    if (board.pieces(Piece::Bishop) & board.color_combined(Color::Black)).popcnt() >= 2 {
        score -= ((25 * phase) + (40 * (24 - phase))) / 24;
    }

    let final_score = if board.side_to_move() == Color::White {
        score
    } else {
        -score
    };
    final_score
}



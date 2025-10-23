use crate::engine::constants::*;
use crate::engine::move_ordering::VALUES;
use chess::{BitBoard, Board, Color, MoveGen, Piece, Rank, Square};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    pub static ref PST: HashMap<Piece, HashMap<&'static str, Vec<i32>>> = {
        let mut m = HashMap::new();
        m.insert(Piece::Pawn, {
            let mut sub = HashMap::new();
            sub.insert(
                "mg",
                vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65,
                    56, 25, -20, -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25,
                    -26, -4, -4, -10, 3, 3, 33, -12, -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0,
                    0, 0, 0, 0, 0,
                ],
            );
            sub.insert(
                "eg",
                vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85,
                    67, 56, 53, 82, 84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1,
                    4, 7, -6, 1, 0, -5, -1, -8, 13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            );
            sub
        });
        m.insert(Piece::Knight, {
            let mut sub = HashMap::new();
            sub.insert(
                "mg",
                vec![
                    -167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47,
                    60, 37, 65, 84, 129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13,
                    28, 19, 21, -8, -23, -9, 12, 10, 19, 17, 25, -16, -29, -53, -12, -3, -1, 18,
                    -14, -19, -105, -21, -58, -33, -17, -28, -19, -23,
                ],
            );
            sub.insert(
                "eg",
                vec![
                    -58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52,
                    -24, -20, 10, 9, -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16,
                    25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22, -42, -20, -10, -5, -2,
                    -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64,
                ],
            );
            sub
        });
        m.insert(Piece::Bishop, {
            let mut sub = HashMap::new();
            sub.insert(
                "mg",
                vec![
                    -29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37,
                    43, 40, 35, 50, 37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12,
                    10, 4, 0, 15, 15, 15, 14, 27, 18, 10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14,
                    -21, -13, -12, -39, -21,
                ],
            );
            sub.insert(
                "eg",
                vec![
                    -14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8,
                    0, -1, -2, 6, 0, 4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9,
                    -12, -3, 8, 10, 13, 3, -7, -15, -14, -18, -7, -1, 4, -9, -15, -27, -23, -9,
                    -23, -5, -9, -16, -5, -17,
                ],
            );
            sub
        });
        m.insert(Piece::Rook, {
            let mut sub = HashMap::new();
            sub.insert(
                "mg",
                vec![
                    32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36,
                    17, 45, 61, 16, -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6,
                    -23, -45, -25, -16, -17, 3, 0, -5, -33, -44, -16, -20, -9, -1, 11, -6, -71,
                    -19, -13, 1, 17, 16, 7, -37, -26,
                ],
            );
            sub.insert(
                "eg",
                vec![
                    13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3,
                    -5, -3, 4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1,
                    -7, -12, -8, -16, -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20,
                ],
            );
            sub
        });
        m.insert(Piece::Queen, {
            let mut sub = HashMap::new();
            sub.insert(
                "mg",
                vec![
                    -28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7,
                    8, 29, 56, 47, 57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4,
                    3, -3, -14, 2, -11, -2, -5, 2, 14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18,
                    -9, 10, -15, -25, -31, -50,
                ],
            );
            sub.insert(
                "eg",
                vec![
                    -9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49,
                    47, 35, 19, 9, 3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23,
                    -16, -27, 15, 6, 9, 17, 10, 5, -22, -23, -30, -16, -16, -23, -36, -32, -33,
                    -28, -22, -43, -5, -32, -20, -41,
                ],
            );
            sub
        });
        m.insert(Piece::King, {
            let mut sub = HashMap::new();
            sub.insert(
                "mg",
                vec![
                    -65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24,
                    2, -16, -20, 6, 22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27,
                    -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27, 1, 7, -8, -64,
                    -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24, 14,
                ],
            );
            sub.insert(
                "eg",
                vec![
                    -74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17,
                    23, 15, 20, 45, 44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23,
                    9, -11, -19, -3, 11, 21, 23, 16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53,
                    -34, -21, -11, -28, -14, -24, -43,
                ],
            );
            sub
        });
        m
    };
    pub static ref MATERIAL_HASH_TABLE: Mutex<MaterialHashTable> =
        Mutex::new(MaterialHashTable::new(16));
    pub static ref PAWN_HASH_TABLE: Mutex<HashMap<u64, i32>> = Mutex::new(HashMap::new());
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
}

pub struct MaterialHashTable {
    table: Vec<[(u64, i32, usize); 2]>,
    age: usize,
    size: usize,
}
impl MaterialHashTable {
    pub fn new(size_mb: usize) -> Self {
        let size = (size_mb * 1024 * 1024) / 48;
        Self {
            table: vec![[(0, 0, 0); 2]; size],
            age: 0,
            size,
        }
    }
    pub fn lookup(&self, key: u64) -> Option<i32> {
        let idx = (key as usize) % self.size;
        let bucket = &self.table[idx];
        if bucket[0].0 == key {
            return Some(bucket[0].1);
        }
        if bucket[1].0 == key {
            return Some(bucket[1].1);
        }
        None
    }
    pub fn store(&mut self, key: u64, value: i32) {
        self.age += 1;
        let idx = (key as usize) % self.size;
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
}
fn compute_pawn_hash(board: &Board) -> u64 {
    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
    white_pawns.0 ^ (black_pawns.0 << 1)
}

fn compute_material_key(board: &Board) -> u64 {
    let mut key = 0u64;
    for &piece in &[
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
    ] {
        let white_count = (board.pieces(piece) & board.color_combined(Color::White)).popcnt();
        let black_count = (board.pieces(piece) & board.color_combined(Color::Black)).popcnt();
        key = (key << 4) | (white_count as u64);
        key = (key << 4) | (black_count as u64);
    }
    key
}
fn compute_material_eval(board: &Board) -> i32 {
    let mut material = 0;
    for (&piece, &value) in VALUES.iter() {
        let white_count =
            (board.pieces(piece) & board.color_combined(Color::White)).popcnt() as i32;
        let black_count =
            (board.pieces(piece) & board.color_combined(Color::Black)).popcnt() as i32;
        material += white_count * value;
        material -= black_count * value;
    }
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
fn evaluate_pawns(board: &Board) -> i32 {
    let pawn_hash = compute_pawn_hash(board);

    {
        let cache = PAWN_HASH_TABLE.lock().unwrap();
        if let Some(&cached_score) = cache.get(&pawn_hash) {
            return cached_score;
        }
    }

    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);

    let mut score = 0;
    let mut white_file_counts = [0u8; 8];
    let mut black_file_counts = [0u8; 8];
    let mut white_file_ranks = [8u8; 8];
    let mut black_file_ranks = [8u8; 8];

    for square in white_pawns {
        let file = square.get_file().to_index();
        let rank = square.get_rank().to_index();
        white_file_counts[file] += 1;
        white_file_ranks[file] = white_file_ranks[file].min(rank as u8);
    }
    for square in black_pawns {
        let file = square.get_file().to_index();
        let rank = square.get_rank().to_index();
        black_file_counts[file] += 1;
        black_file_ranks[file] = black_file_ranks[file].min(rank as u8);
    }

    for file_idx in 0..8 {
        if white_file_counts[file_idx] >= 2 {
            score -= 15 * (white_file_counts[file_idx] - 1) as i32;
        }
        if black_file_counts[file_idx] >= 2 {
            score += 15 * (black_file_counts[file_idx] - 1) as i32;
        }

        if white_file_counts[file_idx] > 0 {
            let has_adjacent = (file_idx > 0 && white_file_counts[file_idx - 1] > 0)
                || (file_idx < 7 && white_file_counts[file_idx + 1] > 0);
            if !has_adjacent {
                score -= 12;
            }
        }

        if black_file_counts[file_idx] > 0 {
            let has_adjacent = (file_idx > 0 && black_file_counts[file_idx - 1] > 0)
                || (file_idx < 7 && black_file_counts[file_idx + 1] > 0);
            if !has_adjacent {
                score += 12;
            }
        }
    }

    for square in white_pawns {
        let sq_idx = square.to_index();
        let file = square.get_file().to_index();
        let rank = square.get_rank().to_index();

        if (WHITE_PASSED_MASKS[sq_idx] & black_pawns) == BitBoard::new(0) {
            let passed_bonus = [0, 5, 10, 20, 35, 55, 80, 0][rank];
            score += passed_bonus;
        } else {
            let mut can_advance = true;
            let mut supported = false;

            if file > 0
                && white_file_counts[file - 1] > 0
                && white_file_ranks[file - 1] <= rank as u8
            {
                supported = true;
            }
            if file < 7
                && white_file_counts[file + 1] > 0
                && white_file_ranks[file + 1] <= rank as u8
            {
                supported = true;
            }

            if rank < 7 {
                let front_mask = BitBoard::new(1u64 << ((rank + 1) * 8 + file));
                if (front_mask & white_pawns) != BitBoard::new(0) {
                    can_advance = false;
                }

                let enemy_mask = if file > 0 {
                    BitBoard::new(1u64 << ((rank + 1) * 8 + file - 1))
                } else {
                    BitBoard::new(0)
                } | if file < 7 {
                    BitBoard::new(1u64 << ((rank + 1) * 8 + file + 1))
                } else {
                    BitBoard::new(0)
                };

                if (enemy_mask & black_pawns) != BitBoard::new(0) {
                    can_advance = false;
                }
            }

            if !can_advance && !supported {
                score -= 10;
            }
        }

        let is_backward = if file > 0 && file < 7 {
            let left_support =
                white_file_counts[file - 1] > 0 && white_file_ranks[file - 1] < rank as u8;
            let right_support =
                white_file_counts[file + 1] > 0 && white_file_ranks[file + 1] < rank as u8;
            let left_enemy =
                black_file_counts[file - 1] > 0 && black_file_ranks[file - 1] >= rank as u8;
            let right_enemy =
                black_file_counts[file + 1] > 0 && black_file_ranks[file + 1] >= rank as u8;

            !left_support && !right_support && (left_enemy || right_enemy)
        } else if file == 0 {
            let right_support = white_file_counts[1] > 0 && white_file_ranks[1] < rank as u8;
            let right_enemy = black_file_counts[1] > 0 && black_file_ranks[1] >= rank as u8;
            !right_support && right_enemy
        } else {
            let left_support = white_file_counts[6] > 0 && white_file_ranks[6] < rank as u8;
            let left_enemy = black_file_counts[6] > 0 && black_file_ranks[6] >= rank as u8;
            !left_support && left_enemy
        };

        if is_backward {
            score -= 8;
        }

        if file > 0 {
            let left_chain = unsafe { Square::new((sq_idx - 9) as u8) };
            if rank > 0 && (BitBoard::from_square(left_chain) & white_pawns) != BitBoard::new(0) {
                score += 3;
            }
        }
        if file < 7 {
            let right_chain = unsafe { Square::new((sq_idx - 7) as u8) };
            if rank > 0 && (BitBoard::from_square(right_chain) & white_pawns) != BitBoard::new(0) {
                score += 3;
            }
        }
    }

    for square in black_pawns {
        let sq_idx = square.to_index();
        let file = square.get_file().to_index();
        let rank = square.get_rank().to_index();

        if (BLACK_PASSED_MASKS[sq_idx] & white_pawns) == BitBoard::new(0) {
            let passed_bonus = [0, 80, 55, 35, 20, 10, 5, 0][rank];
            score -= passed_bonus;
        } else {
            let mut can_advance = true;
            let mut supported = false;

            if file > 0
                && black_file_counts[file - 1] > 0
                && black_file_ranks[file - 1] >= rank as u8
            {
                supported = true;
            }
            if file < 7
                && black_file_counts[file + 1] > 0
                && black_file_ranks[file + 1] >= rank as u8
            {
                supported = true;
            }

            if rank > 0 {
                let front_mask = BitBoard::new(1u64 << ((rank - 1) * 8 + file));
                if (front_mask & black_pawns) != BitBoard::new(0) {
                    can_advance = false;
                }

                let enemy_mask = if file > 0 {
                    BitBoard::new(1u64 << ((rank - 1) * 8 + file - 1))
                } else {
                    BitBoard::new(0)
                } | if file < 7 {
                    BitBoard::new(1u64 << ((rank - 1) * 8 + file + 1))
                } else {
                    BitBoard::new(0)
                };

                if (enemy_mask & white_pawns) != BitBoard::new(0) {
                    can_advance = false;
                }
            }

            if !can_advance && !supported {
                score += 10;
            }
        }

        let is_backward = if file > 0 && file < 7 {
            let left_support =
                black_file_counts[file - 1] > 0 && black_file_ranks[file - 1] > rank as u8;
            let right_support =
                black_file_counts[file + 1] > 0 && black_file_ranks[file + 1] > rank as u8;
            let left_enemy =
                white_file_counts[file - 1] > 0 && white_file_ranks[file - 1] <= rank as u8;
            let right_enemy =
                white_file_counts[file + 1] > 0 && white_file_ranks[file + 1] <= rank as u8;

            !left_support && !right_support && (left_enemy || right_enemy)
        } else if file == 0 {
            let right_support = black_file_counts[1] > 0 && black_file_ranks[1] > rank as u8;
            let right_enemy = white_file_counts[1] > 0 && white_file_ranks[1] <= rank as u8;
            !right_support && right_enemy
        } else {
            let left_support = black_file_counts[6] > 0 && black_file_ranks[6] > rank as u8;
            let left_enemy = white_file_counts[6] > 0 && white_file_ranks[6] <= rank as u8;
            !left_support && left_enemy
        };

        if is_backward {
            score += 8;
        }

        if file > 0 {
            let left_chain = unsafe { Square::new((sq_idx + 7) as u8) };
            if rank < 7 && (BitBoard::from_square(left_chain) & black_pawns) != BitBoard::new(0) {
                score -= 3;
            }
        }
        if file < 7 {
            let right_chain = unsafe { Square::new((sq_idx + 9) as u8) };
            if rank < 7 && (BitBoard::from_square(right_chain) & black_pawns) != BitBoard::new(0) {
                score -= 3;
            }
        }
    }

    {
        let mut cache = PAWN_HASH_TABLE.lock().unwrap();
        if cache.len() >= 10000 {
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }
        cache.insert(pawn_hash, score);
    }

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
fn get_king_ring(king_sq: Square) -> BitBoard {
    let king_file = king_sq.get_file().to_index() as i32;
    let king_rank = king_sq.get_rank().to_index() as i32;
    let mut ring = BitBoard::new(0);

    for file_offset in -1..=1 {
        for rank_offset in -1..=1 {
            if file_offset == 0 && rank_offset == 0 {
                continue;
            }
            let new_file = king_file + file_offset;
            let new_rank = king_rank + rank_offset;
            if new_file >= 0 && new_file < 8 && new_rank >= 0 && new_rank < 8 {
                let sq_idx = (new_rank * 8 + new_file) as u8;
                ring |= BitBoard::from_square(unsafe { Square::new(sq_idx) });
            }
        }
    }
    ring
}

fn count_piece_attacks(
    pieces: BitBoard,
    target_ring: BitBoard,
    piece_type: Piece,
    board: &Board,
) -> i32 {
    let mut attacks = 0;

    for square in pieces {
        let attack_bb = match piece_type {
            Piece::Queen => {
                let rook_attacks = chess::get_rook_rays(square) & !board.combined();
                let bishop_attacks = chess::get_bishop_rays(square) & !board.combined();
                rook_attacks | bishop_attacks
            }
            Piece::Rook => chess::get_rook_rays(square) & !board.combined(),
            Piece::Bishop => chess::get_bishop_rays(square) & !board.combined(),
            Piece::Knight => chess::get_knight_moves(square),
            Piece::Pawn => {
                let color = if (board.color_combined(Color::White) & BitBoard::from_square(square))
                    != BitBoard::new(0)
                {
                    Color::White
                } else {
                    Color::Black
                };
                chess::get_pawn_attacks(square, color, *board.combined())
            }
            _ => BitBoard::new(0),
        };

        attacks += (attack_bb & target_ring).popcnt() as i32;
    }

    attacks
}

fn evaluate_king_ring_attacks(board: &Board, phase: i32) -> i32 {
    let white_king_sq = (board.pieces(Piece::King) & board.color_combined(Color::White))
        .into_iter()
        .next()
        .unwrap();
    let black_king_sq = (board.pieces(Piece::King) & board.color_combined(Color::Black))
        .into_iter()
        .next()
        .unwrap();

    let white_king_ring = get_king_ring(white_king_sq);
    let black_king_ring = get_king_ring(black_king_sq);

    let mut mg_score = 0;
    let mut eg_score = 0;

    for (i, &piece_type) in PIECE_ORDER.iter().enumerate() {
        let white_pieces = board.pieces(piece_type) & board.color_combined(Color::White);
        let black_pieces = board.pieces(piece_type) & board.color_combined(Color::Black);

        let white_attacks = count_piece_attacks(white_pieces, black_king_ring, piece_type, board);
        let black_attacks = count_piece_attacks(black_pieces, white_king_ring, piece_type, board);

        let (mg_weight, eg_weight) = ATTACK_WEIGHTS[i];

        mg_score += white_attacks * mg_weight;
        mg_score -= black_attacks * mg_weight;
        eg_score += white_attacks * eg_weight;
        eg_score -= black_attacks * eg_weight;
    }

    ((mg_score * phase) + (eg_score * (24 - phase))) / 24
}
fn evaluate_mobility(board: &Board, phase: i32) -> i32 {
    let mut mg_score = 0;
    let mut eg_score = 0;

    let white_pieces = board.color_combined(Color::White);
    let black_pieces = board.color_combined(Color::Black);
    let occupied = board.combined();

    // White piece mobility
    let white_knights = board.pieces(Piece::Knight) & white_pieces;
    for knight_sq in white_knights {
        let moves = chess::get_knight_moves(knight_sq);
        let mobility = (moves & !white_pieces).popcnt() as usize;
        let clamped_mobility = mobility.min(8);
        mg_score += KNIGHT_MOBILITY_MG[clamped_mobility];
        eg_score += KNIGHT_MOBILITY_EG[clamped_mobility];
    }

    let white_bishops = board.pieces(Piece::Bishop) & white_pieces;
    for bishop_sq in white_bishops {
        let moves = chess::get_bishop_rays(bishop_sq) & !occupied;
        let mobility = (moves & !white_pieces).popcnt() as usize;
        let clamped_mobility = mobility.min(13);
        mg_score += BISHOP_MOBILITY_MG[clamped_mobility];
        eg_score += BISHOP_MOBILITY_EG[clamped_mobility];
    }

    let white_rooks = board.pieces(Piece::Rook) & white_pieces;
    for rook_sq in white_rooks {
        let moves = chess::get_rook_rays(rook_sq) & !occupied;
        let mobility = (moves & !white_pieces).popcnt() as usize;
        let clamped_mobility = mobility.min(14);
        mg_score += ROOK_MOBILITY_MG[clamped_mobility];
        eg_score += ROOK_MOBILITY_EG[clamped_mobility];
    }

    let white_queens = board.pieces(Piece::Queen) & white_pieces;
    for queen_sq in white_queens {
        let rook_moves = chess::get_rook_rays(queen_sq) & !occupied;
        let bishop_moves = chess::get_bishop_rays(queen_sq) & !occupied;
        let moves = rook_moves | bishop_moves;
        let mobility = (moves & !white_pieces).popcnt() as usize;
        let clamped_mobility = mobility.min(27);
        mg_score += QUEEN_MOBILITY_MG[clamped_mobility];
        eg_score += QUEEN_MOBILITY_EG[clamped_mobility];
    }

    // Black piece mobility
    let black_knights = board.pieces(Piece::Knight) & black_pieces;
    for knight_sq in black_knights {
        let moves = chess::get_knight_moves(knight_sq);
        let mobility = (moves & !black_pieces).popcnt() as usize;
        let clamped_mobility = mobility.min(8);
        mg_score -= KNIGHT_MOBILITY_MG[clamped_mobility];
        eg_score -= KNIGHT_MOBILITY_EG[clamped_mobility];
    }

    let black_bishops = board.pieces(Piece::Bishop) & black_pieces;
    for bishop_sq in black_bishops {
        let moves = chess::get_bishop_rays(bishop_sq) & !occupied;
        let mobility = (moves & !black_pieces).popcnt() as usize;
        let clamped_mobility = mobility.min(13);
        mg_score -= BISHOP_MOBILITY_MG[clamped_mobility];
        eg_score -= BISHOP_MOBILITY_EG[clamped_mobility];
    }

    let black_rooks = board.pieces(Piece::Rook) & black_pieces;
    for rook_sq in black_rooks {
        let moves = chess::get_rook_rays(rook_sq) & !occupied;
        let mobility = (moves & !black_pieces).popcnt() as usize;
        let clamped_mobility = mobility.min(14);
        mg_score -= ROOK_MOBILITY_MG[clamped_mobility];
        eg_score -= ROOK_MOBILITY_EG[clamped_mobility];
    }

    let black_queens = board.pieces(Piece::Queen) & black_pieces;
    for queen_sq in black_queens {
        let rook_moves = chess::get_rook_rays(queen_sq) & !occupied;
        let bishop_moves = chess::get_bishop_rays(queen_sq) & !occupied;
        let moves = rook_moves | bishop_moves;
        let mobility = (moves & !black_pieces).popcnt() as usize;
        let clamped_mobility = mobility.min(27);
        mg_score -= QUEEN_MOBILITY_MG[clamped_mobility];
        eg_score -= QUEEN_MOBILITY_EG[clamped_mobility];
    }

    // Taper the score based on game phase
    ((mg_score * phase) + (eg_score * (24 - phase))) / 24
}
fn evaluate_space(board: &Board, phase: i32) -> i32 {
    if phase < 8 {
        return 0;
    }
    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
    let white_space_mask = BitBoard::new(0x00FFFF0000000000);
    let black_space_mask = BitBoard::new(0x0000000000FFFF00);
    let white_controlled = {
        let mut control = BitBoard::new(0);
        for pawn_sq in white_pawns {
            let file = pawn_sq.get_file().to_index();
            let rank = pawn_sq.get_rank().to_index();
            if rank < 7 {
                if file > 0 {
                    let sq = unsafe { Square::new(((rank + 1) * 8 + file - 1) as u8) };
                    control |= BitBoard::from_square(sq);
                }
                if file < 7 {
                    let sq = unsafe { Square::new(((rank + 1) * 8 + file + 1) as u8) };
                    control |= BitBoard::from_square(sq);
                }
            }
        }
        control
    };
    let black_controlled = {
        let mut control = BitBoard::new(0);
        for pawn_sq in black_pawns {
            let file = pawn_sq.get_file().to_index();
            let rank = pawn_sq.get_rank().to_index();
            if rank > 0 {
                if file > 0 {
                    let sq = unsafe { Square::new(((rank - 1) * 8 + file - 1) as u8) };
                    control |= BitBoard::from_square(sq);
                }
                if file < 7 {
                    let sq = unsafe { Square::new(((rank - 1) * 8 + file + 1) as u8) };
                    control |= BitBoard::from_square(sq);
                }
            }
        }
        control
    };
    let white_space = (white_controlled & white_space_mask & !black_controlled).popcnt() as i32;
    let black_space = (black_controlled & black_space_mask & !white_controlled).popcnt() as i32;
    let space_diff = white_space - black_space;
    (space_diff * phase * 4) / 24
}
pub fn evaluate(board: &Board) -> i32 {
    let in_check = *board.checkers() != BitBoard(0);
    let has_legal_moves = MoveGen::new_legal(board).next().is_some();

    if !has_legal_moves {
        return if in_check {
            if board.side_to_move() == Color::White { -30000 } else { 30000 }
        } else {
            if board.side_to_move() == Color::White { CONTEMPT } else { -CONTEMPT }
        };
    }
    let material_hash_table = MATERIAL_HASH_TABLE.lock().unwrap();
    let material_key = compute_material_key(board);
    let material = material_hash_table.lookup(material_key);
    drop(material_hash_table);
    let material = match material {
        Some(m) => m,
        None => {
            let m = compute_material_eval(board);
            let mut material_hash_table = MATERIAL_HASH_TABLE.lock().unwrap();
            material_hash_table.store(material_key, m);
            m
        }
    };
    let mut score = material;
    let mut phase = 0;
    phase += (board.pieces(Piece::Knight).popcnt() + board.pieces(Piece::Bishop).popcnt()) as i32;
    phase += (board.pieces(Piece::Rook).popcnt() * 2) as i32;
    phase += (board.pieces(Piece::Queen).popcnt() * 4) as i32;
    phase = phase.min(24);
    
    for sq_idx in 0..64 {
        let square = unsafe { Square::new(sq_idx) };
        if let Some(piece) = board.piece_on(square) {
            let piece_color = if (*board.color_combined(Color::White)
                & BitBoard::from_square(square))
                != BitBoard(0)
            {
                Color::White
            } else {
                Color::Black
            };
            let pst_entry = &PST[&piece];
            let pst_index = if piece_color == Color::White {
                (sq_idx ^ 56) as usize
            } else {
                sq_idx as usize
            };
            let mg_pst = pst_entry["mg"][pst_index];
            let eg_pst = pst_entry["eg"][pst_index];
            let pst_value = ((mg_pst * phase) + (eg_pst * (24 - phase))) / 24;
            if piece_color == Color::White {
                score += pst_value;
            } else {
                score -= pst_value;
            }
            // Bonus for minor pieces with pawn directly in front
            if piece == Piece::Knight || piece == Piece::Bishop {
                let rank = square.get_rank().to_index();
                let file = square.get_file().to_index();
                if piece_color == Color::White && rank < 7 {
                    let ahead_sq = unsafe { Square::new(((rank + 1) * 8 + file) as u8) };
                    let white_pawns =
                        board.pieces(Piece::Pawn) & board.color_combined(Color::White);
                    if (white_pawns & BitBoard::from_square(ahead_sq)) != BitBoard::new(0) {
                        score += 15;
                    }
                } else if piece_color == Color::Black && rank > 0 {
                    let ahead_sq = unsafe { Square::new(((rank - 1) * 8 + file) as u8) };
                    let black_pawns =
                        board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
                    if (black_pawns & BitBoard::from_square(ahead_sq)) != BitBoard::new(0) {
                        score -= 15;
                    }
                }
            }
        }
    }
    
    score += evaluate_rooks(board);
    score += evaluate_pawns(board);
    score += evaluate_king_tropism(board, phase);
    score += evaluate_king_ring_attacks(board, phase);
    score += evaluate_mobility(board, phase);
    score += evaluate_space(board, phase);
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

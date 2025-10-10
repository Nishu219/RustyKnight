use chess::{BitBoard, Board, BoardStatus, ChessMove, Color, MoveGen, Piece, Rank, Square};
use lazy_static::lazy_static;
use rand::Rng;
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::str::FromStr;
use std::sync::Mutex;
use std::time::Instant;
const MAX_DEPTH: usize = 40;
const INITIAL_WINDOW: i32 = 50;
const MAX_WINDOW: i32 = 400;
const STATS: bool = true;
const CONTEMPT: i32 = 25;
const LMR_FULL_DEPTH_MOVES: usize = 4;
const LMR_REDUCTION_LIMIT: usize = 3;
const NULL_MOVE_DEPTH: usize = 3;
const IID_DEPTH: usize = 4;
const RAZOR_DEPTH: usize = 4;
const RAZOR_MARGIN: i32 = 400;
const REVERSE_FUTILITY_DEPTH: usize = 6;
const REVERSE_FUTILITY_MARGIN: i32 = 120;
const FUTILITY_DEPTH: usize = 4;
const FUTILITY_MARGINS: [i32; 5] = [0, 200, 400, 600, 800];
const LMP_DEPTH: usize = 4;
const TROPISM_WEIGHTS: [(i32, i32); 4] = [(3, 2), (2, 3), (1, 1), (2, 1)];
const ATTACK_WEIGHTS: [(i32, i32); 5] = [(4, 2), (3, 4), (2, 1), (3, 1), (1, 0)];
const KNIGHT_MOBILITY_MG: [i32; 9] = [-25, -15, -10, -5, 0, 5, 10, 15, 20];
const KNIGHT_MOBILITY_EG: [i32; 9] = [-30, -20, -15, -10, -5, 0, 5, 10, 15];
const BISHOP_MOBILITY_MG: [i32; 14] = [-20, -15, -10, -5, 0, 5, 8, 12, 15, 18, 20, 22, 24, 25];
const BISHOP_MOBILITY_EG: [i32; 14] = [-15, -10, -5, 0, 5, 10, 13, 16, 19, 22, 25, 27, 29, 31];
const ROOK_MOBILITY_MG: [i32; 15] = [-15, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 15];
const ROOK_MOBILITY_EG: [i32; 15] = [-10, -5, -3, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22];
const QUEEN_MOBILITY_MG: [i32; 28] = [-10, -8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
const QUEEN_MOBILITY_EG: [i32; 28] = [-5, -3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
const PIECE_ORDER: [Piece; 5] = [
    Piece::Queen,
    Piece::Rook,
    Piece::Bishop,
    Piece::Knight,
    Piece::Pawn,
];
lazy_static! {
    static ref VALUES: HashMap<Piece, i32> = {
        let mut m = HashMap::new();
        m.insert(Piece::Pawn, 100);
        m.insert(Piece::Knight, 320);
        m.insert(Piece::Bishop, 330);
        m.insert(Piece::Rook, 500);
        m.insert(Piece::Queen, 900);
        m.insert(Piece::King, 0);
        m
    };
    static ref ZOBRIST_PIECES: HashMap<(Piece, Color), Vec<u64>> = {
        let mut rng = rand::thread_rng();
        let mut m = HashMap::new();
        for piece in &[
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ] {
            for color in &[Color::White, Color::Black] {
                let mut v = Vec::with_capacity(64);
                for _ in 0..64 {
                    v.push(rng.gen::<u64>());
                }
                m.insert((*piece, *color), v);
            }
        }
        m
    };
    static ref ZOBRIST_CASTLING: Vec<u64> = {
        let mut rng = rand::thread_rng();
        (0..16).map(|_| rng.gen::<u64>()).collect()
    };
    static ref ZOBRIST_EP: Vec<u64> = {
        let mut rng = rand::thread_rng();
        (0..8).map(|_| rng.gen::<u64>()).collect()
    };
    static ref ZOBRIST_TURN: u64 = rand::thread_rng().gen::<u64>();
    static ref PST: HashMap<Piece, HashMap<&'static str, Vec<i32>>> = {
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
    static ref MATERIAL_HASH_TABLE: Mutex<MaterialHashTable> =
        Mutex::new(MaterialHashTable::new(16));
    static ref KILLER_MOVES: Mutex<Vec<Vec<ChessMove>>> = Mutex::new(Vec::new());
    static ref HISTORY_HEURISTIC: Mutex<HashMap<ChessMove, i32>> = Mutex::new(HashMap::new());
    static ref TRANSPOSITION_TABLE: Mutex<TranspositionTable> =
        Mutex::new(TranspositionTable::new(256));
    static ref REPETITION_TABLE: Mutex<HashMap<u64, usize>> = Mutex::new(HashMap::new());
    static ref PAWN_HASH_TABLE: Mutex<HashMap<u64, i32>> = Mutex::new(HashMap::new());
    static ref FILE_MASKS: [BitBoard; 8] = [
        BitBoard::new(0x0101010101010101),
        BitBoard::new(0x0202020202020202),
        BitBoard::new(0x0404040404040404),
        BitBoard::new(0x0808080808080808),
        BitBoard::new(0x1010101010101010),
        BitBoard::new(0x2020202020202020),
        BitBoard::new(0x4040404040404040),
        BitBoard::new(0x8080808080808080),
    ];
    static ref ADJACENT_FILES: [BitBoard; 8] = [
        BitBoard::new(0x0202020202020202),
        BitBoard::new(0x0505050505050505),
        BitBoard::new(0x0A0A0A0A0A0A0A0A),
        BitBoard::new(0x1414141414141414),
        BitBoard::new(0x2828282828282828),
        BitBoard::new(0x5050505050505050),
        BitBoard::new(0xA0A0A0A0A0A0A0A0),
        BitBoard::new(0x4040404040404040),
    ];
    static ref WHITE_PASSED_MASKS: [BitBoard; 64] = {
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
    static ref BLACK_PASSED_MASKS: [BitBoard; 64] = {
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
struct TranspositionTable {
    table: HashMap<u64, TTEntry>,
    hits: usize,
    age: usize,
    size: usize,
}
#[derive(Clone)]
struct TTEntry {
    depth: usize,
    value: i32,
    flag: TTFlag,
    move_: Option<ChessMove>,
    age: usize,
}
#[derive(Clone, PartialEq)]
enum TTFlag {
    Exact,
    Lower,
    Upper,
}
impl TranspositionTable {
    fn new(size_mb: usize) -> Self {
        Self {
            table: HashMap::new(),
            hits: 0,
            age: 0,
            size: (size_mb * 1024 * 1024) / 24,
        }
    }
    fn clear(&mut self) {
        self.table.clear();
        self.hits = 0;
        self.age += 1;
    }
    fn get(&mut self, key: u64, depth: usize, alpha: i32, beta: i32) -> Option<i32> {
        if let Some(entry) = self.table.get(&key) {
            if entry.depth >= depth {
                self.hits += 1;
                let val = entry.value;
                let flag = &entry.flag;
                match flag {
                    TTFlag::Exact => return Some(val),
                    TTFlag::Lower if val >= beta => return Some(val),
                    TTFlag::Upper if val <= alpha => return Some(val),
                    _ => {}
                }
            }
        }
        None
    }
    fn store(
        &mut self,
        key: u64,
        depth: usize,
        value: i32,
        flag: TTFlag,
        move_: Option<ChessMove>,
    ) {
        if self.table.len() >= self.size {
            let oldest_key = *self
                .table
                .iter()
                .min_by_key(|(_, entry)| entry.age)
                .map(|(k, _)| k)
                .unwrap();
            self.table.remove(&oldest_key);
        }
        self.table.insert(
            key,
            TTEntry {
                depth,
                value,
                flag,
                move_,
                age: self.age,
            },
        );
    }
    fn get_move(&self, key: u64) -> Option<ChessMove> {
        self.table.get(&key).and_then(|entry| entry.move_)
    }
}

struct MaterialHashTable {
    table: Vec<[(u64, i32, usize); 2]>,
    age: usize,
    size: usize,
}
impl MaterialHashTable {
    fn new(size_mb: usize) -> Self {
        let size = (size_mb * 1024 * 1024) / 48;
        Self {
            table: vec![[(0, 0, 0); 2]; size],
            age: 0,
            size,
        }
    }
    fn lookup(&self, key: u64) -> Option<i32> {
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
    fn store(&mut self, key: u64, value: i32) {
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
fn is_repetition(position_hash: u64) -> bool {
    let rep_table = REPETITION_TABLE.lock().unwrap();
    *rep_table.get(&position_hash).unwrap_or(&0) >= 2
}

fn update_repetition_table(position_hash: u64) {
    let mut rep_table = REPETITION_TABLE.lock().unwrap();
    *rep_table.entry(position_hash).or_insert(0) += 1;
}

fn clear_repetition_table() {
    let mut rep_table = REPETITION_TABLE.lock().unwrap();
    rep_table.clear();
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
fn phase_value(board: &Board) -> i32 {
    let mut phase = 0;
    for sq_idx in 0..64 {
        let square = unsafe { Square::new(sq_idx) };
        if let Some(piece) = board.piece_on(square) {
            match piece {
                Piece::Knight | Piece::Bishop => phase += 1,
                Piece::Rook => phase += 2,
                Piece::Queen => phase += 4,
                _ => {}
            }
        }
    }
    std::cmp::min(phase, 24)
}
/// Static Exchange Evaluation with Threshold
/// Returns TRUE if the capture sequence meets or exceeds the threshold
/// Returns FALSE if the capture sequence is below the threshold
/// 
/// This allows efficient pruning decisions:
/// - see_capture(board, mv, 0) -> Is capture winning or equal?
/// - see_capture(board, mv, -100) -> Is capture not too bad (loss < 1 pawn)?
/// - see_capture(board, mv, 200) -> Does capture win at least 2 pawns?
fn see_capture(board: &Board, mv: ChessMove, threshold: i32) -> bool {
    let to_square = mv.get_dest();
    let from_square = mv.get_source();
    
    // Get initial capture value
    let captured_piece = match board.piece_on(to_square) {
        Some(p) => p,
        None => {
            // Handle en passant
            if mv.get_promotion().is_some() {
                return VALUES[&Piece::Queen] - VALUES[&Piece::Pawn] >= threshold;
            }
            return false; // Not a capture
        }
    };
    
    let moving_piece = match board.piece_on(from_square) {
        Some(p) => p,
        None => return false,
    };
    
    let mut see_value = VALUES[&captured_piece];
    let mut trophy_value = VALUES[&moving_piece];
    
    // Quick exit: obviously winning capture (e.g., PxQ)
    if see_value - trophy_value >= threshold {
        return true;
    }
    
    // Quick exit: captured piece value alone doesn't meet threshold
    if see_value < threshold {
        return false;
    }
    
    // Determine sides
    let to_move_mask = board.color_combined(board.side_to_move());
    let to_move = if (BitBoard::from_square(from_square) & to_move_mask) != BitBoard::new(0) {
        board.side_to_move()
    } else {
        !board.side_to_move()
    };
    let opponent = !to_move;
    
    // Generate all attackers for both sides
    let mut attacks_to_move = BitBoard::new(0);
    let mut attacks_opponent = BitBoard::new(0);
    
    // Pawns
    attacks_to_move |= get_pawn_attackers(to_square, to_move, board);
    attacks_opponent |= get_pawn_attackers(to_square, opponent, board);
    
    // Early cutoff: can opponent's pawn capture and cause immediate fail?
    if attacks_opponent != BitBoard::new(0) 
        && see_value - trophy_value + VALUES[&Piece::Pawn] < threshold {
        return false;
    }
    
    // Knights
    let knight_attacks = chess::get_knight_moves(to_square);
    attacks_to_move |= knight_attacks & board.pieces(Piece::Knight) & board.color_combined(to_move);
    attacks_opponent |= knight_attacks & board.pieces(Piece::Knight) & board.color_combined(opponent);
    
    // Kings
    let king_attacks = get_king_attacks(to_square);
    attacks_to_move |= king_attacks & board.pieces(Piece::King) & board.color_combined(to_move);
    attacks_opponent |= king_attacks & board.pieces(Piece::King) & board.color_combined(opponent);
    
    // Bishops and Queens (diagonal)
    let bishop_rays = chess::get_bishop_rays(to_square);
    attacks_to_move |= bishop_rays & (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen)) 
        & board.color_combined(to_move);
    attacks_opponent |= bishop_rays & (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen)) 
        & board.color_combined(opponent);
    
    // Rooks and Queens (straight)
    let rook_rays = chess::get_rook_rays(to_square);
    attacks_to_move |= rook_rays & (board.pieces(Piece::Rook) | board.pieces(Piece::Queen)) 
        & board.color_combined(to_move);
    attacks_opponent |= rook_rays & (board.pieces(Piece::Rook) | board.pieces(Piece::Queen)) 
        & board.color_combined(opponent);
    
    // Track occupied squares (remove initial attacker)
    let mut all_pieces = *board.combined();
    let from_bb = BitBoard::from_square(from_square);
    attacks_to_move ^= from_bb;
    all_pieces ^= from_bb;
    
    // Main exchange loop
    loop {
        // ===== OPPONENT'S TURN TO CAPTURE =====
        
        // Check if opponent has any attackers
        if attacks_opponent == BitBoard::new(0) {
            trophy_value = 0;
            // Jump to opponent cut test
        } else if let Some((attacker_sq, attacker_piece)) = 
            find_least_valuable_attacker_threshold(
                board, 
                opponent, 
                attacks_opponent, 
                all_pieces, 
                to_square
            ) {
            // Opponent captures
            see_value -= trophy_value;
            trophy_value = VALUES[&attacker_piece];
            
            // Remove attacker from board
            let attacker_bb = BitBoard::from_square(attacker_sq);
            attacks_opponent ^= attacker_bb;
            all_pieces ^= attacker_bb;
        } else {
            trophy_value = 0;
        }
        
        // Opponent cut test: Can we prune here?
        
        // Upper bound test: side-to-move can stand pat and still win
        if see_value >= threshold {
            return true;
        }
        
        // Lower bound test: even if side-to-move captures, still loses
        if see_value + trophy_value < threshold {
            return false;
        }
        
        // ===== SIDE-TO-MOVE'S TURN TO RECAPTURE =====
        
        // Check if side-to-move has any attackers
        if attacks_to_move == BitBoard::new(0) {
            trophy_value = 0;
            // Jump to to_move cut test
        } else if let Some((attacker_sq, attacker_piece)) = 
            find_least_valuable_attacker_threshold(
                board, 
                to_move, 
                attacks_to_move, 
                all_pieces, 
                to_square
            ) {
            // Side-to-move recaptures
            see_value += trophy_value;
            trophy_value = VALUES[&attacker_piece];
            
            // Remove attacker from board
            let attacker_bb = BitBoard::from_square(attacker_sq);
            attacks_to_move ^= attacker_bb;
            all_pieces ^= attacker_bb;
        } else {
            trophy_value = 0;
        }
        
        // To-move cut test
        
        // Upper bound test: even if opponent captures, side-to-move still wins
        if see_value - trophy_value >= threshold {
            return true;
        }
        
        // Lower bound test: opponent can stand pat and win
        if see_value < threshold {
            return false;
        }
        
        // Continue exchange loop
    }
}

/// Get pawn attackers of a square for a specific color
fn get_pawn_attackers(square: Square, color: Color, board: &Board) -> BitBoard {
    // Pawns attack diagonally opposite to their move direction
    let file = square.get_file().to_index() as i32;
    let rank = square.get_rank().to_index() as i32;
    let mut attackers = BitBoard::new(0);
    
    match color {
        Color::White => {
            // White pawns attack from below (lower ranks)
            if rank > 0 {
                if file > 0 {
                    let sq = unsafe { Square::new(((rank - 1) * 8 + file - 1) as u8) };
                    attackers |= BitBoard::from_square(sq);
                }
                if file < 7 {
                    let sq = unsafe { Square::new(((rank - 1) * 8 + file + 1) as u8) };
                    attackers |= BitBoard::from_square(sq);
                }
            }
        }
        Color::Black => {
            // Black pawns attack from above (higher ranks)
            if rank < 7 {
                if file > 0 {
                    let sq = unsafe { Square::new(((rank + 1) * 8 + file - 1) as u8) };
                    attackers |= BitBoard::from_square(sq);
                }
                if file < 7 {
                    let sq = unsafe { Square::new(((rank + 1) * 8 + file + 1) as u8) };
                    attackers |= BitBoard::from_square(sq);
                }
            }
        }
    }
    
    attackers & board.pieces(Piece::Pawn) & board.color_combined(color)
}

/// Find least valuable attacker considering blockers
/// Returns (Square, Piece) of the attacker, or None if no unblocked attacker exists
fn find_least_valuable_attacker_threshold(
    board: &Board,
    color: Color,
    attackers: BitBoard,
    all_pieces: BitBoard,
    target_square: Square,
) -> Option<(Square, Piece)> {
    // Check pieces in order of value: Pawn, Knight, Bishop, Rook, Queen, King
    
    // Pawns (no blocking check needed)
    let pawns = attackers & board.pieces(Piece::Pawn) & board.color_combined(color);
    if pawns != BitBoard::new(0) {
        for sq in pawns {
            return Some((sq, Piece::Pawn));
        }
    }
    
    // Knights (no blocking check needed)
    let knights = attackers & board.pieces(Piece::Knight) & board.color_combined(color);
    if knights != BitBoard::new(0) {
        for sq in knights {
            return Some((sq, Piece::Knight));
        }
    }
    
    // Bishops (must check for blockers)
    let bishops = attackers & board.pieces(Piece::Bishop) & board.color_combined(color);
    if bishops != BitBoard::new(0) {
        for sq in bishops {
            if !is_path_blocked(sq, target_square, all_pieces) {
                return Some((sq, Piece::Bishop));
            }
        }
    }
    
    // Rooks (must check for blockers)
    let rooks = attackers & board.pieces(Piece::Rook) & board.color_combined(color);
    if rooks != BitBoard::new(0) {
        for sq in rooks {
            if !is_path_blocked(sq, target_square, all_pieces) {
                return Some((sq, Piece::Rook));
            }
        }
    }
    
    // Queens (must check for blockers)
    let queens = attackers & board.pieces(Piece::Queen) & board.color_combined(color);
    if queens != BitBoard::new(0) {
        for sq in queens {
            if !is_path_blocked(sq, target_square, all_pieces) {
                return Some((sq, Piece::Queen));
            }
        }
    }
    
    // King (no blocking check needed)
    let kings = attackers & board.pieces(Piece::King) & board.color_combined(color);
    if kings != BitBoard::new(0) {
        for sq in kings {
            return Some((sq, Piece::King));
        }
    }
    
    None
}

/// Check if path between two squares is blocked
fn is_path_blocked(from: Square, to: Square, occupied: BitBoard) -> bool {
    let from_file = from.get_file().to_index() as i32;
    let from_rank = from.get_rank().to_index() as i32;
    let to_file = to.get_file().to_index() as i32;
    let to_rank = to.get_rank().to_index() as i32;
    
    let file_diff = to_file - from_file;
    let rank_diff = to_rank - from_rank;
    
    // Determine direction
    let file_step = if file_diff > 0 { 1 } else if file_diff < 0 { -1 } else { 0 };
    let rank_step = if rank_diff > 0 { 1 } else if rank_diff < 0 { -1 } else { 0 };
    
    // Check squares between from and to
    let mut current_file = from_file + file_step;
    let mut current_rank = from_rank + rank_step;
    
    while current_file != to_file || current_rank != to_rank {
        let sq = unsafe { Square::new((current_rank * 8 + current_file) as u8) };
        if (occupied & BitBoard::from_square(sq)) != BitBoard::new(0) {
            return true; // Blocked
        }
        current_file += file_step;
        current_rank += rank_step;
    }
    
    false // Not blocked
}
fn get_king_attacks(square: Square) -> BitBoard {
    let file = square.get_file().to_index() as i32;
    let rank = square.get_rank().to_index() as i32;
    let mut attacks = BitBoard::new(0);
    
    for df in -1..=1 {
        for dr in -1..=1 {
            if df == 0 && dr == 0 {
                continue;
            }
            let new_file = file + df;
            let new_rank = rank + dr;
            if new_file >= 0 && new_file < 8 && new_rank >= 0 && new_rank < 8 {
                let sq = unsafe { Square::new((new_rank * 8 + new_file) as u8) };
                attacks |= BitBoard::from_square(sq);
            }
        }
    }
    
    attacks
}
fn mvv_lva_score(board: &Board, mv: ChessMove) -> i32 {
    let victim_value = if let Some(captured_piece) = board.piece_on(mv.get_dest()) {
        VALUES[&captured_piece]
    } else if mv.get_promotion().is_some() {
        VALUES[&Piece::Queen] // Assume queen promotion
    } else {
        0
    };

    let attacker_value = if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
        VALUES[&attacker_piece]
    } else {
        0
    };

    if victim_value > 0 {
        victim_value * 10 - attacker_value
    } else {
        0
    }
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
fn evaluate(board: &Board) -> i32 {
    let is_checkmate = (*board.checkers() != BitBoard(0)) && {
        let mut moves = MoveGen::new_legal(board);
        moves.next().is_none()
    };
    if is_checkmate {
        let result = if board.side_to_move() == Color::White {
            -30000
        } else {
            30000
        };
        return result;
    }
    let is_stalemate = *board.checkers() == BitBoard(0) && {
        let mut moves = MoveGen::new_legal(board);
        moves.next().is_none()
    };
    if is_stalemate {
        let result = if board.side_to_move() == Color::White {
            CONTEMPT
        } else {
            -CONTEMPT
        };
        return result;
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
    let phase = phase_value(board);
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
fn order_moves(
    board: &Board,
    moves: Vec<ChessMove>,
    hash_move: Option<ChessMove>,
    depth: usize,
) -> Vec<ChessMove> {
    let mut scored_moves: Vec<(ChessMove, i32)> = moves
        .into_iter()
        .map(|mv| {
            let score = if Some(mv) == hash_move {
                10000000
            } else if board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() {
                if see_capture(board, mv, 300)
                {
                    9500000
                }
                else if see_capture(board, mv, 0) {
                    7500000
                } else {
                    500000
                }
            } else {
                let new_board = board.make_move_new(mv);
                if *new_board.checkers() != BitBoard(0) {
                    8000000
                } else {
                    let history = HISTORY_HEURISTIC.lock().unwrap();
                    if depth < 64 {
                        let killers = KILLER_MOVES.lock().unwrap();
                        if killers.len() > depth {
                            if killers[depth].len() > 0 && killers[depth][0] == mv {
                                7000000
                            } else if killers[depth].len() > 1 && killers[depth][1] == mv {
                                6000000
                            } else {
                                *history.get(&mv).unwrap_or(&0)
                            }
                        } else {
                            *history.get(&mv).unwrap_or(&0)
                        }
                    } else {
                        *history.get(&mv).unwrap_or(&0)
                    }
                }
            };
            (mv, score)
        })
        .collect();
    scored_moves.sort_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}
fn quiesce(
    board: &Board,
    mut alpha: i32,
    beta: i32,
    start_time: Instant,
    stats: &mut HashMap<&str, usize>,
    root_color: Color,
    max_time: f64,
    timeout_occurred: &mut bool,
) -> i32 {
    *stats.entry("nodes").or_insert(0) += 1;
    *stats.entry("qnodes").or_insert(0) += 1;

    if start_time.elapsed().as_secs_f64() > max_time {
        *timeout_occurred = true;
        return evaluate(board);
    }

    let stand_pat = evaluate(board);

    if stand_pat >= beta {
        return beta;
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let mut captures = Vec::new();
    let movegen = MoveGen::new_legal(board);

    for mv in movegen {
        if board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() {
            let mvv_lva = mvv_lva_score(board, mv);
            captures.push((mv, mvv_lva));
        }
    }

    captures.sort_by(|a, b| b.1.cmp(&a.1));

    for (mv, _) in captures {
        if *timeout_occurred {
            break;
        }
        if !see_capture(board, mv, 0) {
            continue;
        }

        let new_board = board.make_move_new(mv);
        let score = -quiesce(
            &new_board,
            -beta,
            -alpha,
            start_time,
            stats,
            root_color,
            max_time,
            timeout_occurred,
        );

        if *timeout_occurred {
            break;
        }

        if score >= beta {
            return beta;
        }

        if score > alpha {
            alpha = score;
        }
    }

    alpha
}
fn negamax(
    board: &Board,
    depth: usize,
    mut alpha: i32,
    mut beta: i32,
    start_time: Instant,
    ply: usize,
    stats: &mut HashMap<&str, usize>,
    root_color: Color,
    tt: &mut TranspositionTable,
    max_time: f64,
    timeout_occurred: &mut bool,
) -> i32 {
    let is_pv = beta - alpha > 1;
    let is_root = ply == 0;
    let original_alpha = alpha;

    *stats.entry("nodes").or_insert(0) += 1;

    // Timeout check
    if start_time.elapsed().as_secs_f64() > max_time {
        *timeout_occurred = true;
        return if ply > 0 { evaluate(board) } else { 0 };
    }

    let position_hash = compute_zobrist_hash(board);

    // Draw detection and mate distance pruning
    if ply > 0 {
        if is_repetition(position_hash) {
            return if board.side_to_move() == root_color {
                -CONTEMPT
            } else {
                CONTEMPT
            };
        }

        if board.status() == BoardStatus::Checkmate {
            return -30000 + ply as i32;
        }

        if board.status() == BoardStatus::Stalemate {
            return if board.side_to_move() == root_color {
                -CONTEMPT
            } else {
                CONTEMPT
            };
        }

        // Mate distance pruning
        alpha = alpha.max(-30000 + ply as i32);
        beta = beta.min(30000 - ply as i32);
        if alpha >= beta {
            return alpha;
        }
    }

    let board_hash = compute_zobrist_hash(board);
    let mut hash_move = None;
    let mut tt_value = None;

    // Transposition table lookup
    if let Some(stored_value) = tt.get(board_hash, depth, alpha, beta) {
        *stats.entry("tt_hits").or_insert(0) += 1;
        if !is_pv || !is_root {
            return stored_value;
        }
        tt_value = Some(stored_value);
    }

    if let Some(entry) = tt.table.get(&board_hash) {
        hash_move = entry.move_;
        if tt_value.is_none() {
            tt_value = Some(entry.value);
        }
    }

    let in_check = *board.checkers() != BitBoard(0);
    let static_eval = if in_check {
        -30000 + ply as i32
    } else {
        evaluate(board)
    };

    // Quiescence search at leaf nodes
    if depth == 0 {
        return quiesce(
            board,
            alpha,
            beta,
            start_time,
            stats,
            root_color,
            max_time,
            timeout_occurred,
        );
    }

    // Calculate improvement
    let improving = !in_check && ply >= 2 && {
        if let Some(tt_val) = tt_value {
            static_eval > tt_val
        } else {
            true // Assume improving if no TT info
        }
    };

    // Non-PV pruning techniques
    if !is_pv && !in_check {
        // Enhanced Razoring
        if depth <= RAZOR_DEPTH {
            let razor_margin = RAZOR_MARGIN + 50 * (depth as i32 - 1);
            if static_eval + razor_margin < alpha {
                let razor_alpha = alpha - razor_margin;
                let razor_score = quiesce(
                    board,
                    razor_alpha,
                    razor_alpha + 1,
                    start_time,
                    stats,
                    root_color,
                    max_time,
                    timeout_occurred,
                );
                if *timeout_occurred {
                    return evaluate(board);
                }
                if razor_score <= razor_alpha {
                    return razor_score;
                }
            }
        }

        // Enhanced Reverse Futility Pruning (Static Null Move)
        if depth <= REVERSE_FUTILITY_DEPTH {
            let rfp_margin =
                REVERSE_FUTILITY_MARGIN * depth as i32 + if improving { 50 } else { 0 };
            if static_eval - rfp_margin >= beta {
                return static_eval;
            }
        }

        // Enhanced Null Move Pruning
        if depth >= NULL_MOVE_DEPTH && static_eval >= beta {
            let has_non_pawn_pieces =
                (board.combined() & board.color_combined(board.side_to_move())).0
                    != ((board.pieces(Piece::King) | board.pieces(Piece::Pawn))
                        & board.color_combined(board.side_to_move()))
                    .0;

            if has_non_pawn_pieces {
                // Dynamic reduction based on depth, eval, and improving
                let mut r = 3 + depth / 4;
                r += ((static_eval - beta) / 200).min(3) as usize;
                if !improving {
                    r += 1;
                }
                if depth > 12 {
                    r += 1;
                }

                r = r.min(depth - 1);

                if let Some(null_board) = board.null_move() {
                    let null_score = -negamax(
                        &null_board,
                        depth.saturating_sub(r),
                        -beta,
                        -beta + 1,
                        start_time,
                        ply + 1,
                        stats,
                        root_color,
                        tt,
                        max_time,
                        timeout_occurred,
                    );

                    if *timeout_occurred {
                        return evaluate(board);
                    }

                    if null_score >= beta {
                        // Null move verification for high depths
                        if depth >= 12 {
                            let verification_depth = depth.saturating_sub(r + 1);
                            let verification_score = negamax(
                                board,
                                verification_depth,
                                beta - 1,
                                beta,
                                start_time,
                                ply,
                                stats,
                                root_color,
                                tt,
                                max_time,
                                timeout_occurred,
                            );

                            if *timeout_occurred {
                                return evaluate(board);
                            }

                            if verification_score >= beta {
                                return null_score;
                            }
                        } else {
                            return null_score;
                        }
                    }
                }
            }
        }

        // ProbCut - probabilistic cut based on reduced search
        if depth >= 5 && !hash_move.is_some() {
            let probcut_beta = beta + 200;
            let probcut_depth = depth / 4 * 3;

            if probcut_depth >= 1 {
                let probcut_score = negamax(
                    board,
                    probcut_depth,
                    probcut_beta - 1,
                    probcut_beta,
                    start_time,
                    ply,
                    stats,
                    root_color,
                    tt,
                    max_time,
                    timeout_occurred,
                );

                if *timeout_occurred {
                    return evaluate(board);
                }

                if probcut_score >= probcut_beta {
                    return probcut_score;
                }
            }
        }
    }

    // Internal Iterative Deepening
    if is_pv && depth >= IID_DEPTH && hash_move.is_none() {
        let iid_depth = depth - 2 - if is_pv { 0 } else { 1 };
        negamax(
            board,
            iid_depth,
            alpha,
            beta,
            start_time,
            ply,
            stats,
            root_color,
            tt,
            max_time,
            timeout_occurred,
        );
        if !*timeout_occurred {
            hash_move = tt.get_move(board_hash);
        }
    }

    let mut moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();

    if moves.is_empty() {
        return if in_check { -30000 + ply as i32 } else { 0 };
    }

    moves = order_moves(board, moves, hash_move, ply);

    let mut best_value = -31000;
    let mut best_move = None;
    let mut moves_searched = 0;
    let mut quiet_moves_searched = 0;

    // Multi-Cut variables
    let mut cut_count = 0;
    let cut_threshold = 3 + (depth / 4);
    let multi_cut_depth = 3;

    for (i, mv) in moves.iter().enumerate() {
        if *timeout_occurred {
            break;
        }

        let is_capture = board.piece_on(mv.get_dest()).is_some();
        let is_promotion = mv.get_promotion().is_some();
        let is_quiet = !is_capture && !is_promotion;

        // Enhanced Late Move Pruning with better thresholds
        if !is_pv && !in_check && is_quiet && depth <= LMP_DEPTH {
            let mut lmp_threshold = match depth {
                1 => {
                    if improving {
                        4
                    } else {
                        3
                    }
                }
                2 => {
                    if improving {
                        7
                    } else {
                        6
                    }
                }
                3 => {
                    if improving {
                        13
                    } else {
                        12
                    }
                }
                4 => {
                    if improving {
                        20
                    } else {
                        18
                    }
                }
                _ => 25,
            };

            // Adjust threshold based on evaluation
            if static_eval + 200 < alpha {
                lmp_threshold /= 2;
            } else if static_eval > alpha + 200 {
                lmp_threshold = (lmp_threshold * 3) / 2;
            }

            if quiet_moves_searched >= lmp_threshold {
                continue;
            }
        }

        // Enhanced Futility Pruning
        if !is_pv && !in_check && is_quiet && depth <= FUTILITY_DEPTH && i > 0 {
            let mut futility_margin = FUTILITY_MARGINS[depth.min(4)];
            if !improving {
                futility_margin += 50;
            }

            let futility_value = static_eval + futility_margin;
            if futility_value <= alpha {
                continue;
            }
        }

        let new_board = board.make_move_new(*mv);
        let new_position_hash = compute_zobrist_hash(&new_board);
        update_repetition_table(new_position_hash);

        let gives_check = *new_board.checkers() != BitBoard(0);

        // Calculate extensions
        let mut extension = 0;
        if gives_check {
            extension = 1;
        } else if let Some(piece) = board.piece_on(mv.get_source()) {
            // Pawn to 7th/2nd rank extension
            if piece == Piece::Pawn {
                let dest_rank = mv.get_dest().get_rank();
                if dest_rank == Rank::Seventh || dest_rank == Rank::Second {
                    extension = 1;
                }
            }
        }

        let mut new_depth = depth - 1 + extension;
        let mut do_full_search = true;

        // Late Move Reductions with enhanced conditions
        if depth >= 3 && i >= LMR_FULL_DEPTH_MOVES && is_quiet && !gives_check {
            let mut r: usize = 1;

            // Base reduction
            if i >= 6 {
                r += 1;
            }
            if i >= 12 {
                r += 1;
            }
            if i >= 24 {
                r += 1;
            }

            // Adjust based on various factors
            if !improving {
                r += 1;
            }
            if !is_pv {
                r += 1;
            }
            if depth > 8 {
                r += 1;
            }

            // Reduce less for killer moves and good history
            let history = HISTORY_HEURISTIC.lock().unwrap();
            let history_score = *history.get(mv).unwrap_or(&0);
            if history_score > 1000 {
                r = r.saturating_sub(1);
            }

            r = r.min(LMR_REDUCTION_LIMIT).min(new_depth.saturating_sub(1));
            new_depth = new_depth.saturating_sub(r);
            do_full_search = false;
        }

        // Multi-Cut implementation
        if !is_pv && depth >= multi_cut_depth && i >= cut_threshold && !is_capture && cut_count > 0
        {
            let multi_cut_score = -negamax(
                &new_board,
                depth / 2,
                -beta,
                -beta + 1,
                start_time,
                ply + 1,
                stats,
                root_color,
                tt,
                max_time,
                timeout_occurred,
            );

            if *timeout_occurred {
                let mut rep_table = REPETITION_TABLE.lock().unwrap();
                if let Some(count) = rep_table.get_mut(&new_position_hash) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        rep_table.remove(&new_position_hash);
                    }
                }
                break;
            }

            if multi_cut_score >= beta {
                cut_count += 1;
                if cut_count >= cut_threshold {
                    let mut rep_table = REPETITION_TABLE.lock().unwrap();
                    if let Some(count) = rep_table.get_mut(&new_position_hash) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            rep_table.remove(&new_position_hash);
                        }
                    }
                    return beta;
                }
            }
        }

        // Principal Variation Search
        let score = if i == 0 {
            // Always search first move with full window
            -negamax(
                &new_board,
                new_depth,
                -beta,
                -alpha,
                start_time,
                ply + 1,
                stats,
                root_color,
                tt,
                max_time,
                timeout_occurred,
            )
        } else {
            // Scout search with null window
            let mut search_score = -negamax(
                &new_board,
                new_depth,
                -alpha - 1,
                -alpha,
                start_time,
                ply + 1,
                stats,
                root_color,
                tt,
                max_time,
                timeout_occurred,
            );

            if *timeout_occurred {
                evaluate(board)
            } else {
                // Check if reduced search needs full-depth re-search
                if !do_full_search && search_score > alpha {
                    new_depth = depth - 1 + extension;
                    search_score = -negamax(
                        &new_board,
                        new_depth,
                        -alpha - 1,
                        -alpha,
                        start_time,
                        ply + 1,
                        stats,
                        root_color,
                        tt,
                        max_time,
                        timeout_occurred,
                    );
                }

                // PV re-search with full window if score beats alpha
                if *timeout_occurred {
                    evaluate(board)
                } else if search_score > alpha {
                    // Only do full re-search if we're in PV node or score is in window
                    if is_pv || (search_score < beta) {
                        -negamax(
                            &new_board,
                            new_depth,
                            -beta,
                            -alpha,
                            start_time,
                            ply + 1,
                            stats,
                            root_color,
                            tt,
                            max_time,
                            timeout_occurred,
                        )
                    } else {
                        search_score
                    }
                } else {
                    search_score
                }
            }
        };

        // Cleanup repetition table
        {
            let mut rep_table = REPETITION_TABLE.lock().unwrap();
            if let Some(count) = rep_table.get_mut(&new_position_hash) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    rep_table.remove(&new_position_hash);
                }
            }
        }

        if *timeout_occurred {
            break;
        }

        moves_searched += 1;
        if is_quiet {
            quiet_moves_searched += 1;
        }

        if score > best_value {
            best_value = score;
            best_move = Some(*mv);
        }

        if score > alpha {
            alpha = score;

            // Update heuristics for quiet moves
            if is_quiet {
                // Update killer moves
                {
                    let mut killers = KILLER_MOVES.lock().unwrap();
                    if ply < 64 && killers.len() > ply {
                        if killers[ply].is_empty() || killers[ply][0] != *mv {
                            if killers[ply].len() >= 2 {
                                killers[ply][1] = killers[ply][0];
                            } else if killers[ply].len() == 1 {
                                let first_killer = killers[ply][0];
                                killers[ply].push(first_killer);
                            }
                            if killers[ply].is_empty() {
                                killers[ply].push(*mv);
                            } else {
                                killers[ply][0] = *mv;
                            }
                        }
                    }
                }

                // Update history heuristic with depth-based bonus
                {
                    let mut history = HISTORY_HEURISTIC.lock().unwrap();
                    let bonus = (depth * depth) as i32;
                    *history.entry(*mv).or_insert(0) += bonus;

                    // Cap history values to prevent overflow
                    if let Some(value) = history.get_mut(mv) {
                        if *value > 10000 {
                            *value = 10000;
                        }
                    }
                }

                // Reduce history for previous quiet moves that failed
                for prev_mv in &moves[0..i] {
                    if board.piece_on(prev_mv.get_dest()).is_none()
                        && prev_mv.get_promotion().is_none()
                    {
                        let mut history = HISTORY_HEURISTIC.lock().unwrap();
                        if let Some(value) = history.get_mut(prev_mv) {
                            *value -= (depth * depth / 4) as i32;
                            if *value < -1000 {
                                *value = -1000;
                            }
                        }
                    }
                }
            }
        }

        // Beta cutoff
        if alpha >= beta {
            break;
        }
    }

    // Store result in transposition table
    if !*timeout_occurred && moves_searched > 0 {
        let flag = if best_value <= original_alpha {
            TTFlag::Upper
        } else if best_value >= beta {
            TTFlag::Lower
        } else {
            TTFlag::Exact
        };

        tt.store(board_hash, depth, best_value, flag, best_move);
    }

    best_value
}
fn iterative_deepening(board: &Board, max_time: f64) -> Option<ChessMove> {
    let start_time = Instant::now();
    let mut best_move = None;
    let mut best_score = 0;
    let root_color = board.side_to_move();

    // Check for immediate checkmate
    let movegen = MoveGen::new_legal(board);
    for mv in movegen {
        let new_board = board.make_move_new(mv);
        if new_board.status() == BoardStatus::Checkmate {
            return Some(mv);
        }
    }

    let mut tt_guard = TRANSPOSITION_TABLE.lock().unwrap();
    for depth in 1..=MAX_DEPTH {
        if start_time.elapsed().as_secs_f64() > max_time * 0.8 {
            break;
        }

        let mut stats: HashMap<&str, usize> = HashMap::new();
        let mut timeout_occurred = false;

        // Determine initial aspiration window bounds
        let (initial_alpha, initial_beta) = if depth <= 4 || best_move.is_none() {
            // Full window for early depths or when no best move exists
            (-31000, 31000)
        } else {
            // Narrow aspiration window centered on previous iteration's score
            (best_score - INITIAL_WINDOW, best_score + INITIAL_WINDOW)
        };

        let mut alpha = initial_alpha;
        let mut beta = initial_beta;
        let mut window_size = INITIAL_WINDOW;
        let mut search_iterations = 0;
        const MAX_ASPIRATION_ITERATIONS: usize = 6;

        // Aspiration window re-search loop
        loop {
            search_iterations += 1;

            let score = negamax(
                board,
                depth,
                alpha,
                beta,
                start_time,
                0,
                &mut stats,
                root_color,
                &mut *tt_guard,
                max_time,
                &mut timeout_occurred,
            );

            if timeout_occurred {
                break;
            }

            // Check if we failed low or high
            if score <= alpha {
                // FAIL LOW: True score is <= alpha
                // Standard PVS: Open the window downward, keep beta
                if STATS {
                    println!(
                        "info string Aspiration fail-low at depth {} (score {} <= alpha {})",
                        depth, score, alpha
                    );
                }

                // Widen window exponentially
                window_size *= 2;

                if window_size >= MAX_WINDOW || search_iterations >= MAX_ASPIRATION_ITERATIONS {
                    // Give up on aspiration, do full search
                    alpha = -31000;
                    beta = initial_beta.max(score + INITIAL_WINDOW);
                } else {
                    // Re-search with lowered alpha, keeping beta stable
                    alpha = (score - window_size).max(-31000);
                    // Beta remains at initial_beta or can be narrowed slightly
                    beta = initial_beta;
                }
            } else if score >= beta {
                // FAIL HIGH: True score is >= beta
                // Standard PVS: Open the window upward, keep alpha
                if STATS {
                    println!(
                        "info string Aspiration fail-high at depth {} (score {} >= beta {})",
                        depth, score, beta
                    );
                }

                // Widen window exponentially
                window_size *= 2;

                if window_size >= MAX_WINDOW || search_iterations >= MAX_ASPIRATION_ITERATIONS {
                    // Give up on aspiration, do full search
                    alpha = initial_alpha.min(score - INITIAL_WINDOW);
                    beta = 31000;
                } else {
                    // Re-search with raised beta, keeping alpha stable
                    beta = (score + window_size).min(31000);
                    // Alpha remains at initial_alpha or can be narrowed slightly
                    alpha = initial_alpha;
                }
            } else {
                // SUCCESS: Score is within [alpha, beta]
                best_score = score;

                let board_hash = compute_zobrist_hash(board);
                if let Some(mv) = tt_guard.get_move(board_hash) {
                    let movegen = MoveGen::new_legal(board);
                    if movegen.into_iter().any(|legal_move| legal_move == mv) {
                        best_move = Some(mv);

                        if STATS {
                            // Extract and display principal variation
                            let mut pv = Vec::new();
                            let mut temp_board = *board;
                            let mut pv_depth = 0;

                            for _ in 0..depth.min(20) {
                                let h = compute_zobrist_hash(&temp_board);
                                if let Some(pv_move) = tt_guard.get_move(h) {
                                    let movegen = MoveGen::new_legal(&temp_board);
                                    if movegen.into_iter().any(|legal_move| legal_move == pv_move) {
                                        pv.push(pv_move);
                                        temp_board = temp_board.make_move_new(pv_move);
                                        pv_depth += 1;
                                    } else {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }

                            let nodes = *stats.get("nodes").unwrap_or(&0);
                            let time_ms = (start_time.elapsed().as_secs_f64() * 1000.0) as u64;
                            let nps = if time_ms > 0 {
                                nodes * 1000 / time_ms as usize
                            } else {
                                0
                            };
                            let pv_string = pv
                                .iter()
                                .map(|m| format!("{}", m))
                                .collect::<Vec<_>>()
                                .join(" ");

                            // Calculate hash usage percentage
                            let hashfull =
                                (tt_guard.table.len() * 1000 / tt_guard.size.max(1)).min(1000);

                            println!(
                                "info depth {} seldepth {} score cp {} nodes {} nps {} time {} hashfull {} pv {}",
                                depth, 
                                pv_depth.max(depth), 
                                best_score, 
                                nodes, 
                                nps, 
                                time_ms,
                                hashfull,
                                pv_string
                            );
                        }

                        // Early exit if mate found
                        if best_score.abs() > 29000 {
                            break;
                        }
                    }
                }
                break; // Exit aspiration window loop on success
            }

            // Safety: prevent infinite loops
            if timeout_occurred || search_iterations >= MAX_ASPIRATION_ITERATIONS * 2 {
                if STATS && search_iterations >= MAX_ASPIRATION_ITERATIONS * 2 {
                    println!(
                        "info string Aspiration window abandoned after {} iterations",
                        search_iterations
                    );
                }
                break;
            }
        }

        if timeout_occurred {
            break;
        }
    }

    best_move.or_else(|| {
        let moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        moves.first().copied()
    })
}
fn compute_zobrist_hash(board: &Board) -> u64 {
    let mut h = 0;
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
            h ^= ZOBRIST_PIECES[&(piece, piece_color)][sq_idx as usize];
        }
    }
    let mut castling_rights = 0;
    if board.castle_rights(Color::White).has_kingside() {
        castling_rights |= 1;
    }
    if board.castle_rights(Color::White).has_queenside() {
        castling_rights |= 2;
    }
    if board.castle_rights(Color::Black).has_kingside() {
        castling_rights |= 4;
    }
    if board.castle_rights(Color::Black).has_queenside() {
        castling_rights |= 8;
    }
    h ^= ZOBRIST_CASTLING[castling_rights as usize];
    if let Some(ep_sq) = board.en_passant() {
        h ^= ZOBRIST_EP[ep_sq.get_file() as usize];
    }
    if board.side_to_move() == Color::Black {
        h ^= *ZOBRIST_TURN;
    }
    h
}
struct UCIEngine {
    board: Board,
    debug: bool,
    position_history: Vec<u64>,
    hash_size: usize,
}

impl UCIEngine {
    fn new() -> Self {
        UCIEngine {
            board: Board::default(),
            debug: false,
            position_history: Vec::new(),
            hash_size: 256,
        }
    }

    fn handle_uci(&self) {
        println!("id name RustKnightv2.1");
        println!("id author Anish");
        println!("option name Hash type spin default 256 min 1 max 4096");
        println!("uciok");
    }

    fn handle_isready(&self) {
        println!("readyok");
    }
    fn handle_setoption(&mut self, tokens: &[&str]) {
        if tokens.len() >= 5 && tokens[1] == "name" && tokens[3] == "value" {
            let option_name = tokens[2].to_lowercase();
            match option_name.as_str() {
                "hash" => {
                    if let Ok(size) = tokens[4].parse::<usize>() {
                        let clamped_size = size.max(1).min(4096);
                        self.hash_size = clamped_size;
                        {
                            let mut tt = TRANSPOSITION_TABLE.lock().unwrap();
                            *tt = TranspositionTable::new(clamped_size);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    fn handle_position(&mut self, tokens: &[&str]) {
        if tokens.len() < 2 {
            return;
        }

        self.position_history.clear();

        match tokens[1] {
            "startpos" => {
                self.board = Board::default();
                self.position_history
                    .push(compute_zobrist_hash(&self.board));

                if tokens.len() > 2 && tokens[2] == "moves" {
                    for move_str in &tokens[3..] {
                        if let Ok(chess_move) = ChessMove::from_str(move_str) {
                            self.board = self.board.make_move_new(chess_move);
                            self.position_history
                                .push(compute_zobrist_hash(&self.board));
                        }
                    }
                }
            }
            "fen" => {
                if tokens.len() < 8 {
                    return;
                }
                let fen = tokens[2..8].join(" ");
                if let Ok(board) = Board::from_str(&fen) {
                    self.board = board;
                    self.position_history
                        .push(compute_zobrist_hash(&self.board));

                    let mut moves_start = None;
                    for (i, &token) in tokens.iter().enumerate() {
                        if token == "moves" {
                            moves_start = Some(i + 1);
                            break;
                        }
                    }

                    if let Some(start) = moves_start {
                        for move_str in &tokens[start..] {
                            if let Ok(chess_move) = ChessMove::from_str(move_str) {
                                self.board = self.board.make_move_new(chess_move);
                                self.position_history
                                    .push(compute_zobrist_hash(&self.board));
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        clear_repetition_table();
        for &hash in &self.position_history {
            update_repetition_table(hash);
        }
    }

    fn calculate_time_budget(
        &self,
        wtime: Option<u64>,
        btime: Option<u64>,
        winc: Option<u64>,
        binc: Option<u64>,
        movetime: Option<u64>,
    ) -> f64 {
        if let Some(mt) = movetime {
            return (mt as f64) / 1000.0;
        }

        let (time_left, increment) = match self.board.side_to_move() {
            Color::White => (wtime.unwrap_or(0), winc.unwrap_or(0)),
            Color::Black => (btime.unwrap_or(0), binc.unwrap_or(0)),
        };

        if time_left == 0 {
            return 1.0;
        }

        let base_time = (time_left as f64) / 1000.0;
        let inc_time = (increment as f64) / 1000.0;

        let moves_to_go = 40;
        let time_per_move = (base_time / moves_to_go as f64) + (inc_time * 0.8);

        time_per_move.max(0.01).min(base_time * 0.1)
    }

    fn handle_go(&self, tokens: &[&str]) {
        let mut wtime = None;
        let mut btime = None;
        let mut winc = None;
        let mut binc = None;
        let mut movetime = None;
        let mut infinite = false;

        let mut i = 1;
        while i < tokens.len() {
            match tokens[i] {
                "wtime" => {
                    if i + 1 < tokens.len() {
                        wtime = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "btime" => {
                    if i + 1 < tokens.len() {
                        btime = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "winc" => {
                    if i + 1 < tokens.len() {
                        winc = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "binc" => {
                    if i + 1 < tokens.len() {
                        binc = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "movetime" => {
                    if i + 1 < tokens.len() {
                        movetime = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "infinite" => {
                    infinite = true;
                    i += 1;
                }
                _ => i += 1,
            }
        }

        let max_time = if infinite {
            10000000.0
        } else {
            self.calculate_time_budget(wtime, btime, winc, binc, movetime)
        };

        if let Some(best_move) = iterative_deepening(&self.board, max_time) {
            println!("bestmove {}", best_move);
        } else {
            println!("bestmove 0000");
        }
    }

    fn handle_debug(&mut self, tokens: &[&str]) {
        if tokens.len() > 1 {
            self.debug = tokens[1] == "on";
        }
    }

    fn run(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(line) => line,
                Err(_) => break,
            };

            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            match tokens[0] {
                "uci" => self.handle_uci(),
                "debug" => self.handle_debug(&tokens),
                "isready" => self.handle_isready(),
                "setoption" => self.handle_setoption(&tokens),
                "register" => {}
                "ucinewgame" => {
                    self.board = Board::default();
                    self.position_history.clear();
                    self.position_history
                        .push(compute_zobrist_hash(&self.board));
                    {
                        let mut tt = TRANSPOSITION_TABLE.lock().unwrap();
                        *tt = TranspositionTable::new(self.hash_size);
                        tt.clear();
                    }
                    {
                        let mut killers = KILLER_MOVES.lock().unwrap();
                        killers.clear();
                        killers.resize(MAX_DEPTH, Vec::new());
                    }
                    {
                        let mut history = HISTORY_HEURISTIC.lock().unwrap();
                        history.clear();
                    }
                    clear_repetition_table();
                }
                "position" => self.handle_position(&tokens),
                "go" => self.handle_go(&tokens),
                "stop" => {}
                "ponderhit" => {}
                "quit" => break,
                _ => {
                    if self.debug {
                        eprintln!("Unknown command: {}", tokens[0]);
                    }
                }
            }
        }
    }
}

fn main() {
    let mut engine = UCIEngine::new();
    engine.run();
}

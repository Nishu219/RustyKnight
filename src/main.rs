use chess::{
    Board, ChessMove, Color, File, MoveGen, Piece, Rank, Square, BitBoard, BoardStatus,
};
use rand::Rng;
use std::collections::HashMap;
use std::sync::Mutex;
use lazy_static::lazy_static;
use std::io::{self, BufRead};
use std::time::Instant;
use std::str::FromStr;
const MAX_DEPTH: usize = 40;
const TIME_LIMIT: f64 = 5.0;
const ASPIRATION_WINDOW: i32 = 30;
const DELTA_MARGIN: i32 = 200;
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
const LMP_MOVE_COUNTS: [usize; 5] = [0, 3, 6, 12, 18];
const SEE_QUIET_THRESHOLD: i32 = -60;
const SEE_CAPTURE_THRESHOLD: i32 = -100;
const MOBILITY_BONUS: [i32; 5] = [0, 4, 5, 2, 1];
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
        for piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
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
            sub.insert("mg", vec![
                0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0
            ]);
            sub.insert("eg", vec![
                 0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0
            ]);
            sub
        });
        m.insert(Piece::Knight, {
            let mut sub = HashMap::new();
            sub.insert("mg", vec![
                -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23
            ]);
            sub.insert("eg", vec![
                -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64
            ]);
            sub
        });
        m.insert(Piece::Bishop, {
            let mut sub = HashMap::new();
            sub.insert("mg", vec![
                -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21
            ]);
            sub.insert("eg", vec![
                -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17
            ]);
            sub
        });
        m.insert(Piece::Rook, {
            let mut sub = HashMap::new();
            sub.insert("mg", vec![
                32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26
            ]);
            sub.insert("eg", vec![
                13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20
            ]);
            sub
        });
        m.insert(Piece::Queen, {
            let mut sub = HashMap::new();
            sub.insert("mg", vec![
                -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50
            ]);
            sub.insert("eg", vec![
                -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41
            ]);
            sub
        });
        m.insert(Piece::King, {
            let mut sub = HashMap::new();
            sub.insert("mg", vec![
                -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14
            ]);
            sub.insert("eg", vec![
                -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
            ]);
            sub
        });
        m
    };
    static ref EVAL_CACHE: Mutex<HashMap<u64, i32>> = Mutex::new(HashMap::new());
    static ref MATERIAL_HASH_TABLE: Mutex<MaterialHashTable> = Mutex::new(MaterialHashTable::new(16));
    static ref KILLER_MOVES: Mutex<Vec<Vec<ChessMove>>> = Mutex::new(Vec::new());
    static ref HISTORY_HEURISTIC: Mutex<HashMap<ChessMove, i32>> = Mutex::new(HashMap::new());
    static ref TRANSPOSITION_TABLE: Mutex<TranspositionTable> = Mutex::new(TranspositionTable::new(1024));
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
            size: (size_mb * 1024 * 1024) / 32,
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
    fn store(&mut self, key: u64, depth: usize, value: i32, flag: TTFlag, move_: Option<ChessMove>) {
        if self.table.len() >= self.size {
            let oldest_key = *self.table.iter()
                .min_by_key(|(_, entry)| entry.age)
                .map(|(k, _)| k)
                .unwrap();
            self.table.remove(&oldest_key);
        }
        self.table.insert(key, TTEntry {
            depth,
            value,
            flag,
            move_,
            age: self.age,
        });
    }
    fn get_move(&self, key: u64) -> Option<ChessMove> {
        self.table.get(&key).and_then(|entry| entry.move_)
    }
}
const PAWN_HASH_SIZE: usize = 16384;
static mut PAWN_HASH_TABLE: [[(u64, i32, usize); 2]; PAWN_HASH_SIZE] = [[(0, 0, 0); 2]; PAWN_HASH_SIZE];
static mut PAWN_HASH_AGE: usize = 0;
fn compute_pawn_hash(board: &Board) -> u64 {
    let mut h = 0;
    for sq_idx in 0..64 {
        let square = unsafe { Square::new(sq_idx) }; // Square::new is unsafe
        if let Some(piece) = board.piece_on(square) {
            if piece == Piece::Pawn {
                let piece_color = if (*board.color_combined(Color::White) & BitBoard::from_square(square)) != BitBoard(0) { // Use != BitBoard(0)
                    Color::White
                } else {
                    Color::Black
                };
                h ^= ZOBRIST_PIECES[&(Piece::Pawn, piece_color)][sq_idx as usize];
            }
        }
    }
    h
}
struct MaterialHashTable {
    table: Vec<[(u64, i32, usize); 2]>,
    age: usize,
    size: usize,
}
impl MaterialHashTable {
    fn new(size_mb: usize) -> Self {
        let size = (size_mb * 1024 * 1024) / 24;
        Self {
            table: vec![[ (0, 0, 0); 2 ]; size],
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

fn compute_material_key(board: &Board) -> u64 {
    let mut key = 0u64;
    for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
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
        let white_count = (board.pieces(piece) & board.color_combined(Color::White)).popcnt() as i32;
        let black_count = (board.pieces(piece) & board.color_combined(Color::Black)).popcnt() as i32;
        material += white_count * value;
        material -= black_count * value;
    }
    material
}
fn phase_value(board: &Board) -> i32 {
    let mut phase = 0;
    for sq_idx in 0..64 {
        let square = unsafe { Square::new(sq_idx) }; // Square::new is unsafe
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
fn is_endgame(board: &Board) -> bool {
    phase_value(board) < 8
}
fn see(board: &Board, mv: ChessMove) -> i32 {
    let to_sq = mv.get_dest();
    let from_sq = mv.get_source();
    let piece = match board.piece_on(from_sq) {
        Some(p) => p,
        None => return 0,
    };
    let mut captured_value = 0;
    if let Some(captured) = board.piece_on(to_sq) {
        captured_value = VALUES[&captured];
    }
    if let Some(promotion) = mv.get_promotion() {
        captured_value += VALUES[&promotion] - VALUES[&Piece::Pawn];
    }
    let attackers_bb = board.combined() ^ BitBoard::from_square(from_sq);
    let mut side = !board.side_to_move();
    let mut gains = vec![captured_value];
    let mut depth = 1;
    let mut cur_attackers = attackers_bb;
    while depth <= 32 {
        let mut least_valuable_attacker: Option<(Piece, Square)> = None;
        let mut min_value = i32::MAX;
        for &pt in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
            let pieces_bb = *board.pieces(pt);
            let candidates_bb = cur_attackers & pieces_bb & board.color_combined(side);
            if candidates_bb != BitBoard(0) { // Use != BitBoard(0)
                if let Some(attacker_sq) = candidates_bb.into_iter().next() {
                     let value = VALUES[&pt];
                     if value < min_value {
                         min_value = value;
                         least_valuable_attacker = Some((pt, attacker_sq));
                     }
                }
            }
        }
        if let Some((pt, attacker_sq)) = least_valuable_attacker {
            gains.push(VALUES[&pt] - gains[depth - 1]);
            cur_attackers ^= BitBoard::from_square(attacker_sq);
            side = !side;
            depth += 1;
        } else {
            break;
        }
    }
    for i in (0..gains.len() - 1).rev() {
        gains[i] = -std::cmp::max(-gains[i], gains[i + 1]);
    }
    gains[0]
}

fn evaluate_pawn_structure(board: &Board) -> i32 {
    unsafe {
        let pawn_key = compute_pawn_hash(board);
        let index = (pawn_key as usize) % PAWN_HASH_SIZE;
        let bucket = &PAWN_HASH_TABLE[index];
        if bucket[0].0 == pawn_key {
            return bucket[0].1;
        }
        if bucket[1].0 == pawn_key {
            return bucket[1].1;
        }
        
        let mut score = 0;
        let white_pawns_bb = *board.pieces(Piece::Pawn) & board.color_combined(Color::White);
        let black_pawns_bb = *board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
        let all_pawns_bb = white_pawns_bb | black_pawns_bb;
        let all_pieces_bb = board.combined();
        
        let mut white_file_masks = [BitBoard(0); 8];
        let mut black_file_masks = [BitBoard(0); 8];
        let mut white_rank_masks = [BitBoard(0); 8];
        let mut black_rank_masks = [BitBoard(0); 8];
        
        for file_idx in 0..8 {
            white_file_masks[file_idx] = chess::get_file(File::from_index(file_idx)) & white_pawns_bb;
            black_file_masks[file_idx] = chess::get_file(File::from_index(file_idx)) & black_pawns_bb;
        }
        
        for rank_idx in 0..8 {
            white_rank_masks[rank_idx] = chess::get_rank(Rank::from_index(rank_idx)) & white_pawns_bb;
            black_rank_masks[rank_idx] = chess::get_rank(Rank::from_index(rank_idx)) & black_pawns_bb;
        }
        
        let mut white_pawn_islands = 0;
        let mut black_pawn_islands = 0;
        let mut in_white_island = false;
        let mut in_black_island = false;
        
        for file_idx in 0..8 {
            let has_white_pawn = white_file_masks[file_idx] != BitBoard(0);
            let has_black_pawn = black_file_masks[file_idx] != BitBoard(0);
            
            if has_white_pawn && !in_white_island {
                white_pawn_islands += 1;
                in_white_island = true;
            } else if !has_white_pawn {
                in_white_island = false;
            }
            
            if has_black_pawn && !in_black_island {
                black_pawn_islands += 1;
                in_black_island = true;
            } else if !has_black_pawn {
                in_black_island = false;
            }
        }
        
        score -= (white_pawn_islands - 1) * 8;
        score += (black_pawn_islands - 1) * 8;
        
        let center_files = [File::D, File::E];
        let mut white_center_pawns = 0;
        let mut black_center_pawns = 0;
        
        for &file in &center_files {
            let center_file_bb = chess::get_file(file);
            let center_ranks = [Rank::Fourth, Rank::Fifth];
            for &rank in &center_ranks {
                let center_square = Square::make_square(rank, file);
                if (white_pawns_bb & BitBoard::from_square(center_square)) != BitBoard(0) {
                    white_center_pawns += 1;
                }
                if (black_pawns_bb & BitBoard::from_square(center_square)) != BitBoard(0) {
                    black_center_pawns += 1;
                }
            }
        }
        
        score += white_center_pawns * 15;
        score -= black_center_pawns * 15;
        
        let mut white_holes = BitBoard(0);
        let mut black_holes = BitBoard(0);
        
        for file_idx in 0..8 {
            let file = File::from_index(file_idx);
            for rank_idx in 2..6 {
                let rank = Rank::from_index(rank_idx);
                let sq = Square::make_square(rank, file);
                let sq_bb = BitBoard::from_square(sq);
                
                let mut is_white_hole = true;
                let mut is_black_hole = true;
                
                for adj_file_idx in [(file_idx as i32 - 1), (file_idx as i32 + 1)] {
                    if adj_file_idx >= 0 && adj_file_idx < 8 {
                        let adj_file = File::from_index(adj_file_idx as usize);
                        for r in 0..rank_idx {
                            let adj_sq = Square::make_square(Rank::from_index(r), adj_file);
                            if (white_pawns_bb & BitBoard::from_square(adj_sq)) != BitBoard(0) {
                                is_black_hole = false;
                            }
                        }
                        for r in (rank_idx + 1)..8 {
                            let check_sq = Square::make_square(Rank::from_index(r), adj_file);
                            if (black_pawns_bb & BitBoard::from_square(check_sq)) != BitBoard(0) {
                                is_white_hole = false;
                            }
                        }
                    }
                }
                
                if is_white_hole && (all_pawns_bb & sq_bb) == BitBoard(0) {
                    white_holes |= sq_bb;
                }
                if is_black_hole && (all_pawns_bb & sq_bb) == BitBoard(0) {
                    black_holes |= sq_bb;
                }
            }
        }
        
        score += white_holes.popcnt() as i32 * 12;
        score -= black_holes.popcnt() as i32 * 12;
        
        let mut white_pawn_majority = 0;
        let mut black_pawn_majority = 0;
        
        let queenside_files = [0, 1, 2, 3];
        let kingside_files = [4, 5, 6, 7];
        
        let mut white_queenside = 0;
        let mut black_queenside = 0;
        let mut white_kingside = 0;
        let mut black_kingside = 0;
        
        for &file_idx in &queenside_files {
            white_queenside += white_file_masks[file_idx].popcnt();
            black_queenside += black_file_masks[file_idx].popcnt();
        }
        
        for &file_idx in &kingside_files {
            white_kingside += white_file_masks[file_idx].popcnt();
            black_kingside += black_file_masks[file_idx].popcnt();
        }
        
        if white_queenside > black_queenside {
            white_pawn_majority += (white_queenside - black_queenside) as i32;
        }
        if white_kingside > black_kingside {
            white_pawn_majority += (white_kingside - black_kingside) as i32;
        }
        if black_queenside > white_queenside {
            black_pawn_majority += (black_queenside - white_queenside) as i32;
        }
        if black_kingside > white_kingside {
            black_pawn_majority += (black_kingside - white_kingside) as i32;
        }
        
        score += white_pawn_majority * 8;
        score -= black_pawn_majority * 8;
        
        for sq in white_pawns_bb {
            let file = sq.get_file();
            let rank_idx = sq.get_rank() as usize;
            let file_idx = file as usize;
            
            let mut is_passed = true;
            let mut is_backward = false;
            let mut is_candidate = false;
            let mut is_hidden_passed = false;
            let mut is_sentry = false;
            let mut is_faker = false;
            let mut is_weak = false;
            let mut has_support = false;
            let mut is_connected = false;
            let mut blocking_pieces = 0;
            
            for enemy_sq in black_pawns_bb {
                let enemy_file = enemy_sq.get_file();
                let enemy_rank_idx = enemy_sq.get_rank() as usize;
                let enemy_file_idx = enemy_file as usize;
                
                if (enemy_file_idx as i32 - file_idx as i32).abs() <= 1 && enemy_rank_idx > rank_idx {
                    is_passed = false;
                    
                    if enemy_rank_idx == rank_idx + 1 && (enemy_file_idx as i32 - file_idx as i32).abs() == 1 {
                        for adj_file_idx in [file_idx as i32 - 1, file_idx as i32 + 1] {
                            if adj_file_idx >= 0 && adj_file_idx < 8 {
                                let adj_file = File::from_index(adj_file_idx as usize);
                                for r in 0..rank_idx {
                                    let adj_sq = Square::make_square(Rank::from_index(r), adj_file);
                                    if (white_pawns_bb & BitBoard::from_square(adj_sq)) != BitBoard(0) {
                                        has_support = true;
                                        break;
                                    }
                                }
                                if has_support {
                                    break;
                                }
                            }
                        }
                        if !has_support {
                            is_backward = true;
                        }
                    }
                }
            }
            
            for adj_file_idx in [file_idx as i32 - 1, file_idx as i32 + 1] {
                if adj_file_idx >= 0 && adj_file_idx < 8 {
                    let adj_file = File::from_index(adj_file_idx as usize);
                    
                    if (white_file_masks[adj_file_idx as usize] & 
                        chess::get_rank(Rank::from_index(rank_idx))) != BitBoard(0) {
                        is_connected = true;
                    }
                    
                    for r in (rank_idx + 1)..8 {
                        let check_sq = Square::make_square(Rank::from_index(r), adj_file);
                        if (white_pawns_bb & BitBoard::from_square(check_sq)) != BitBoard(0) {
                            has_support = true;
                        }
                    }
                }
            }
            
            if !is_passed {
                let mut attackers = 0;
                let mut defenders = 0;
                
                for enemy_sq in black_pawns_bb {
                    let enemy_file_idx = enemy_sq.get_file() as usize;
                    let enemy_rank_idx = enemy_sq.get_rank() as usize;
                    
                    if (enemy_file_idx as i32 - file_idx as i32).abs() <= 1 && 
                       enemy_rank_idx > rank_idx {
                        attackers += 1;
                    }
                }
                
                for friend_sq in white_pawns_bb {
                    let friend_file_idx = friend_sq.get_file() as usize;
                    let friend_rank_idx = friend_sq.get_rank() as usize;
                    
                    if (friend_file_idx as i32 - file_idx as i32).abs() <= 1 && 
                       friend_rank_idx >= rank_idx {
                        defenders += 1;
                    }
                }
                
                if defenders >= attackers {
                    is_candidate = true;
                }
            }
            
            for check_rank in (rank_idx + 1)..8 {
                let check_sq = Square::make_square(Rank::from_index(check_rank), file);
                if (all_pieces_bb & BitBoard::from_square(check_sq)) != BitBoard(0) {
                    blocking_pieces += 1;
                }
            }
            
            if blocking_pieces > 0 && is_passed {
                is_hidden_passed = true;
            }
            
            if rank_idx >= 4 && !is_passed {
                let mut can_advance = true;
                for r in (rank_idx + 1)..8 {
                    let advance_sq = Square::make_square(Rank::from_index(r), file);
                    if (black_pawns_bb & BitBoard::from_square(advance_sq)) != BitBoard(0) {
                        can_advance = false;
                        break;
                    }
                }
                if can_advance {
                    is_sentry = true;
                }
            }
            
            if rank_idx <= 3 && !has_support && !is_connected {
                let mut enemy_attacks = false;
                for enemy_sq in black_pawns_bb {
                    let enemy_file_idx = enemy_sq.get_file() as usize;
                    let enemy_rank_idx = enemy_sq.get_rank() as usize;
                    
                    if (enemy_file_idx as i32 - file_idx as i32).abs() == 1 && 
                       enemy_rank_idx == rank_idx + 1 {
                        enemy_attacks = true;
                        break;
                    }
                }
                if enemy_attacks {
                    is_faker = true;
                }
            }
            
            if is_backward || (!has_support && !is_connected) {
                is_weak = true;
            }
            
            let file_bb = chess::get_file(file);
            let doubled = (white_pawns_bb & file_bb).popcnt() > 1;
            
            let mut isolated = true;
            for adj_file_idx in [file_idx as i32 - 1, file_idx as i32 + 1] {
                if adj_file_idx >= 0 && adj_file_idx < 8 {
                    let adj_file_bb = chess::get_file(File::from_index(adj_file_idx as usize));
                    if (white_pawns_bb & adj_file_bb) != BitBoard(0) {
                        isolated = false;
                        break;
                    }
                }
            }
            
            if is_passed {
                let passed_bonus = [0, 5, 10, 20, 35, 60, 100, 200][rank_idx];
                score += passed_bonus;
                
                if is_hidden_passed {
                    score += passed_bonus / 2;
                }
            }
            
            if is_candidate {
                let candidate_bonus = [0, 2, 4, 8, 16, 32, 64, 128][rank_idx];
                score += candidate_bonus;
            }
            
            if doubled {
                score -= 15;
            }
            
            if isolated {
                score -= 10;
                if file_idx == 0 || file_idx == 7 {
                    score -= 5;
                }
            }
            
            if is_backward {
                score -= 25;
            }
            
            if is_connected {
                score += 8;
                if rank_idx >= 4 {
                    score += (rank_idx - 3) as i32 * 4;
                }
            }
            
            if is_sentry {
                score += 12;
            }
            
            if is_faker {
                score -= 15;
            }
            
            if is_weak {
                score -= 8;
            }
            
            if blocking_pieces > 0 {
                score -= blocking_pieces * 3;
            }
            
            if white_file_masks[file_idx].popcnt() == 1 && 
               (file_idx == 0 || white_file_masks[file_idx - 1] == BitBoard(0)) &&
               (file_idx == 7 || white_file_masks[file_idx + 1] == BitBoard(0)) {
                score -= 5;
            }
        }
        
        for sq in black_pawns_bb {
            let file = sq.get_file();
            let rank_idx = sq.get_rank() as usize;
            let file_idx = file as usize;
            
            let mut is_passed = true;
            let mut is_backward = false;
            let mut is_candidate = false;
            let mut is_hidden_passed = false;
            let mut is_sentry = false;
            let mut is_faker = false;
            let mut is_weak = false;
            let mut has_support = false;
            let mut is_connected = false;
            let mut blocking_pieces = 0;
            
            for enemy_sq in white_pawns_bb {
                let enemy_file = enemy_sq.get_file();
                let enemy_rank_idx = enemy_sq.get_rank() as usize;
                let enemy_file_idx = enemy_file as usize;
                
                if (enemy_file_idx as i32 - file_idx as i32).abs() <= 1 && enemy_rank_idx < rank_idx {
                    is_passed = false;
                    
                    if enemy_rank_idx == rank_idx - 1 && (enemy_file_idx as i32 - file_idx as i32).abs() == 1 {
                        for adj_file_idx in [file_idx as i32 - 1, file_idx as i32 + 1] {
                            if adj_file_idx >= 0 && adj_file_idx < 8 {
                                let adj_file = File::from_index(adj_file_idx as usize);
                                for r in (rank_idx + 1)..8 {
                                    let adj_sq = Square::make_square(Rank::from_index(r), adj_file);
                                    if (black_pawns_bb & BitBoard::from_square(adj_sq)) != BitBoard(0) {
                                        has_support = true;
                                        break;
                                    }
                                }
                                if has_support {
                                    break;
                                }
                            }
                        }
                        if !has_support {
                            is_backward = true;
                        }
                    }
                }
            }
            
            for adj_file_idx in [file_idx as i32 - 1, file_idx as i32 + 1] {
                if adj_file_idx >= 0 && adj_file_idx < 8 {
                    let adj_file = File::from_index(adj_file_idx as usize);
                    
                    if (black_file_masks[adj_file_idx as usize] & 
                        chess::get_rank(Rank::from_index(rank_idx))) != BitBoard(0) {
                        is_connected = true;
                    }
                    
                    for r in 0..rank_idx {
                        let check_sq = Square::make_square(Rank::from_index(r), adj_file);
                        if (black_pawns_bb & BitBoard::from_square(check_sq)) != BitBoard(0) {
                            has_support = true;
                        }
                    }
                }
            }
            
            if !is_passed {
                let mut attackers = 0;
                let mut defenders = 0;
                
                for enemy_sq in white_pawns_bb {
                    let enemy_file_idx = enemy_sq.get_file() as usize;
                    let enemy_rank_idx = enemy_sq.get_rank() as usize;
                    
                    if (enemy_file_idx as i32 - file_idx as i32).abs() <= 1 && 
                       enemy_rank_idx < rank_idx {
                        attackers += 1;
                    }
                }
                
                for friend_sq in black_pawns_bb {
                    let friend_file_idx = friend_sq.get_file() as usize;
                    let friend_rank_idx = friend_sq.get_rank() as usize;
                    
                    if (friend_file_idx as i32 - file_idx as i32).abs() <= 1 && 
                       friend_rank_idx <= rank_idx {
                        defenders += 1;
                    }
                }
                
                if defenders >= attackers {
                    is_candidate = true;
                }
            }
            
            for check_rank in 0..rank_idx {
                let check_sq = Square::make_square(Rank::from_index(check_rank), file);
                if (all_pieces_bb & BitBoard::from_square(check_sq)) != BitBoard(0) {
                    blocking_pieces += 1;
                }
            }
            
            if blocking_pieces > 0 && is_passed {
                is_hidden_passed = true;
            }
            
            if rank_idx <= 3 && !is_passed {
                let mut can_advance = true;
                for r in 0..rank_idx {
                    let advance_sq = Square::make_square(Rank::from_index(r), file);
                    if (white_pawns_bb & BitBoard::from_square(advance_sq)) != BitBoard(0) {
                        can_advance = false;
                        break;
                    }
                }
                if can_advance {
                    is_sentry = true;
                }
            }
            
            if rank_idx >= 4 && !has_support && !is_connected {
                let mut enemy_attacks = false;
                for enemy_sq in white_pawns_bb {
                    let enemy_file_idx = enemy_sq.get_file() as usize;
                    let enemy_rank_idx = enemy_sq.get_rank() as usize;
                    
                    if (enemy_file_idx as i32 - file_idx as i32).abs() == 1 && 
                       enemy_rank_idx == rank_idx - 1 {
                        enemy_attacks = true;
                        break;
                    }
                }
                if enemy_attacks {
                    is_faker = true;
                }
            }
            
            if is_backward || (!has_support && !is_connected) {
                is_weak = true;
            }
            
            let file_bb = chess::get_file(file);
            let doubled = (black_pawns_bb & file_bb).popcnt() > 1;
            
            let mut isolated = true;
            for adj_file_idx in [file_idx as i32 - 1, file_idx as i32 + 1] {
                if adj_file_idx >= 0 && adj_file_idx < 8 {
                    let adj_file_bb = chess::get_file(File::from_index(adj_file_idx as usize));
                    if (black_pawns_bb & adj_file_bb) != BitBoard(0) {
                        isolated = false;
                        break;
                    }
                }
            }
            
            if is_passed {
                let passed_bonus = [0, 5, 10, 20, 35, 60, 100, 200][7 - rank_idx];
                score -= passed_bonus;
                
                if is_hidden_passed {
                    score -= passed_bonus / 2;
                }
            }
            
            if is_candidate {
                let candidate_bonus = [0, 2, 4, 8, 16, 32, 64, 128][7 - rank_idx];
                score -= candidate_bonus;
            }
            
            if doubled {
                score += 15;
            }
            
            if isolated {
                score += 10;
                if file_idx == 0 || file_idx == 7 {
                    score += 5;
                }
            }
            
            if is_backward {
                score += 25;
            }
            
            if is_connected {
                score -= 8;
                if rank_idx <= 3 {
                    score -= (4 - rank_idx) as i32 * 4;
                }
            }
            
            if is_sentry {
                score -= 12;
            }
            
            if is_faker {
                score += 15;
            }
            
            if is_weak {
                score += 8;
            }
            
            if blocking_pieces > 0 {
                score += blocking_pieces * 3;
            }
            
            if black_file_masks[file_idx].popcnt() == 1 && 
               (file_idx == 0 || black_file_masks[file_idx - 1] == BitBoard(0)) &&
               (file_idx == 7 || black_file_masks[file_idx + 1] == BitBoard(0)) {
                score += 5;
            }
        }
        
        let mut hanging_pairs = 0;
        for file_idx in 2..6 {
            if white_file_masks[file_idx] != BitBoard(0) && 
               white_file_masks[file_idx + 1] != BitBoard(0) &&
               white_file_masks[file_idx - 1] == BitBoard(0) &&
               white_file_masks[file_idx + 2] == BitBoard(0) {
                hanging_pairs += 1;
            }
        }
        score -= hanging_pairs * 20;
        
        hanging_pairs = 0;
        for file_idx in 2..6 {
            if black_file_masks[file_idx] != BitBoard(0) && 
               black_file_masks[file_idx + 1] != BitBoard(0) &&
               black_file_masks[file_idx - 1] == BitBoard(0) &&
               black_file_masks[file_idx + 2] == BitBoard(0) {
                hanging_pairs += 1;
            }
        }
        score += hanging_pairs * 20;
        
        let mut white_minority_attack = 0;
        let mut black_minority_attack = 0;
        
        if white_queenside < black_queenside && white_queenside > 0 {
            white_minority_attack = 1;
        }
        if black_queenside < white_queenside && black_queenside > 0 {
            black_minority_attack = 1;
        }
        
        score += white_minority_attack * 15;
        score -= black_minority_attack * 15;
        
        let mut white_chains = 0;
        let mut black_chains = 0;
        
        for sq in white_pawns_bb {
            let file_idx = sq.get_file() as usize;
            let rank_idx = sq.get_rank() as usize;
            
            if rank_idx > 0 {
                for adj_file_idx in [file_idx as i32 - 1, file_idx as i32 + 1] {
                    if adj_file_idx >= 0 && adj_file_idx < 8 {
                        let support_sq = Square::make_square(
                            Rank::from_index(rank_idx - 1), 
                            File::from_index(adj_file_idx as usize)
                        );
                        if (white_pawns_bb & BitBoard::from_square(support_sq)) != BitBoard(0) {
                            white_chains += 1;
                            break;
                        }
                    }
                }
            }
        }
        
        for sq in black_pawns_bb {
            let file_idx = sq.get_file() as usize;
            let rank_idx = sq.get_rank() as usize;
            
            if rank_idx < 7 {
                for adj_file_idx in [file_idx as i32 - 1, file_idx as i32 + 1] {
                    if adj_file_idx >= 0 && adj_file_idx < 8 {
                        let support_sq = Square::make_square(
                            Rank::from_index(rank_idx + 1), 
                            File::from_index(adj_file_idx as usize)
                        );
                        if (black_pawns_bb & BitBoard::from_square(support_sq)) != BitBoard(0) {
                            black_chains += 1;
                            break;
                        }
                    }
                }
            }
        }
        
        score += white_chains * 6;
        score -= black_chains * 6;
        
        let mut white_race_potential = 0;
        let mut black_race_potential = 0;
        
        for sq in white_pawns_bb {
            let rank_idx = sq.get_rank() as usize;
            if rank_idx >= 4 {
                white_race_potential += (8 - rank_idx) as i32;
            }
        }
        
        for sq in black_pawns_bb {
            let rank_idx = sq.get_rank() as usize;
            if rank_idx <= 3 {
                black_race_potential += rank_idx as i32;
            }
        }
        
        if white_race_potential > black_race_potential {
            score += (white_race_potential - black_race_potential) * 2;
        } else {
            score -= (black_race_potential - white_race_potential) * 2;
        }
        
        PAWN_HASH_AGE += 1;
        if PAWN_HASH_TABLE[index][0].2 < PAWN_HASH_TABLE[index][1].2 {
            PAWN_HASH_TABLE[index][0] = (pawn_key, score, PAWN_HASH_AGE);
        } else {
            PAWN_HASH_TABLE[index][1] = (pawn_key, score, PAWN_HASH_AGE);
        }
        
        score
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
fn evaluate_mobility(board: &Board, color: Color) -> i32 {
    let mut mobility = 0;
    let piece_types = [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen];
    
    for piece_type in piece_types {
        let pieces = *board.pieces(piece_type) & *board.color_combined(color);
        
        for square in pieces {
            let moves = MoveGen::new_legal(board)
                .filter(|m| m.get_source() == square)
                .count() as i32;
            
            let bonus = match piece_type {
                Piece::Knight => (moves - 4).max(0) * MOBILITY_BONUS[1],
                Piece::Bishop => (moves - 7).max(0) * MOBILITY_BONUS[2],
                Piece::Rook => (moves - 7).max(0) * MOBILITY_BONUS[3],
                Piece::Queen => (moves - 14).max(0) * MOBILITY_BONUS[4],
                _ => 0,
            };
            
            mobility += bonus;
        }
    }
    
    mobility
}
const EVAL_CACHE_MAX: usize = 20000;
fn evaluate(board: &Board) -> i32 {
    let key = compute_zobrist_hash(board);
    {
        let cache = EVAL_CACHE.lock().unwrap();
        if let Some(&cached) = cache.get(&key) {
            return cached;
        }
    }
    let is_checkmate = (*board.checkers() != BitBoard(0)) && { // Use != BitBoard(0)
        let mut moves = MoveGen::new_legal(board);
        moves.next().is_none()
    };
    if is_checkmate {
        let result = if board.side_to_move() == Color::White { -30000 } else { 30000 };
        let mut cache = EVAL_CACHE.lock().unwrap();
        cache.insert(key.clone(), result);
        if cache.len() > EVAL_CACHE_MAX {
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }
        return result;
    }
    let is_stalemate = *board.checkers() == BitBoard(0) && { // Use == BitBoard(0)
        let mut moves = MoveGen::new_legal(board);
        moves.next().is_none()
    };
    if is_stalemate {
        let result = if board.side_to_move() == Color::White { CONTEMPT } else { -CONTEMPT };
        let mut cache = EVAL_CACHE.lock().unwrap();
        cache.insert(key.clone(), result);
        if cache.len() > EVAL_CACHE_MAX {
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }
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
    let endgame = is_endgame(board);
    let phase = phase_value(board);
    for sq_idx in 0..64 {
        let square = unsafe { Square::new(sq_idx) }; // Square::new is unsafe
        if let Some(piece) = board.piece_on(square) {
            let piece_color = if (*board.color_combined(Color::White) & BitBoard::from_square(square)) != BitBoard(0) { // Use != BitBoard(0)
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
        }
    }   
    score += evaluate_mobility(board, Color::White);
    score -= evaluate_mobility(board, Color::Black);
    score += evaluate_pawn_structure(board);
    score += evaluate_rooks(board);
    if (board.pieces(Piece::Bishop) & board.color_combined(Color::White)).popcnt() >= 2 {
        score += 30;
    }
    if (board.pieces(Piece::Bishop) & board.color_combined(Color::Black)).popcnt() >= 2 {
        score -= 30;
    }
    
    let final_score = if board.side_to_move() == Color::White { score } else { -score };
    let mut cache = EVAL_CACHE.lock().unwrap();
    cache.insert(key, final_score);
    if cache.len() > EVAL_CACHE_MAX {
        if let Some(first_key) = cache.keys().next().cloned() {
            cache.remove(&first_key);
        }
    }
    final_score
}
fn order_moves(board: &Board, moves: Vec<ChessMove>, hash_move: Option<ChessMove>, depth: usize) -> Vec<ChessMove> {
    let mut scored_moves: Vec<(ChessMove, i32)> = moves.into_iter().map(|mv| {
        let score = if Some(mv) == hash_move {
            1000000
        } else if board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() {
            let see_value = see(board, mv);
            if see_value >= 0 {
                900000 + see_value
            } else {
                100000 + see_value
            }
        } else {
            let new_board = board.make_move_new(mv);
            if *new_board.checkers() != BitBoard(0) {
                800000
            } else {
                let history = HISTORY_HEURISTIC.lock().unwrap();
                if depth < 64 {
                    let killers = KILLER_MOVES.lock().unwrap();
                    if killers.len() > depth {
                        if killers[depth].len() > 0 && killers[depth][0] == mv {
                            700000
                        } else if killers[depth].len() > 1 && killers[depth][1] == mv {
                            600000
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
    }).collect();
    scored_moves.sort_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}

fn quiesce(board: &Board, mut alpha: i32, beta: i32, start_time: Instant, stats: &mut HashMap<&str, usize>, root_color: Color) -> i32 {
    *stats.entry("nodes").or_insert(0) += 1;
    *stats.entry("qnodes").or_insert(0) += 1;
    
    if start_time.elapsed().as_secs_f64() > TIME_LIMIT {
        return alpha;
    }
    
    let stand_pat = evaluate(board);
    
    if stand_pat >= beta {
        return beta;
    }
    
    let big_delta = 900;
    if stand_pat < alpha - big_delta {
        return alpha;
    }
    
    if stand_pat > alpha {
        alpha = stand_pat;
    }
    
    let mut captures = Vec::new();
    let movegen = MoveGen::new_legal(board);
    
    for mv in movegen {
        if board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() {
            let see_value = see(board, mv);
            if see_value >= SEE_CAPTURE_THRESHOLD || mv.get_promotion().is_some() {
                captures.push((mv, see_value));
            }
        }
    }
    
    captures.sort_by(|a, b| b.1.cmp(&a.1));
    
    for (mv, see_value) in captures {
        if stand_pat + see_value + DELTA_MARGIN < alpha {
            continue;
        }
        
        let new_board = board.make_move_new(mv);
        let score = -quiesce(&new_board, -beta, -alpha, start_time, stats, root_color);
        
        if score >= beta {
            return beta;
        }
        
        if score > alpha {
            alpha = score;
        }
    }
    
    alpha
}

fn negamax(board: &Board, depth: usize, mut alpha: i32, mut beta: i32, start_time: Instant, ply: usize, stats: &mut HashMap<&str, usize>, root_color: Color, tt: &mut TranspositionTable) -> i32 {
    let is_pv = beta - alpha > 1;
    let is_root = ply == 0;
    
    *stats.entry("nodes").or_insert(0) += 1;
    
    if start_time.elapsed().as_secs_f64() > TIME_LIMIT {
        return alpha;
    }
    
    if ply > 0 {
        if board.status() == BoardStatus::Checkmate {
            return -30000 + ply as i32;
        }
        
        if board.status() == BoardStatus::Stalemate {
            return if board.side_to_move() == root_color { -CONTEMPT } else { CONTEMPT };
        }
        if alpha >= beta {
            return alpha;
        }
    }
    
    let board_hash = compute_zobrist_hash(board);
    let mut hash_move = None;
    
    if let Some(tt_value) = tt.get(board_hash, depth, alpha, beta) {
        *stats.entry("tt_hits").or_insert(0) += 1;
        if !is_pv || (tt_value <= alpha || tt_value >= beta) {
            return tt_value;
        }
    }
    
    hash_move = tt.get_move(board_hash);
    
    let in_check = *board.checkers() != BitBoard(0);
    let static_eval = if in_check { -30000 + ply as i32 } else { evaluate(board) };
    
    if depth == 0 {
        return quiesce(board, alpha, beta, start_time, stats, root_color);
    }
    
    let improving = !in_check && ply >= 2;
    
    if !is_pv && !in_check {
        if depth <= RAZOR_DEPTH && static_eval + RAZOR_MARGIN < alpha {
            let razor_alpha = alpha - RAZOR_MARGIN;
            let razor_score = quiesce(board, razor_alpha, razor_alpha + 1, start_time, stats, root_color);
            if razor_score <= razor_alpha {
                return razor_score;
            }
        }
        
        if depth <= REVERSE_FUTILITY_DEPTH && static_eval - REVERSE_FUTILITY_MARGIN * depth as i32 >= beta {
            return static_eval;
        }
        
        if depth >= NULL_MOVE_DEPTH && static_eval >= beta {
            let has_non_pawn_pieces = (board.combined() & board.color_combined(board.side_to_move())).0 !=
                ((board.pieces(Piece::King) | board.pieces(Piece::Pawn)) & board.color_combined(board.side_to_move())).0;
                
            if has_non_pawn_pieces {
                let r = 3 + depth / 4 + ((static_eval - beta) / 200).min(3) as usize;
                
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
                    );
                    
                    if null_score >= beta {
                        if depth < 12 {
                            return null_score;
                        }
                        
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
                        );
                        
                        if verification_score >= beta {
                            return null_score;
                        }
                    }
                }
            }
        }
    }
    
    if is_pv && depth >= IID_DEPTH && hash_move.is_none() {
        negamax(board, depth - 2, alpha, beta, start_time, ply, stats, root_color, tt);
        hash_move = tt.get_move(board_hash);
    }
    
    let alpha_orig = alpha;
    let mut moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
    
    if moves.is_empty() {
        return if in_check { -30000 + ply as i32 } else { 0 };
    }
    
    moves = order_moves(board, moves, hash_move, ply);
    
    let mut best_value = -31000;
    let mut best_move = None;
    let mut moves_searched = 0;
    
    for (i, mv) in moves.iter().enumerate() {
        let is_quiet = board.piece_on(mv.get_dest()).is_none() && mv.get_promotion().is_none();
        
        if !is_pv && !in_check && is_quiet && depth <= FUTILITY_DEPTH && i > 0 {
            let futility_value = static_eval + FUTILITY_MARGINS[depth.min(4)] + 200;
            if futility_value <= alpha {
                continue;
            }
        }
        
        if !is_pv && depth <= LMP_DEPTH && is_quiet && i >= LMP_MOVE_COUNTS[depth.min(4)] {
            continue;
        }
        
        if depth >= 4 && !is_pv && is_quiet && i >= 3 {
            let see_value = see(board, *mv);
            if see_value < SEE_QUIET_THRESHOLD {
                continue;
            }
        }
        
        let new_board = board.make_move_new(*mv);
        let gives_check = *new_board.checkers() != BitBoard(0);
        
        let mut extension = 0;
        if gives_check {
            extension = 1;
        } else if let Some(piece) = board.piece_on(mv.get_source()) {
            if piece == Piece::Pawn && (mv.get_dest().get_rank() == Rank::Eighth || mv.get_dest().get_rank() == Rank::First) {
                extension = 1;
            }
        }
        
        let mut new_depth = depth - 1 + extension;
        let mut do_full_search = true;
        
        if depth >= 3 && i >= LMR_FULL_DEPTH_MOVES && is_quiet && !gives_check {
            let mut r = 1;
            if i >= 6 { r += 1; }
            if i >= 12 { r += 1; }
            if !improving { r += 1; }
            
            r = r.min(LMR_REDUCTION_LIMIT).min(new_depth);
            
            new_depth = new_depth.saturating_sub(r);
            do_full_search = false;
        }
        
        let score = if i == 0 {
            -negamax(&new_board, new_depth, -beta, -alpha, start_time, ply + 1, stats, root_color, tt)
        } else {
            let mut search_score = -negamax(&new_board, new_depth, -alpha - 1, -alpha, start_time, ply + 1, stats, root_color, tt);
            
            if !do_full_search && search_score > alpha {
                new_depth = depth - 1 + extension;
                search_score = -negamax(&new_board, new_depth, -alpha - 1, -alpha, start_time, ply + 1, stats, root_color, tt);
            }
            
            if is_pv && search_score > alpha && search_score < beta {
                search_score = -negamax(&new_board, new_depth, -beta, -alpha, start_time, ply + 1, stats, root_color, tt);
            }
            
            search_score
        };
        
        moves_searched += 1;
        
        if score > best_value {
            best_value = score;
            best_move = Some(*mv);
        }
        
        if score > alpha {
            alpha = score;
            
            if is_quiet {
                {
                    let mut killers = KILLER_MOVES.lock().unwrap();
                    if ply < 64 && killers.len() > ply {
                        if killers[ply].is_empty() || killers[ply][0] != *mv {
                            if killers[ply].len() >= 2 {
                                killers[ply][1] = killers[ply][0];
                            } else if killers[ply].len() == 1 {
                                let first_killer = killers[ply][0];  // Store the value first
                                killers[ply].push(first_killer);     // Then use it
                            }
                            if killers[ply].is_empty() {
                                killers[ply].push(*mv);
                            } else {
                                killers[ply][0] = *mv;
                            }
                        }
                    }
                }
                
                {
                    let mut history = HISTORY_HEURISTIC.lock().unwrap();
                    *history.entry(*mv).or_insert(0) += (depth * depth) as i32;
                }
            }
        }
        
        if alpha >= beta {
            break;
        }
    }
    
    let flag = if best_value <= alpha_orig {
        TTFlag::Upper
    } else if best_value >= beta {
        TTFlag::Lower
    } else {
        TTFlag::Exact
    };
    
    tt.store(board_hash, depth, best_value, flag, best_move);
    
    best_value
}
fn iterative_deepening(board: &Board, max_time: f64) -> Option<ChessMove> {
    let start_time = Instant::now();
    let mut best_move = None;
    let mut best_score = 0;
    let root_color = board.side_to_move();    
    let movegen = MoveGen::new_legal(board);
    for mv in movegen {
        let new_board = board.make_move_new(mv);
        let is_checkmate = (*new_board.checkers() != BitBoard(0)) && {
            let mut moves = MoveGen::new_legal(&new_board);
            moves.next().is_none()
        };
        if is_checkmate {
            return Some(mv);
        }
    }
    
    let mut tt_guard = TRANSPOSITION_TABLE.lock().unwrap();
    
    for depth in 1..=MAX_DEPTH {
        if start_time.elapsed().as_secs_f64() > max_time * 0.85 {
            break;
        }
        
        let mut stats: HashMap<&str, usize> = HashMap::new();
        let start_depth_time = Instant::now();
        
        let (mut alpha, mut beta) = if depth <= 4 {
            (-31000, 31000)
        } else {
            (best_score - ASPIRATION_WINDOW, best_score + ASPIRATION_WINDOW)
        };
        
        let mut aspirations_failed = 0;
        let mut depth_best_move = None;
        let mut depth_best_score = best_score;
        let mut aspiration_found_valid_result = false;  // Track if aspiration found something good
        
        loop {
            let board_hash = compute_zobrist_hash(board);
            
            let score = negamax(board, depth, alpha, beta, start_time, 0, &mut stats, root_color, &mut *tt_guard);
            
            if start_time.elapsed().as_secs_f64() > max_time * 0.95 {
                break;
            }
            
            if score <= alpha {
                println!("info string depth {} failed low (score={}, alpha={})", depth, score, alpha);
                let multiplier = (aspirations_failed + 1).min(4) as i32;
                alpha = alpha.saturating_sub(ASPIRATION_WINDOW * multiplier).max(-31000);
                aspirations_failed += 1;
            } else if score >= beta {
                println!("info string depth {} failed high (score={}, beta={})", depth, score, beta);
                let multiplier = (aspirations_failed + 1).min(4) as i32;
                beta = beta.saturating_add(ASPIRATION_WINDOW * multiplier).min(31000);
                aspirations_failed += 1;
            } else {
                // SUCCESSFUL ASPIRATION SEARCH
                println!("info string depth {} aspiration success (score={})", depth, score);
                
                // Only accept non-mate scores from aspiration search, or mate-for-us scores
                if score.abs() < 30000 || score > 30000 {
                    depth_best_score = score;
                    aspiration_found_valid_result = true;
                    
                    if let Some(mv) = tt_guard.get_move(board_hash) {
                        let movegen = MoveGen::new_legal(board);
                        if movegen.into_iter().any(|legal_move| legal_move == mv) {
                            depth_best_move = Some(mv);
                            println!("info string depth {} aspiration found move: {}", depth, mv);
                        }
                    }
                } else {
                    println!("info string depth {} aspiration found mate-against-us, ignoring", depth);
                }
                break;
            }
            
            if aspirations_failed > 4 {
                println!("info string depth {} doing full window search", depth);
                alpha = -31000;
                beta = 31000;
                
                let final_score = negamax(board, depth, alpha, beta, start_time, 0, &mut stats, root_color, &mut *tt_guard);
                
                println!("info string depth {} full window result: score={}, aspiration_valid={}", 
                         depth, final_score, aspiration_found_valid_result);
                
                // Decision logic for accepting full window result:
                let should_accept_full_window = if !aspiration_found_valid_result {
                    // No valid aspiration result, accept anything
                    println!("info string depth {} no valid aspiration result, accepting full window", depth);
                    true
                } else if final_score < -30000 {
                    // Full window found mate-against-us, definitely keep aspiration
                    println!("info string depth {} full window found mate-against-us, keeping aspiration", depth);
                    false
                }  else if final_score > depth_best_score + 100 {
                    // Full window is significantly better (normal case)
                    println!("info string depth {} full window significantly better ({} vs {})", 
                             depth, final_score, depth_best_score);
                    true
                } else {
                    // Keep aspiration result
                    println!("info string depth {} keeping aspiration result ({} vs {})", 
                             depth, depth_best_score, final_score);
                    false
                };
                
                if should_accept_full_window {
                    depth_best_score = final_score;
                    
                    if let Some(mv) = tt_guard.get_move(board_hash) {
                        let movegen = MoveGen::new_legal(board);
                        if movegen.into_iter().any(|legal_move| legal_move == mv) {
                            depth_best_move = Some(mv);
                            println!("info string depth {} full window found move: {}", depth, mv);
                        }
                    }
                }
                break;
            }
        }
        
        // Update overall best if we found a valid move for this depth
        if let Some(mv) = depth_best_move {
            // Additional check: don't accept mate-against-us scores unless we have no choice
            if depth_best_score < -30000 && best_move.is_some() {
                println!("info string depth {} refusing mate-against-us score, keeping previous best", depth);
                break;
            }
            
            best_move = Some(mv);
            best_score = depth_best_score;
            println!("info string depth {} completed, final move: {} (score={})", depth, mv, best_score);
        } else {
            println!("info string depth {} incomplete, no valid move found", depth);
            break;
        }
        
        // Break if we found a mate-for-us
        if best_score > 30000 {
            println!("info string found mate-for-us, stopping search");
            break;
        }
        
        // Break if we're getting mate-against-us scores (something's wrong)
        if best_score < -30000 {
            println!("info string getting mate-against-us scores, stopping search");
            break;
        }
        
        if STATS {
            let mut pv = Vec::new();
            let mut temp_board = *board;
            for _ in 0..depth {
                let h = compute_zobrist_hash(&temp_board);
                if let Some(mv) = tt_guard.get_move(h) {
                    let movegen = MoveGen::new_legal(&temp_board);
                    if movegen.into_iter().any(|legal_move| legal_move == mv) {
                        pv.push(mv);
                        temp_board = temp_board.make_move_new(mv);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            
            let depth_time_ms = (start_depth_time.elapsed().as_secs_f64() * 1000.0) as u64;
            let nodes = *stats.get("nodes").unwrap_or(&0);
            let nps = if depth_time_ms > 0 { nodes * 1000 / depth_time_ms as usize } else { 0 };
            let pv_string = pv.iter().map(|m| format!("{}", m)).collect::<Vec<_>>().join(" ");

            println!("info depth {} seldepth {} score cp {} nodes {} nps {} time {} pv {}", 
                     depth, depth, best_score, nodes, nps, depth_time_ms, pv_string);
        }
    }
    
    if let Some(chosen_move) = best_move {
        println!("info string final choice: {}", chosen_move);
    }
    
    best_move.or_else(|| {
        let moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if moves.is_empty() {
            None
        } else {
            println!("info string fallback to first legal move: {}", moves[0]);
            Some(moves[0])
        }
    })
}
fn compute_zobrist_hash(board: &Board) -> u64 {
    let mut h = 0;
    for sq_idx in 0..64 {
        let square = unsafe { Square::new(sq_idx) }; // Square::new is unsafe
        if let Some(piece) = board.piece_on(square) {
            let piece_color = if (*board.color_combined(Color::White) & BitBoard::from_square(square)) != BitBoard(0) { // Use != BitBoard(0)
                Color::White
            } else {
                Color::Black
            };
            h ^= ZOBRIST_PIECES[&(piece, piece_color)][sq_idx as usize];
        }
    }
    let mut castling_rights = 0;
    if board.castle_rights(Color::White).has_kingside() { castling_rights |= 1; }
    if board.castle_rights(Color::White).has_queenside() { castling_rights |= 2; }
    if board.castle_rights(Color::Black).has_kingside() { castling_rights |= 4; }
    if board.castle_rights(Color::Black).has_queenside() { castling_rights |= 8; }
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
}

impl UCIEngine {
    fn new() -> Self {
        UCIEngine {
            board: Board::default(),
            debug: false,
        }
    }

    fn handle_uci(&self) {
        println!("id name RustKnightv1.7");
        println!("id author Anish");
        println!("uciok");
    }

    fn handle_isready(&self) {
        println!("readyok");
    }

    fn handle_position(&mut self, tokens: &[&str]) {
        if tokens.len() < 2 {
            return;
        }

        match tokens[1] {
            "startpos" => {
                self.board = Board::default();
                if tokens.len() > 2 && tokens[2] == "moves" {
                    for move_str in &tokens[3..] {
                        if let Ok(chess_move) = ChessMove::from_str(move_str) {
                            self.board = self.board.make_move_new(chess_move);
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
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn calculate_time_budget(&self, wtime: Option<u64>, btime: Option<u64>, winc: Option<u64>, binc: Option<u64>, movetime: Option<u64>) -> f64 {
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
                "setoption" => {},
                "register" => {},
                "ucinewgame" => {
                    self.board = Board::default();
                    {
                        let mut tt = TRANSPOSITION_TABLE.lock().unwrap();
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
                }
                "position" => self.handle_position(&tokens),
                "go" => self.handle_go(&tokens),
                "stop" => {},
                "ponderhit" => {},
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

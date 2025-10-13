use chess::{BitBoard, Board, Color, Piece, Square};
use lazy_static::lazy_static;
use rand::Rng;
use std::collections::HashMap;

lazy_static! {
    pub static ref ZOBRIST_PIECES: HashMap<(Piece, Color), Vec<u64>> = {
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
    pub static ref ZOBRIST_CASTLING: Vec<u64> = {
        let mut rng = rand::thread_rng();
        (0..16).map(|_| rng.gen::<u64>()).collect()
    };
    pub static ref ZOBRIST_EP: Vec<u64> = {
        let mut rng = rand::thread_rng();
        (0..8).map(|_| rng.gen::<u64>()).collect()
    };
    pub static ref ZOBRIST_TURN: u64 = rand::thread_rng().gen::<u64>();
}

pub fn compute_zobrist_hash(board: &Board) -> u64 {
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
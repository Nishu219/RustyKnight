use chess::{BitBoard, Board, Color, Square};
use lazy_static::lazy_static;
use rand::Rng;

lazy_static! {
    pub static ref ZOBRIST_PIECES: [[[u64; 64]; 2]; 6] = {
        let mut rng = rand::thread_rng();
        let mut arr = [[[0; 64]; 2]; 6];
        for piece in 0..6 {
            for color in 0..2 {
                for square in 0..64 {
                    arr[piece][color][square] = rng.gen::<u64>();
                }
            }
        }
        arr
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
            h ^= ZOBRIST_PIECES[piece.to_index()][piece_color.to_index()][sq_idx as usize];
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

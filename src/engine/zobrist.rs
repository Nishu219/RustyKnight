use chess::{BitBoard, Board, Color};
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
    
    // Iterate over occupied squares 
    let occupied_squares = board.color_combined(Color::White) | board.color_combined(Color::Black);
    for square in occupied_squares {
        let piece = board.piece_on(square).unwrap();
        let piece_color = if (board.color_combined(Color::White) & BitBoard::from_square(square)) != BitBoard(0) {
            Color::White
        } else {
            Color::Black
        };
        h ^= ZOBRIST_PIECES[piece.to_index()][piece_color.to_index()][square.to_index()];
    }
    
    // Castling rights calculation
    let white_castle = board.castle_rights(Color::White);
    let black_castle = board.castle_rights(Color::Black);
    let castling_rights = (white_castle.has_kingside() as u8) 
        | ((white_castle.has_queenside() as u8) << 1)
        | ((black_castle.has_kingside() as u8) << 2)
        | ((black_castle.has_queenside() as u8) << 3);
    h ^= ZOBRIST_CASTLING[castling_rights as usize];
    
    if let Some(ep_sq) = board.en_passant() {
        h ^= ZOBRIST_EP[ep_sq.get_file() as usize];
    }
    
    if board.side_to_move() == Color::Black {
        h ^= *ZOBRIST_TURN;
    }
    
    h
}

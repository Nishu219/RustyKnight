use chess::{BitBoard, Board, Color, Piece};
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

/// Index into ZOBRIST_CASTLING for the castling rights currently held by `board`.
/// Shared by both the from-scratch and the incremental hash computation so the two
/// always agree on the encoding.
#[inline]
fn castling_rights_index(board: &Board) -> usize {
    let white_castle = board.castle_rights(Color::White);
    let black_castle = board.castle_rights(Color::Black);
    (white_castle.has_kingside() as usize)
        | ((white_castle.has_queenside() as usize) << 1)
        | ((black_castle.has_kingside() as usize) << 2)
        | ((black_castle.has_queenside() as usize) << 3)
}

/// Zobrist contribution of `board`'s en-passant file (0 if there isn't one).
#[inline]
fn en_passant_hash(board: &Board) -> u64 {
    match board.en_passant() {
        Some(ep_sq) => ZOBRIST_EP[ep_sq.get_file() as usize],
        None => 0,
    }
}

/// Computes the Zobrist hash of `board` from scratch by scanning every occupied square.
/// Only needed when there's no previous hash to update incrementally from (e.g. setting up
/// a brand-new search from a UCI `position` command). Hot search paths should use
/// `update_zobrist_hash` / `SearchState::make_move` instead, which are much cheaper.
pub fn compute_zobrist_hash(board: &Board) -> u64 {
    let mut h = 0;
    
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
    
    h ^= ZOBRIST_CASTLING[castling_rights_index(board)];
    h ^= en_passant_hash(board);
    
    if board.side_to_move() == Color::Black {
        h ^= *ZOBRIST_TURN;
    }
    
    h
}

/// Incrementally derives the Zobrist hash of the position after a move, given the hash of
/// the position before it. `old_board` is the position before the move, `new_board` is
/// `old_board.make_move_new(mv)`, and `old_hash` is the already-known hash of `old_board`.
///
/// Instead of rescanning every square, this only touches squares whose occupant actually
/// changed (found by XOR-diffing the per-color occupancy bitboards), plus the cheap
/// castling-rights / en-passant / side-to-move fields. Diffing occupancy this way naturally
/// covers quiet moves, captures, promotions, castling (rook shift included), and en-passant
/// captures without needing any move-type-specific branching.
pub fn update_zobrist_hash(old_board: &Board, new_board: &Board, old_hash: u64) -> u64 {
    let mut h = old_hash;

    for &color in &[Color::White, Color::Black] {
        let old_bb = *old_board.color_combined(color);
        let new_bb = *new_board.color_combined(color);
        let changed = old_bb ^ new_bb;
        let color_idx = color.to_index();

        for square in changed {
            let sq_bb = BitBoard::from_square(square);
            if (old_bb & sq_bb) != BitBoard(0) {
                // A piece of this color left this square (moved away, or was captured).
                let piece = old_board.piece_on(square).unwrap();
                h ^= ZOBRIST_PIECES[piece.to_index()][color_idx][square.to_index()];
            } else {
                // A piece of this color arrived on this square (moved here, or promoted).
                let piece = new_board.piece_on(square).unwrap();
                h ^= ZOBRIST_PIECES[piece.to_index()][color_idx][square.to_index()];
            }
        }
    }

    let old_castling = castling_rights_index(old_board);
    let new_castling = castling_rights_index(new_board);
    if old_castling != new_castling {
        h ^= ZOBRIST_CASTLING[old_castling];
        h ^= ZOBRIST_CASTLING[new_castling];
    }

    h ^= en_passant_hash(old_board);
    h ^= en_passant_hash(new_board);

    // Side to move flips on every move.
    h ^= *ZOBRIST_TURN;

    h
}

/// Incrementally derives the Zobrist hash after a null move: side to move flips and any
/// en-passant square is cleared, but no piece or castling-rights change.
pub fn update_zobrist_hash_null_move(old_board: &Board, old_hash: u64) -> u64 {
    old_hash ^ *ZOBRIST_TURN ^ en_passant_hash(old_board)
}

pub fn compute_pawn_zobrist_hash(board: &Board) -> u64 {
    let mut h = 0u64;
    let pawn_idx = Piece::Pawn.to_index();

    let white_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::White);
    for square in white_pawns {
        h ^= ZOBRIST_PIECES[pawn_idx][Color::White.to_index()][square.to_index()];
    }

    let black_pawns = board.pieces(Piece::Pawn) & board.color_combined(Color::Black);
    for square in black_pawns {
        h ^= ZOBRIST_PIECES[pawn_idx][Color::Black.to_index()][square.to_index()];
    }

    h
}


use chess::{BitBoard, Board, ChessMove, Color, Piece, Square};
use std::cell::RefCell;

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

thread_local! {
    pub static KILLER_MOVES: RefCell<Vec<Vec<ChessMove>>> = RefCell::new(Vec::new());
    pub static HISTORY_HEURISTIC: RefCell<[[i32; 64]; 64]> = RefCell::new([[0; 64]; 64]);
    pub static COUNTER_MOVES: RefCell<[[Option<ChessMove>; 64]; 64]> = RefCell::new([[None; 64]; 64]);
    pub static CAPTURE_HISTORY: RefCell<[[[i32; 64]; 6]; 6]> = RefCell::new([[[0; 64]; 6]; 6]);
}

/// Static Exchange Evaluation with Threshold
/// Returns TRUE if the capture sequence meets or exceeds the threshold
/// Returns FALSE if the capture sequence is below the threshold
///
/// This allows efficient pruning decisions:
/// - see_capture(board, mv, 0) -> Is capture winning or equal?
/// - see_capture(board, mv, -100) -> Is capture not too bad (loss < 1 pawn)?
/// - see_capture(board, mv, 200) -> Does capture win at least 2 pawns?
pub fn see_capture(board: &Board, mv: ChessMove, threshold: i32) -> bool {
    let to_square = mv.get_dest();
    let from_square = mv.get_source();

    let captured_piece = match board.piece_on(to_square) {
        Some(p) => p,
        None => {
            if mv.get_promotion().is_some() {
                return PIECE_VALUES[Piece::Queen.to_index()] - PIECE_VALUES[Piece::Pawn.to_index()]
                    >= threshold;
            }
            return false;
        }
    };

    let moving_piece = match board.piece_on(from_square) {
        Some(p) => p,
        None => return false,
    };

    let mut see_value = PIECE_VALUES[captured_piece.to_index()];
    let mut trophy_value = PIECE_VALUES[moving_piece.to_index()];

    if see_value - trophy_value >= threshold {
        return true;
    }

    if see_value < threshold {
        return false;
    }

    let to_move_mask = board.color_combined(board.side_to_move());
    let to_move = if (BitBoard::from_square(from_square) & to_move_mask) != BitBoard::new(0) {
        board.side_to_move()
    } else {
        !board.side_to_move()
    };
    let opponent = !to_move;

    let mut attacks_to_move = BitBoard::new(0);
    let mut attacks_opponent = BitBoard::new(0);

    attacks_to_move |= get_pawn_attackers(to_square, to_move, board);
    attacks_opponent |= get_pawn_attackers(to_square, opponent, board);

    if attacks_opponent != BitBoard::new(0)
        && see_value - trophy_value + PIECE_VALUES[Piece::Pawn.to_index()] < threshold
    {
        return false;
    }

    let knight_attacks = chess::get_knight_moves(to_square);
    attacks_to_move |= knight_attacks & board.pieces(Piece::Knight) & board.color_combined(to_move);
    attacks_opponent |=
        knight_attacks & board.pieces(Piece::Knight) & board.color_combined(opponent);

    let king_attacks = get_king_attacks(to_square);
    attacks_to_move |= king_attacks & board.pieces(Piece::King) & board.color_combined(to_move);
    attacks_opponent |= king_attacks & board.pieces(Piece::King) & board.color_combined(opponent);

    let bishop_rays = chess::get_bishop_rays(to_square);
    attacks_to_move |= bishop_rays
        & (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen))
        & board.color_combined(to_move);
    attacks_opponent |= bishop_rays
        & (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen))
        & board.color_combined(opponent);

    let rook_rays = chess::get_rook_rays(to_square);
    attacks_to_move |= rook_rays
        & (board.pieces(Piece::Rook) | board.pieces(Piece::Queen))
        & board.color_combined(to_move);
    attacks_opponent |= rook_rays
        & (board.pieces(Piece::Rook) | board.pieces(Piece::Queen))
        & board.color_combined(opponent);

    let mut all_pieces = *board.combined();
    let from_bb = BitBoard::from_square(from_square);
    attacks_to_move ^= from_bb;
    all_pieces ^= from_bb;

    loop {
        if attacks_opponent == BitBoard::new(0) {
            trophy_value = 0;
        } else if let Some((attacker_sq, attacker_piece)) =
            find_least_valuable_attacker_threshold(
                board,
                opponent,
                attacks_opponent,
                all_pieces,
                to_square,
            )
        {
            see_value -= trophy_value;
            trophy_value = PIECE_VALUES[attacker_piece.to_index()];

            let attacker_bb = BitBoard::from_square(attacker_sq);
            attacks_opponent ^= attacker_bb;
            all_pieces ^= attacker_bb;
        } else {
            trophy_value = 0;
        }

        if see_value >= threshold {
            return true;
        }

        if see_value + trophy_value < threshold {
            return false;
        }

        if attacks_to_move == BitBoard::new(0) {
            trophy_value = 0;
        } else if let Some((attacker_sq, attacker_piece)) = find_least_valuable_attacker_threshold(
            board,
            to_move,
            attacks_to_move,
            all_pieces,
            to_square,
        ) {
            see_value += trophy_value;
            trophy_value = PIECE_VALUES[attacker_piece.to_index()];

            let attacker_bb = BitBoard::from_square(attacker_sq);
            attacks_to_move ^= attacker_bb;
            all_pieces ^= attacker_bb;
        } else {
            trophy_value = 0;
        }

        if see_value - trophy_value >= threshold {
            return true;
        }

        if see_value < threshold {
            return false;
        }
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

fn find_least_valuable_attacker_threshold(
    board: &Board,
    color: Color,
    attackers: BitBoard,
    all_pieces: BitBoard,
    target_square: Square,
) -> Option<(Square, Piece)> {
    let pawns = attackers & board.pieces(Piece::Pawn) & board.color_combined(color);
    if pawns != BitBoard::new(0) {
        for sq in pawns {
            return Some((sq, Piece::Pawn));
        }
    }

    let knights = attackers & board.pieces(Piece::Knight) & board.color_combined(color);
    if knights != BitBoard::new(0) {
        for sq in knights {
            return Some((sq, Piece::Knight));
        }
    }

    let bishops = attackers & board.pieces(Piece::Bishop) & board.color_combined(color);
    if bishops != BitBoard::new(0) {
        for sq in bishops {
            if !is_path_blocked(sq, target_square, all_pieces) {
                return Some((sq, Piece::Bishop));
            }
        }
    }

    let rooks = attackers & board.pieces(Piece::Rook) & board.color_combined(color);
    if rooks != BitBoard::new(0) {
        for sq in rooks {
            if !is_path_blocked(sq, target_square, all_pieces) {
                return Some((sq, Piece::Rook));
            }
        }
    }

    let queens = attackers & board.pieces(Piece::Queen) & board.color_combined(color);
    if queens != BitBoard::new(0) {
        for sq in queens {
            if !is_path_blocked(sq, target_square, all_pieces) {
                return Some((sq, Piece::Queen));
            }
        }
    }

    let kings = attackers & board.pieces(Piece::King) & board.color_combined(color);
    if kings != BitBoard::new(0) {
        for sq in kings {
            return Some((sq, Piece::King));
        }
    }

    None
}

fn is_path_blocked(from: Square, to: Square, occupied: BitBoard) -> bool {
    let from_file = from.get_file().to_index() as i32;
    let from_rank = from.get_rank().to_index() as i32;
    let to_file = to.get_file().to_index() as i32;
    let to_rank = to.get_rank().to_index() as i32;

    let file_diff = to_file - from_file;
    let rank_diff = to_rank - from_rank;

    let file_step = if file_diff > 0 {
        1
    } else if file_diff < 0 {
        -1
    } else {
        0
    };
    let rank_step = if rank_diff > 0 {
        1
    } else if rank_diff < 0 {
        -1
    } else {
        0
    };

    let mut current_file = from_file + file_step;
    let mut current_rank = from_rank + rank_step;

    while current_file != to_file || current_rank != to_rank {
        let sq = unsafe { Square::new((current_rank * 8 + current_file) as u8) };
        if (occupied & BitBoard::from_square(sq)) != BitBoard::new(0) {
            return true;
        }
        current_file += file_step;
        current_rank += rank_step;
    }

    false
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
pub fn mvv_lva_score(board: &Board, mv: ChessMove) -> i32 {
    let victim_value = if let Some(captured_piece) = board.piece_on(mv.get_dest()) {
        PIECE_VALUES[captured_piece.to_index()]
    } else if mv.get_promotion().is_some() {
        PIECE_VALUES[Piece::Queen.to_index()]
    } else {
        0
    };

    let attacker_value = if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
        PIECE_VALUES[attacker_piece.to_index()]
    } else {
        0
    };

    if victim_value > 0 {
        victim_value * 10 - attacker_value
    } else {
        0
    }
}
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScoredMove {
    pub mv: ChessMove,
    pub score: i32,
}

impl ScoredMove {
    pub fn new(mv: ChessMove, score: i32) -> Self {
        Self { mv, score }
    }
}

pub fn order_moves(
    board: &Board,
    moves: &mut [ScoredMove],
    hash_move: Option<ChessMove>,
    depth: usize,
    previous_move: Option<ChessMove>, 
) {
    let counter_move = previous_move.and_then(|prev_mv| get_counter_move(prev_mv));
    
    for scored_move in moves.iter_mut() {
        let mv = scored_move.mv;
        let score = if Some(mv) == hash_move {
            10000000
        } else if board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() {
            let base_mvv_lva = mvv_lva_score(board, mv);
            let capture_hist = get_capture_history_score(mv, board);
            
            if see_capture(board, mv, -50) {
                9500000 + base_mvv_lva + capture_hist
            } else {
                500000 + capture_hist
            }
        } else {
            let new_board = board.make_move_new(mv);
            if *new_board.checkers() != BitBoard(0) {
                8000000
            } else {
                HISTORY_HEURISTIC.with(|history| {
                    let history = history.borrow();
                    if Some(mv) == counter_move {
                        6500000
                    } else if depth < 64 {
                        KILLER_MOVES.with(|killers| {
                            let killers = killers.borrow();
                            if killers.len() > depth {
                                if killers[depth].len() > 0 && killers[depth][0] == mv {
                                    7000000
                                } else if killers[depth].len() > 1 && killers[depth][1] == mv {
                                    6000000
                                } else {
                                    history[mv.get_source().to_index()][mv.get_dest().to_index()]
                                }
                            } else {
                                history[mv.get_source().to_index()][mv.get_dest().to_index()]
                            }
                        })
                    } else {
                        history[mv.get_source().to_index()][mv.get_dest().to_index()]
                    }
                })
            }
        };
        scored_move.score = score;
    }
    
    moves.sort_by(|a, b| b.score.cmp(&a.score));
}

pub fn update_counter_move(previous_move: Option<ChessMove>, refutation: ChessMove) {
    if let Some(prev_mv) = previous_move {
        let from_sq = prev_mv.get_source().to_index();
        let to_sq = prev_mv.get_dest().to_index();
        
        COUNTER_MOVES.with(|counter_moves| {
            counter_moves.borrow_mut()[from_sq][to_sq] = Some(refutation);
        });
    }
}

pub fn get_counter_move(mv: ChessMove) -> Option<ChessMove> {
    let from_sq = mv.get_source().to_index();
    let to_sq = mv.get_dest().to_index();
    
    COUNTER_MOVES.with(|counter_moves| {
        counter_moves.borrow()[from_sq][to_sq]
    })
}

pub fn clear_counter_moves() {
    COUNTER_MOVES.with(|counter_moves| {
        *counter_moves.borrow_mut() = [[None; 64]; 64];
    });
}

pub fn update_capture_history(mv: ChessMove, board: &Board, depth: usize, failed_quiets: bool) {
    if let Some(captured_piece) = board.piece_on(mv.get_dest()) {
        if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
            let attacker_idx = attacker_piece.to_index();
            let victim_idx = captured_piece.to_index();
            let to_sq = mv.get_dest().to_index();
            
            let bonus = if failed_quiets {
                (depth * depth + depth * 2) as i32
            } else {
                (depth * depth) as i32
            };
            
            CAPTURE_HISTORY.with(|ch| {
                let mut ch = ch.borrow_mut();
                ch[attacker_idx][victim_idx][to_sq] += bonus;
                
                if ch[attacker_idx][victim_idx][to_sq] > 16000 {
                    ch[attacker_idx][victim_idx][to_sq] = 16000;
                }
            });
        }
    }
}

pub fn penalize_capture_history(mv: ChessMove, board: &Board, depth: usize) {
    if let Some(captured_piece) = board.piece_on(mv.get_dest()) {
        if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
            let attacker_idx = attacker_piece.to_index();
            let victim_idx = captured_piece.to_index();
            let to_sq = mv.get_dest().to_index();
            
            let penalty = (depth * depth / 2) as i32;
            
            CAPTURE_HISTORY.with(|ch| {
                let mut ch = ch.borrow_mut();
                ch[attacker_idx][victim_idx][to_sq] -= penalty;
                
                if ch[attacker_idx][victim_idx][to_sq] < -4000 {
                    ch[attacker_idx][victim_idx][to_sq] = -4000;
                }
            });
        }
    }
}

pub fn get_capture_history_score(mv: ChessMove, board: &Board) -> i32 {
    if let Some(captured_piece) = board.piece_on(mv.get_dest()) {
        if let Some(attacker_piece) = board.piece_on(mv.get_source()) {
            let attacker_idx = attacker_piece.to_index();
            let victim_idx = captured_piece.to_index();
            let to_sq = mv.get_dest().to_index();
            
            return CAPTURE_HISTORY.with(|ch| {
                ch.borrow()[attacker_idx][victim_idx][to_sq]
            });
        }
    }
    0
}

pub fn clear_capture_history() {
    CAPTURE_HISTORY.with(|ch| {
        *ch.borrow_mut() = [[[0; 64]; 6]; 6];
    });
}

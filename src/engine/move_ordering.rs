
use chess::{BitBoard, Board, ChessMove, Color, Piece, Square};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    pub static ref VALUES: HashMap<Piece, i32> = {
        let mut m = HashMap::new();
        m.insert(Piece::Pawn, 100);
        m.insert(Piece::Knight, 320);
        m.insert(Piece::Bishop, 330);
        m.insert(Piece::Rook, 500);
        m.insert(Piece::Queen, 900);
        m.insert(Piece::King, 0);
        m
    };
    pub static ref KILLER_MOVES: Mutex<Vec<Vec<ChessMove>>> = Mutex::new(Vec::new());
    pub static ref HISTORY_HEURISTIC: Mutex<HashMap<ChessMove, i32>> = Mutex::new(HashMap::new());
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
        && see_value - trophy_value + VALUES[&Piece::Pawn] < threshold
    {
        return false;
    }

    // Knights
    let knight_attacks = chess::get_knight_moves(to_square);
    attacks_to_move |= knight_attacks & board.pieces(Piece::Knight) & board.color_combined(to_move);
    attacks_opponent |=
        knight_attacks & board.pieces(Piece::Knight) & board.color_combined(opponent);

    // Kings
    let king_attacks = get_king_attacks(to_square);
    attacks_to_move |= king_attacks & board.pieces(Piece::King) & board.color_combined(to_move);
    attacks_opponent |= king_attacks & board.pieces(Piece::King) & board.color_combined(opponent);

    // Bishops and Queens (diagonal)
    let bishop_rays = chess::get_bishop_rays(to_square);
    attacks_to_move |= bishop_rays
        & (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen))
        & board.color_combined(to_move);
    attacks_opponent |= bishop_rays
        & (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen))
        & board.color_combined(opponent);

    // Rooks and Queens (straight)
    let rook_rays = chess::get_rook_rays(to_square);
    attacks_to_move |= rook_rays
        & (board.pieces(Piece::Rook) | board.pieces(Piece::Queen))
        & board.color_combined(to_move);
    attacks_opponent |= rook_rays
        & (board.pieces(Piece::Rook) | board.pieces(Piece::Queen))
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
                to_square,
            )
        {
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
        } else if let Some((attacker_sq, attacker_piece)) = find_least_valuable_attacker_threshold(
            board,
            to_move,
            attacks_to_move,
            all_pieces,
            to_square,
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
pub fn mvv_lva_score(board: &Board, mv: ChessMove) -> i32 {
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
pub fn order_moves(
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
                if see_capture(board, mv, 300) {
                    9500000 + mvv_lva_score(board, mv)
                } else if see_capture(board, mv, 0) {
                    7500000 + mvv_lva_score(board, mv)
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

use chess::{BitBoard, Board, ChessMove, Color, Piece, Square};
use std::cell::RefCell;

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

pub type ContinuationHistoryTable = [[[[i16; 64]; 6]; 64]; 6];

thread_local! {
    pub static KILLER_MOVES: RefCell<Vec<Vec<ChessMove>>> = RefCell::new(Vec::new());
    pub static HISTORY_HEURISTIC: RefCell<[[i32; 64]; 64]> = RefCell::new([[0; 64]; 64]);
    pub static COUNTER_MOVES: RefCell<[[Option<ChessMove>; 64]; 6]> = RefCell::new([[None; 64]; 6]);
    pub static CAPTURE_HISTORY: RefCell<[[[i32; 64]; 6]; 6]> = RefCell::new([[[0; 64]; 6]; 6]);
    pub static CMH_TABLE: RefCell<Box<ContinuationHistoryTable>> = 
        RefCell::new(Box::new([[[[0; 64]; 6]; 64]; 6]));
    pub static FMH_TABLE: RefCell<Box<ContinuationHistoryTable>> = 
        RefCell::new(Box::new([[[[0; 64]; 6]; 64]; 6]));
}

/// Static Exchange Evaluation with Threshold
/// Returns TRUE if the capture sequence meets or exceeds the threshold
/// Returns FALSE if the capture sequence is below the threshold
///
/// This allows efficient pruning decisions:
/// - see_capture(board, mv, 0) -> Is capture winning or equal?
/// - see_capture(board, mv, -100) -> Is capture not too bad (loss < 1 pawn)?
/// - see_capture(board, mv, 200) -> Does capture win at least 2 pawns?
#[inline]
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
#[inline]
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

#[inline]
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

#[inline]
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

#[inline]
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

#[inline]
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
    #[inline(always)]
    pub fn new(mv: ChessMove, score: i32) -> Self {
        Self { mv, score }
    }
}

#[inline]
pub fn order_moves(
    board: &Board,
    moves: &mut [ScoredMove],
    hash_move: Option<ChessMove>,
    depth: usize,
    previous_move: Option<ChessMove>,
    previous_2_move: Option<ChessMove>,
) {
    let counter_move = previous_move.and_then(|prev_mv| get_counter_move(board, prev_mv));

    let (killer_1, killer_2) = if depth < 64 {
        KILLER_MOVES.with(|killers| {
            let killers = killers.borrow();
            if killers.len() > depth {
                let k1 = if !killers[depth].is_empty() { Some(killers[depth][0]) } else { None };
                let k2 = if killers[depth].len() > 1 { Some(killers[depth][1]) } else { None };
                (k1, k2)
            } else {
                (None, None)
            }
        })
    } else {
        (None, None)
    };

    let prev_info = previous_move.and_then(|prev_mv| {
        board.piece_on(prev_mv.get_dest()).map(|p| (p.to_index(), prev_mv.get_dest().to_index()))
    });

    let prev_2_info = previous_2_move.and_then(|prev_mv| {
        board.piece_on(prev_mv.get_dest()).map(|p| (p.to_index(), prev_mv.get_dest().to_index()))
    });

    HISTORY_HEURISTIC.with(|history| {
        let history = history.borrow();
        CMH_TABLE.with(|cmh| {
            let cmh = cmh.borrow();
            FMH_TABLE.with(|fmh| {
                let fmh = fmh.borrow();
                CAPTURE_HISTORY.with(|capture_hist| {
                    let capture_hist = capture_hist.borrow();

                    let cmh_sub = prev_info.map(|(p, t)| &cmh[p][t]);
                    let fmh_sub = prev_2_info.map(|(p, t)| &fmh[p][t]);

                    for scored_move in moves.iter_mut() {
                        let mv = scored_move.mv;
                        let dest_piece = board.piece_on(mv.get_dest());
                        let is_promotion = mv.get_promotion().is_some();

                        let score = if Some(mv) == hash_move {
                            10000000
                        } else if dest_piece.is_some() || is_promotion {
                            let base_mvv_lva = mvv_lva_score(board, mv);
                            let capture_hist_score = match (dest_piece, board.piece_on(mv.get_source())) {
                                (Some(vic), Some(att)) => {
                                    capture_hist[att.to_index()][vic.to_index()][mv.get_dest().to_index()]
                                }
                                _ => 0,
                            };
                            
                            if see_capture(board, mv, -50) {
                                9500000 + base_mvv_lva + capture_hist_score
                            } else {
                                500000 + capture_hist_score
                            }
                        } else {
                            let new_board = board.make_move_new(mv);
                            if *new_board.checkers() != BitBoard(0) {
                                8000000
                            } else if Some(mv) == counter_move {
                                6500000
                            } else if Some(mv) == killer_1 {
                                7000000
                            } else if Some(mv) == killer_2 {
                                6000000
                            } else {
                                let curr_from = mv.get_source().to_index();
                                let curr_to = mv.get_dest().to_index();
                                let hist_score = history[curr_from][curr_to];
                                
                                let curr_p = board.piece_on(mv.get_source()).map(|p| p.to_index());
                                let cmh_score = match (cmh_sub, curr_p) {
                                    (Some(sub), Some(cp)) => sub[cp][curr_to] as i32,
                                    _ => 0,
                                };
                                let fmh_score = match (fmh_sub, curr_p) {
                                    (Some(sub), Some(cp)) => sub[cp][curr_to] as i32,
                                    _ => 0,
                                };
                                2 * hist_score + 2 * cmh_score + fmh_score
                            }
                        };
                        scored_move.score = score;
                    }
                });
            });
        });
    });
    
    moves.sort_by(|a, b| b.score.cmp(&a.score));
}

#[inline]
pub fn update_counter_move(board: &Board, previous_move: Option<ChessMove>, refutation: ChessMove) {
    if let Some(prev_mv) = previous_move {
        if let Some(piece) = board.piece_on(prev_mv.get_dest()) {
            let piece_idx = piece.to_index();
            let to_sq = prev_mv.get_dest().to_index();
            
            COUNTER_MOVES.with(|counter_moves| {
                counter_moves.borrow_mut()[piece_idx][to_sq] = Some(refutation);
            });
        }
    }
}

#[inline]
pub fn get_counter_move(board: &Board, mv: ChessMove) -> Option<ChessMove> {
    let piece = board.piece_on(mv.get_dest())?;
    let piece_idx = piece.to_index();
    let to_sq = mv.get_dest().to_index();
    
    COUNTER_MOVES.with(|counter_moves| {
        counter_moves.borrow()[piece_idx][to_sq]
    })
}

#[inline]
pub fn clear_counter_moves() {
    COUNTER_MOVES.with(|counter_moves| {
        *counter_moves.borrow_mut() = [[None; 64]; 6];
    });
}

#[inline]
pub fn continuation_bonus(depth: usize) -> i32 {
    let d = depth as i32;
    (crate::engine::constants::CONTHIST_BONUS_S * d * d + crate::engine::constants::CONTHIST_BONUS_C * d)
        .min(crate::engine::constants::CONTHIST_MAX_BONUS)
}

#[inline]
pub fn update_single_conthist(
    table: &mut ContinuationHistoryTable,
    prev_piece: Piece,
    prev_to: Square,
    curr_piece: Piece,
    curr_to: Square,
    bonus: i32,
) {
    let entry = &mut table[prev_piece.to_index()][prev_to.to_index()][curr_piece.to_index()][curr_to.to_index()];
    let current = *entry as i32;
    let updated = current + bonus - (current * bonus.abs()) / crate::engine::constants::CONTHIST_GRAVITY;
    *entry = updated as i16;
}

#[allow(dead_code)]
#[inline(always)]
pub fn get_cmh_score_direct(
    cmh: &ContinuationHistoryTable,
    prev_info: Option<(usize, usize)>,
    curr_piece: Option<Piece>,
    curr_dest: Square,
) -> i32 {
    match (prev_info, curr_piece) {
        (Some((prev_p, prev_t)), Some(cp)) => {
            cmh[prev_p][prev_t][cp.to_index()][curr_dest.to_index()] as i32
        }
        _ => 0,
    }
}

#[allow(dead_code)]
#[inline(always)]
pub fn get_fmh_score_direct(
    fmh: &ContinuationHistoryTable,
    prev_2_info: Option<(usize, usize)>,
    curr_piece: Option<Piece>,
    curr_dest: Square,
) -> i32 {
    match (prev_2_info, curr_piece) {
        (Some((p2_p, p2_t)), Some(cp)) => {
            fmh[p2_p][p2_t][cp.to_index()][curr_dest.to_index()] as i32
        }
        _ => 0,
    }
}

#[allow(dead_code)]
#[inline]
pub fn get_cmh_score(
    board: &Board,
    previous_move: Option<ChessMove>,
    curr_move: ChessMove,
) -> i32 {
    if let Some(prev_mv) = previous_move {
        if let Some(prev_piece) = board.piece_on(prev_mv.get_dest()) {
            if let Some(curr_piece) = board.piece_on(curr_move.get_source()) {
                let prev_p = prev_piece.to_index();
                let prev_t = prev_mv.get_dest().to_index();
                let curr_p = curr_piece.to_index();
                let curr_t = curr_move.get_dest().to_index();
                return CMH_TABLE.with(|cmh| {
                    cmh.borrow()[prev_p][prev_t][curr_p][curr_t] as i32
                });
            }
        }
    }
    0
}

#[allow(dead_code)]
#[inline]
pub fn update_cmh(
    board: &Board,
    previous_move: Option<ChessMove>,
    curr_move: ChessMove,
    bonus: i32,
) {
    if let Some(prev_mv) = previous_move {
        if let Some(prev_piece) = board.piece_on(prev_mv.get_dest()) {
            if let Some(curr_piece) = board.piece_on(curr_move.get_source()) {
                CMH_TABLE.with(|cmh| {
                    update_single_conthist(
                        &mut cmh.borrow_mut(),
                        prev_piece,
                        prev_mv.get_dest(),
                        curr_piece,
                        curr_move.get_dest(),
                        bonus,
                    );
                });
            }
        }
    }
}

#[inline]
pub fn clear_cmh_table() {
    CMH_TABLE.with(|cmh| {
        *cmh.borrow_mut() = Box::new([[[[0; 64]; 6]; 64]; 6]);
    });
}

#[allow(dead_code)]
#[inline]
pub fn get_fmh_score(
    board: &Board,
    prev_2_move: Option<ChessMove>,
    curr_move: ChessMove,
) -> i32 {
    if let Some(prev_mv) = prev_2_move {
        if let Some(prev_piece) = board.piece_on(prev_mv.get_dest()) {
            if let Some(curr_piece) = board.piece_on(curr_move.get_source()) {
                let prev_p = prev_piece.to_index();
                let prev_t = prev_mv.get_dest().to_index();
                let curr_p = curr_piece.to_index();
                let curr_t = curr_move.get_dest().to_index();
                return FMH_TABLE.with(|fmh| {
                    fmh.borrow()[prev_p][prev_t][curr_p][curr_t] as i32
                });
            }
        }
    }
    0
}

#[allow(dead_code)]
#[inline]
pub fn update_fmh(
    board: &Board,
    prev_2_move: Option<ChessMove>,
    curr_move: ChessMove,
    bonus: i32,
) {
    if let Some(prev_mv) = prev_2_move {
        if let Some(prev_piece) = board.piece_on(prev_mv.get_dest()) {
            if let Some(curr_piece) = board.piece_on(curr_move.get_source()) {
                FMH_TABLE.with(|fmh| {
                    update_single_conthist(
                        &mut fmh.borrow_mut(),
                        prev_piece,
                        prev_mv.get_dest(),
                        curr_piece,
                        curr_move.get_dest(),
                        bonus,
                    );
                });
            }
        }
    }
}

#[inline]
pub fn clear_fmh_table() {
    FMH_TABLE.with(|fmh| {
        *fmh.borrow_mut() = Box::new([[[[0; 64]; 6]; 64]; 6]);
    });
}

#[inline]
pub fn update_conthist_batch(
    table: &mut ContinuationHistoryTable,
    board: &Board,
    prev_move: Option<ChessMove>,
    cutoff_mv: ChessMove,
    failed_moves: &[ScoredMove],
    bonus: i32,
) {
    if let Some(prev_mv) = prev_move {
        if let Some(prev_piece) = board.piece_on(prev_mv.get_dest()) {
            let prev_dest = prev_mv.get_dest();
            let malus = -bonus;

            // Reward cutoff move
            if let Some(curr_piece) = board.piece_on(cutoff_mv.get_source()) {
                update_single_conthist(
                    table,
                    prev_piece,
                    prev_dest,
                    curr_piece,
                    cutoff_mv.get_dest(),
                    bonus,
                );
            }

            // Penalize prior quiet moves that failed to produce a cutoff
            for scored_mv in failed_moves {
                let prev_mv_quiet = scored_mv.mv;
                if board.piece_on(prev_mv_quiet.get_dest()).is_none() && prev_mv_quiet.get_promotion().is_none() {
                    if let Some(curr_piece) = board.piece_on(prev_mv_quiet.get_source()) {
                        update_single_conthist(
                            table,
                            prev_piece,
                            prev_dest,
                            curr_piece,
                            prev_mv_quiet.get_dest(),
                            malus,
                        );
                    }
                }
            }
        }
    }
}

#[allow(dead_code)]
#[inline]
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

#[allow(dead_code)]
#[inline]
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

#[inline]
pub fn update_capture_history_batch(
    board: &Board,
    cutoff_mv: ChessMove,
    failed_moves: &[ScoredMove],
    depth: usize,
) {
    let bonus = if !failed_moves.is_empty() {
        (depth * depth + depth * 2) as i32
    } else {
        (depth * depth) as i32
    };
    let penalty = (depth * depth / 2) as i32;

    CAPTURE_HISTORY.with(|ch| {
        let mut ch = ch.borrow_mut();
        if let Some(captured_piece) = board.piece_on(cutoff_mv.get_dest()) {
            if let Some(attacker_piece) = board.piece_on(cutoff_mv.get_source()) {
                let att_idx = attacker_piece.to_index();
                let vic_idx = captured_piece.to_index();
                let to_sq = cutoff_mv.get_dest().to_index();
                let val = &mut ch[att_idx][vic_idx][to_sq];
                *val = (*val + bonus).min(16000);
            }
        }

        for prev_scored_mv in failed_moves {
            let prev_mv = prev_scored_mv.mv;
            if board.piece_on(prev_mv.get_dest()).is_some() || prev_mv.get_promotion().is_some() {
                if let Some(captured_piece) = board.piece_on(prev_mv.get_dest()) {
                    if let Some(attacker_piece) = board.piece_on(prev_mv.get_source()) {
                        let att_idx = attacker_piece.to_index();
                        let vic_idx = captured_piece.to_index();
                        let to_sq = prev_mv.get_dest().to_index();
                        let val = &mut ch[att_idx][vic_idx][to_sq];
                        *val = (*val - penalty).max(-4000);
                    }
                }
            }
        }
    });
}

#[allow(dead_code)]
#[inline]
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

#[inline]
pub fn clear_capture_history() {
    CAPTURE_HISTORY.with(|ch| {
        *ch.borrow_mut() = [[[0; 64]; 6]; 6];
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::MoveGen;

    #[test]
    fn test_continuation_bonus() {
        let bonus_1 = continuation_bonus(1);
        let bonus_4 = continuation_bonus(4);
        assert!(bonus_1 > 0);
        assert!(bonus_4 > bonus_1);
        assert!(bonus_4 <= crate::engine::constants::CONTHIST_MAX_BONUS);
    }

    #[test]
    fn test_update_single_conthist() {
        let mut table = Box::new([[[[0i16; 64]; 6]; 64]; 6]);
        let prev_sq = unsafe { Square::new(12) };
        let curr_sq = unsafe { Square::new(28) };
        update_single_conthist(&mut table, Piece::Knight, prev_sq, Piece::Bishop, curr_sq, 150);
        let val = table[Piece::Knight.to_index()][prev_sq.to_index()][Piece::Bishop.to_index()][curr_sq.to_index()];
        assert_eq!(val, 150);

        // Test malus
        update_single_conthist(&mut table, Piece::Knight, prev_sq, Piece::Bishop, curr_sq, -50);
        let val2 = table[Piece::Knight.to_index()][prev_sq.to_index()][Piece::Bishop.to_index()][curr_sq.to_index()];
        assert!(val2 < 150);
    }

    #[test]
    fn test_order_moves_with_conthist() {
        let board = Board::default();
        let mut moves: Vec<ScoredMove> = MoveGen::new_legal(&board).map(|m| ScoredMove::new(m, 0)).collect();
        let prev_mv = ChessMove::new(unsafe { Square::new(12) }, unsafe { Square::new(28) }, None);
        order_moves(&board, &mut moves, None, 1, Some(prev_mv), None);
        assert!(!moves.is_empty());
        for i in 1..moves.len() {
            assert!(moves[i - 1].score >= moves[i].score);
        }
    }
}


use chess::{BitBoard, Board, ChessMove, Color, MoveGen, Piece, Square};
use crate::engine::worker::Worker;

pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

pub type ContinuationHistoryTable = [[[[i16; 64]; 6]; 64]; 6];

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
            if board.piece_on(from_square) == Some(Piece::Pawn) && board.en_passant() == Some(to_square) {
                Piece::Pawn
            } else if mv.get_promotion().is_some() {
                return PIECE_VALUES[Piece::Queen.to_index()] - PIECE_VALUES[Piece::Pawn.to_index()]
                    >= threshold;
            } else {
                return false;
            }
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
    } else if board.piece_on(mv.get_source()) == Some(Piece::Pawn) && board.en_passant() == Some(mv.get_dest()) {
        PIECE_VALUES[Piece::Pawn.to_index()]
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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MovePickerStage {
    TTMove,
    GenerateCaptures,
    GoodCaptures,
    Killer1,
    Killer2,
    CounterMove,
    GenerateQuiets,
    Quiets,
    BadCaptures,
    Done,
}

pub struct MovePicker {
    pub stage: MovePickerStage,
    pub hash_move: Option<ChessMove>,
    pub killer_1: Option<ChessMove>,
    pub killer_2: Option<ChessMove>,
    pub counter_move: Option<ChessMove>,

    moves: [ScoredMove; 128],
    move_count: usize,
    move_idx: usize,

    bad_captures: [ChessMove; 64],
    bad_count: usize,
    bad_idx: usize,

    prev_info: Option<(usize, usize)>,
    prev_2_info: Option<(usize, usize)>,
}

impl MovePicker {
    pub fn new(
        board: &Board,
        hash_move: Option<ChessMove>,
        depth: usize,
        previous_move: Option<ChessMove>,
        previous_2_move: Option<ChessMove>,
        worker: &Worker,
    ) -> Self {
        let counter_move = previous_move.and_then(|prev_mv| {
            let p = board.piece_on(prev_mv.get_dest())?;
            worker.counter_moves[p.to_index()][prev_mv.get_dest().to_index()]
        });

        let (killer_1, killer_2) = if depth < 64 {
            (worker.killers[depth][0], worker.killers[depth][1])
        } else {
            (None, None)
        };

        let prev_info = previous_move.and_then(|prev_mv| {
            board.piece_on(prev_mv.get_dest()).map(|p| (p.to_index(), prev_mv.get_dest().to_index()))
        });

        let prev_2_info = previous_2_move.and_then(|prev_mv| {
            board.piece_on(prev_mv.get_dest()).map(|p| (p.to_index(), prev_mv.get_dest().to_index()))
        });

        let dummy_mv = ChessMove::new(unsafe { Square::new(0) }, unsafe { Square::new(0) }, None);
        Self {
            stage: MovePickerStage::TTMove,
            hash_move,
            killer_1,
            killer_2,
            counter_move,
            moves: [ScoredMove::new(dummy_mv, 0); 128],
            move_count: 0,
            move_idx: 0,
            bad_captures: [dummy_mv; 64],
            bad_count: 0,
            bad_idx: 0,
            prev_info,
            prev_2_info,
        }
    }

    #[inline(always)]
    fn pick_best(&mut self) -> ChessMove {
        let mut best_score = self.moves[self.move_idx].score;
        let mut best_idx = self.move_idx;

        for i in (self.move_idx + 1)..self.move_count {
            if self.moves[i].score > best_score {
                best_score = self.moves[i].score;
                best_idx = i;
            }
        }

        self.moves.swap(self.move_idx, best_idx);
        let mv = self.moves[self.move_idx].mv;
        self.move_idx += 1;
        mv
    }

    pub fn next(&mut self, board: &Board, worker: &Worker) -> Option<ChessMove> {
        loop {
            match self.stage {
                MovePickerStage::TTMove => {
                    self.stage = MovePickerStage::GenerateCaptures;
                    if let Some(mv) = self.hash_move {
                        if board.legal(mv) {
                            return Some(mv);
                        }
                    }
                }

                MovePickerStage::GenerateCaptures => {
                    self.move_idx = 0;
                    self.move_count = 0;

                    let capture_hist = &worker.capture_history;
                    for mv in MoveGen::new_legal(board) {
                        let dest_piece = board.piece_on(mv.get_dest());
                        let is_promotion = mv.get_promotion().is_some();
                        let is_en_passant = board.piece_on(mv.get_source()) == Some(Piece::Pawn) && board.en_passant() == Some(mv.get_dest());
                        if dest_piece.is_some() || is_promotion || is_en_passant {
                            if Some(mv) == self.hash_move {
                                continue;
                            }
                            let base_mvv_lva = mvv_lva_score(board, mv);
                            let capture_hist_score = match (dest_piece, board.piece_on(mv.get_source())) {
                                (Some(vic), Some(att)) => {
                                    capture_hist[att.to_index()][vic.to_index()][mv.get_dest().to_index()]
                                }
                                _ => 0,
                            };
                            let score = base_mvv_lva + capture_hist_score;
                            if self.move_count < 128 {
                                self.moves[self.move_count] = ScoredMove::new(mv, score);
                                self.move_count += 1;
                            }
                        }
                    }

                    self.stage = MovePickerStage::GoodCaptures;
                }

                MovePickerStage::GoodCaptures => {
                    while self.move_idx < self.move_count {
                        let mv = self.pick_best();
                        if see_capture(board, mv, 0) {
                            return Some(mv);
                        } else {
                            if self.bad_count < 64 {
                                self.bad_captures[self.bad_count] = mv;
                                self.bad_count += 1;
                            }
                        }
                    }
                    self.stage = MovePickerStage::Killer1;
                }

                MovePickerStage::Killer1 => {
                    self.stage = MovePickerStage::Killer2;
                    if let Some(k1) = self.killer_1 {
                        let is_ep = board.piece_on(k1.get_source()) == Some(Piece::Pawn) && board.en_passant() == Some(k1.get_dest());
                        if Some(k1) != self.hash_move 
                            && board.piece_on(k1.get_dest()).is_none() 
                            && k1.get_promotion().is_none() 
                            && !is_ep
                            && board.legal(k1) 
                        {
                            return Some(k1);
                        }
                    }
                }

                MovePickerStage::Killer2 => {
                    self.stage = MovePickerStage::CounterMove;
                    if let Some(k2) = self.killer_2 {
                        let is_ep = board.piece_on(k2.get_source()) == Some(Piece::Pawn) && board.en_passant() == Some(k2.get_dest());
                        if Some(k2) != self.hash_move 
                            && Some(k2) != self.killer_1 
                            && board.piece_on(k2.get_dest()).is_none() 
                            && k2.get_promotion().is_none() 
                            && !is_ep
                            && board.legal(k2) 
                        {
                            return Some(k2);
                        }
                    }
                }

                MovePickerStage::CounterMove => {
                    self.stage = MovePickerStage::GenerateQuiets;
                    if let Some(cm) = self.counter_move {
                        let is_ep = board.piece_on(cm.get_source()) == Some(Piece::Pawn) && board.en_passant() == Some(cm.get_dest());
                        if Some(cm) != self.hash_move 
                            && Some(cm) != self.killer_1 
                            && Some(cm) != self.killer_2 
                            && board.piece_on(cm.get_dest()).is_none() 
                            && cm.get_promotion().is_none() 
                            && !is_ep
                            && board.legal(cm) 
                        {
                            return Some(cm);
                        }
                    }
                }

                MovePickerStage::GenerateQuiets => {
                    self.move_idx = 0;
                    self.move_count = 0;

                    let prev_info = self.prev_info;
                    let prev_2_info = self.prev_2_info;
                    let history = &worker.history;
                    let cmh = &worker.cmh_table;
                    let fmh = &worker.fmh_table;

                    let cmh_sub = prev_info.map(|(p, t)| &cmh[p][t]);
                    let fmh_sub = prev_2_info.map(|(p, t)| &fmh[p][t]);

                    for mv in MoveGen::new_legal(board) {
                        let is_ep = board.piece_on(mv.get_source()) == Some(Piece::Pawn) && board.en_passant() == Some(mv.get_dest());
                        if board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() || is_ep {
                            continue;
                        }
                        if Some(mv) == self.hash_move 
                            || Some(mv) == self.killer_1 
                            || Some(mv) == self.killer_2 
                            || Some(mv) == self.counter_move 
                        {
                            continue;
                        }

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

                        let gives_check = *board.make_move_new(mv).checkers() != BitBoard(0);
                        let check_bonus = if gives_check { 1_000_000 } else { 0 };

                        let score = check_bonus + 2 * hist_score + 2 * cmh_score + fmh_score;
                        if self.move_count < 128 {
                            self.moves[self.move_count] = ScoredMove::new(mv, score);
                            self.move_count += 1;
                        }
                    }

                    self.stage = MovePickerStage::Quiets;
                }

                MovePickerStage::Quiets => {
                    if self.move_idx < self.move_count {
                        let mv = self.pick_best();
                        return Some(mv);
                    }
                    self.stage = MovePickerStage::BadCaptures;
                }

                MovePickerStage::BadCaptures => {
                    if self.bad_idx < self.bad_count {
                        let mv = self.bad_captures[self.bad_idx];
                        self.bad_idx += 1;
                        return Some(mv);
                    }
                    self.stage = MovePickerStage::Done;
                }

                MovePickerStage::Done => return None,
            }
        }
    }
}

#[inline]
pub fn continuation_bonus(depth: usize) -> i32 {
    let d = depth as i32;
    (crate::engine::constants::CONTHIST_BONUS_S * d * d + crate::engine::constants::CONTHIST_BONUS_C * d)
        .min(crate::engine::constants::CONTHIST_MAX_BONUS)
}

use crate::engine::constants::*;
use crate::engine::constants::{CMH_PRUNING_DEPTH, CMH_PRUNING_THRESHOLD, LMR_HIST_DIVISOR};
use crate::engine::evaluation::{
    compute_psqt_and_phase, evaluate, piece_to_index, Score, KING_INDEX,
    PAWN_INDEX, PHASE_VALUES, PSQT, ROOK_INDEX,
};
use crate::engine::move_ordering::{
    continuation_bonus, mvv_lva_score, see_capture, MovePicker, ScoredMove, PIECE_VALUES,
};
use crate::engine::transposition_table::{TranspositionTable, TTFlag};
use crate::engine::worker::{SearchStats, Worker};
use crate::engine::zobrist::{compute_zobrist_hash, compute_pawn_zobrist_hash, update_zobrist_hash, update_zobrist_hash_null_move, ZOBRIST_PIECES};
use chess::{BitBoard, Board, ChessMove, Color, MoveGen, Piece, Square};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[inline]
fn other_color(c: Color) -> Color {
    if c == Color::White { Color::Black } else { Color::White }
}

#[derive(Clone, Copy)]
pub struct SearchState {
    pub board: Board,
    pub psqt: Score,
    pub phase: i32,
    pub pawn_hash: u64,
    pub position_hash: u64,
}

impl SearchState {
    pub fn new(board: Board) -> Self {
        let (psqt, phase) = compute_psqt_and_phase(&board);
        let pawn_hash = compute_pawn_zobrist_hash(&board);
        let position_hash = compute_zobrist_hash(&board);
        SearchState { board, psqt, phase, pawn_hash, position_hash }
    }

    pub fn make_move(&self, mv: ChessMove) -> Self {
        let board = &self.board;
        let moving_color = board.side_to_move();
        let moving_color_idx = moving_color.to_index();
        let moving_piece = board.piece_on(mv.get_source()).unwrap();
        let moving_piece_idx = piece_to_index(moving_piece);
        let from_sq = mv.get_source();
        let to_sq = mv.get_dest();
        let from_idx = from_sq.to_index();
        let to_idx = to_sq.to_index();

        let mut psqt = self.psqt;
        let mut phase = self.phase;
        let mut pawn_hash = self.pawn_hash;

        let is_en_passant = moving_piece == Piece::Pawn
            && board.piece_on(to_sq).is_none()
            && from_sq.get_file() != to_sq.get_file();

        if is_en_passant {
            let captured_sq = Square::make_square(from_sq.get_rank(), to_sq.get_file());
            let captured_color = other_color(moving_color);
            let captured_color_idx = captured_color.to_index();
            let cap_idx = captured_sq.to_index();

            if captured_color == Color::White {
                psqt -= PSQT[PAWN_INDEX][0][cap_idx];
            } else {
                psqt += PSQT[PAWN_INDEX][1][cap_idx];
            }
            pawn_hash ^= ZOBRIST_PIECES[PAWN_INDEX][captured_color_idx][cap_idx];
        } else if let Some(captured) = board.piece_on(to_sq) {
            let captured_color = other_color(moving_color);
            let captured_color_idx = captured_color.to_index();
            let cap_piece_idx = piece_to_index(captured);

            if captured_color == Color::White {
                psqt -= PSQT[cap_piece_idx][0][to_idx];
            } else {
                psqt += PSQT[cap_piece_idx][1][to_idx];
            }
            phase -= PHASE_VALUES[cap_piece_idx];
            if captured == Piece::Pawn {
                pawn_hash ^= ZOBRIST_PIECES[PAWN_INDEX][captured_color_idx][to_idx];
            }
        }

        if moving_piece == Piece::Pawn {
            pawn_hash ^= ZOBRIST_PIECES[PAWN_INDEX][moving_color_idx][from_idx];
            if let Some(promotion) = mv.get_promotion() {
                let prom_idx = piece_to_index(promotion);
                if moving_color == Color::White {
                    psqt -= PSQT[PAWN_INDEX][0][from_idx];
                    psqt += PSQT[prom_idx][0][to_idx];
                } else {
                    psqt += PSQT[PAWN_INDEX][1][from_idx];
                    psqt -= PSQT[prom_idx][1][to_idx];
                }
                phase += PHASE_VALUES[prom_idx];
            } else {
                if moving_color == Color::White {
                    psqt -= PSQT[PAWN_INDEX][0][from_idx];
                    psqt += PSQT[PAWN_INDEX][0][to_idx];
                } else {
                    psqt += PSQT[PAWN_INDEX][1][from_idx];
                    psqt -= PSQT[PAWN_INDEX][1][to_idx];
                }
                pawn_hash ^= ZOBRIST_PIECES[PAWN_INDEX][moving_color_idx][to_idx];
            }
        } else if moving_piece == Piece::King {
            let file_diff = (from_sq.get_file().to_index() as i32 - to_sq.get_file().to_index() as i32).abs();
            if file_diff == 2 {
                if moving_color == Color::White {
                    psqt -= PSQT[KING_INDEX][0][from_idx];
                    psqt += PSQT[KING_INDEX][0][to_idx];
                    if to_sq == Square::G1 {
                        psqt -= PSQT[ROOK_INDEX][0][Square::H1.to_index()];
                        psqt += PSQT[ROOK_INDEX][0][Square::F1.to_index()];
                    } else if to_sq == Square::C1 {
                        psqt -= PSQT[ROOK_INDEX][0][Square::A1.to_index()];
                        psqt += PSQT[ROOK_INDEX][0][Square::D1.to_index()];
                    }
                } else {
                    psqt += PSQT[KING_INDEX][1][from_idx];
                    psqt -= PSQT[KING_INDEX][1][to_idx];
                    if to_sq == Square::G8 {
                        psqt += PSQT[ROOK_INDEX][1][Square::H8.to_index()];
                        psqt -= PSQT[ROOK_INDEX][1][Square::F8.to_index()];
                    } else if to_sq == Square::C8 {
                        psqt += PSQT[ROOK_INDEX][1][Square::A8.to_index()];
                        psqt -= PSQT[ROOK_INDEX][1][Square::D8.to_index()];
                    }
                }
            } else {
                if moving_color == Color::White {
                    psqt -= PSQT[KING_INDEX][0][from_idx];
                    psqt += PSQT[KING_INDEX][0][to_idx];
                } else {
                    psqt += PSQT[KING_INDEX][1][from_idx];
                    psqt -= PSQT[KING_INDEX][1][to_idx];
                }
            }
        } else {
            if moving_color == Color::White {
                psqt -= PSQT[moving_piece_idx][0][from_idx];
                psqt += PSQT[moving_piece_idx][0][to_idx];
            } else {
                psqt += PSQT[moving_piece_idx][1][from_idx];
                psqt -= PSQT[moving_piece_idx][1][to_idx];
            }
        }

        phase = phase.clamp(0, 24);

        let new_board = board.make_move_new(mv);
        let position_hash = update_zobrist_hash(board, &new_board, self.position_hash);

        SearchState {
            board: new_board,
            psqt,
            phase,
            pawn_hash,
            position_hash,
        }
    }

    pub fn null_move(&self) -> Option<Self> {
        let position_hash = update_zobrist_hash_null_move(&self.board, self.position_hash);
        self.board.null_move().map(|b| SearchState {
            board: b,
            psqt: self.psqt,
            phase: self.phase,
            pawn_hash: self.pawn_hash,
            position_hash,
        })
    }
}

fn quiesce(
    state: &SearchState,
    mut alpha: i32,
    beta: i32,
    start_time: Instant,
    worker: &mut Worker,
    root_color: Color,
    max_time: f64,
    timeout_occurred: &mut bool,
    ply: usize,
    stop_flag: &AtomicBool,
) -> i32 {
    let board = &state.board;
    worker.stats.nodes += 1;
    worker.stats.qnodes += 1;
    worker.stats.seldepth = worker.stats.seldepth.max(ply);
    
    if worker.stats.nodes % TIME_CHECK_INTERVAL == 0 && (stop_flag.load(Ordering::Relaxed) || start_time.elapsed().as_secs_f64() > max_time) {
        *timeout_occurred = true;
        return evaluate(board, state.psqt, state.phase, state.pawn_hash, 0, -30000, 30000, &mut worker.pawn_table);
    }

    let stand_pat = evaluate(board, state.psqt, state.phase, state.pawn_hash, 0, alpha, beta, &mut worker.pawn_table);

    if stand_pat >= beta {
        return beta;
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Delta pruning
    let big_delta = PIECE_VALUES[Piece::Queen.to_index()] + DELTA_MARGIN;

    if stand_pat + big_delta < alpha {
        return alpha;
    }

    let mut captures = [ScoredMove::new(ChessMove::new(unsafe { Square::new(0) }, unsafe { Square::new(0) }, None), 0); 128];
    let mut count = 0;
    for mv in MoveGen::new_legal(board) {
        let is_en_passant = board.piece_on(mv.get_source()) == Some(Piece::Pawn) && board.en_passant() == Some(mv.get_dest());
        if board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() || is_en_passant {
            let mvv_lva = mvv_lva_score(board, mv);
            if count < 128 {
                captures[count] = ScoredMove::new(mv, mvv_lva);
                count += 1;
            }
        }
    }

    for current_idx in 0..count {
        let mut best_score = captures[current_idx].score;
        let mut best_idx = current_idx;
        for j in (current_idx + 1)..count {
            if captures[j].score > best_score {
                best_score = captures[j].score;
                best_idx = j;
            }
        }
        captures.swap(current_idx, best_idx);
        let mv = captures[current_idx].mv;

        if !see_capture(board, mv, -50) {
            continue;
        }

        if *timeout_occurred {
            break;
        }
        
        let captured_value = if let Some(captured_piece) = board.piece_on(mv.get_dest()) {
            PIECE_VALUES[captured_piece.to_index()]
        } else if mv.get_promotion().is_some() {
            PIECE_VALUES[Piece::Queen.to_index()] - PIECE_VALUES[Piece::Pawn.to_index()]
        } else if board.piece_on(mv.get_source()) == Some(Piece::Pawn) && board.en_passant() == Some(mv.get_dest()) {
            PIECE_VALUES[Piece::Pawn.to_index()]
        } else {
            0
        };
        
        if stand_pat + captured_value + DELTA_MARGIN < alpha {
            continue;
        }
        
        let new_state = state.make_move(mv);
        let score = -quiesce(
            &new_state,
            -beta,
            -alpha,
            start_time,
            worker,
            root_color,
            max_time,
            timeout_occurred,
            ply + 1,
            stop_flag,
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
    state: &SearchState,
    depth: usize,
    mut alpha: i32,
    mut beta: i32,
    start_time: Instant,
    ply: usize,
    worker: &mut Worker,
    root_color: Color,
    tt: &TranspositionTable,
    max_time: f64,
    timeout_occurred: &mut bool,
    previous_move: Option<ChessMove>,
    previous_2_move: Option<ChessMove>,
    contempt: i32,
    halfmove_clock: u16,
    stop_flag: &AtomicBool,
) -> i32 {
    let board = &state.board;
    let is_pv = beta - alpha > 1;
    let is_root = ply == 0;
    let original_alpha = alpha;

    worker.stats.nodes += 1;
    worker.stats.seldepth = worker.stats.seldepth.max(ply);

    if worker.stats.nodes % TIME_CHECK_INTERVAL == 0 && (stop_flag.load(Ordering::Relaxed) || start_time.elapsed().as_secs_f64() > max_time) {
        *timeout_occurred = true;
        return if ply > 0 { evaluate(board, state.psqt, state.phase, state.pawn_hash, 0, -30000, 30000, &mut worker.pawn_table) } else { 0 };
    }

    let position_hash = state.position_hash;

    // Draw and mate distance pruning
    if ply > 0 {
        if halfmove_clock >= 100 || worker.is_repetition(position_hash) {
            return 0; 
        }

        alpha = alpha.max(-30000 + ply as i32);
        beta = beta.min(30000 - ply as i32);
        if alpha >= beta {
            return alpha;
        }
    }

    let board_hash = position_hash;
    let (tt_score, tt_move) = tt.get_with_move(board_hash, depth, alpha, beta);
    let mut hash_move = tt_move;
    let mut tt_value = None;
    if let Some(stored_value) = tt_score {
        worker.stats.tt_hits += 1;
        if !is_pv || !is_root {
            return stored_value;
        }
        tt_value = Some(stored_value);
    }
    
    let in_check = *board.checkers() != BitBoard(0);
    let mut depth = depth;
    if in_check {
        depth += 1;
    }

    let static_eval = if in_check {
        -30000 + ply as i32
    } else {
        evaluate(board, state.psqt, state.phase, state.pawn_hash, contempt, alpha, beta, &mut worker.pawn_table)
    };

    if depth == 0 {
        return quiesce(
            state,
            alpha,
            beta,
            start_time,
            worker,
            root_color,
            max_time,
            timeout_occurred,
            ply,
            stop_flag,
        );
    }

    if !is_pv && !in_check {
        // Razoring
        if depth <= RAZOR_DEPTH {
            let razor_margin = RAZOR_MARGIN + 50 * (depth as i32 - 1);
            if static_eval + razor_margin < alpha {
                let razor_alpha = alpha - razor_margin;
                let razor_score = quiesce(
                    state,
                    razor_alpha,
                    razor_alpha + 1,
                    start_time,
                    worker,
                    root_color,
                    max_time,
                    timeout_occurred,
                    ply,
                    stop_flag,
                );
                if *timeout_occurred {
                    return evaluate(board, state.psqt, state.phase, state.pawn_hash, 0, -30000, 30000, &mut worker.pawn_table);
                }
                if razor_score <= razor_alpha {
                    return razor_score;
                }
            }
        }

        // RFP
        if depth <= REVERSE_FUTILITY_DEPTH {
            let rfp_margin = REVERSE_FUTILITY_MARGIN * depth as i32;
            
            if static_eval - rfp_margin >= beta {
                return static_eval;
            }
        }

        // Null Move Pruning
        if depth >= NULL_MOVE_DEPTH && static_eval >= beta {
            let has_non_pawn_pieces = (board.combined() & board.color_combined(board.side_to_move())).0
                    != ((board.pieces(chess::Piece::King) | board.pieces(chess::Piece::Pawn))
                        & board.color_combined(board.side_to_move()))
                    .0;

            if has_non_pawn_pieces {
                let mut r = 3 + depth / 4;
                r += ((static_eval - beta) / 200).min(3) as usize;
                r += 1;
                if depth > 12 {
                    r += 1;
                }
                r = r.min(depth - 1);

                if let Some(null_state) = state.null_move() {
                    let null_score = -negamax(
                        &null_state,
                        depth.saturating_sub(r),
                        -beta,
                        -beta + 1,
                        start_time,
                        ply + 1,
                        worker,
                        root_color,
                        tt,
                        max_time,
                        timeout_occurred,
                        None,
                        None,
                        contempt,
                        halfmove_clock + 1,
                        stop_flag,
                    );

                    if *timeout_occurred {
                        return evaluate(board, state.psqt, state.phase, state.pawn_hash, 0, -30000, 30000, &mut worker.pawn_table);
                    }

                    if null_score >= beta {
                        return null_score;
                    }
                }
            }
        }

        // ProbCut
        if depth >= 5 && !in_check && beta.abs() < 29000 {
            let probcut_beta = beta + 200;
            let probcut_see_threshold = probcut_beta - static_eval;

            if let Some(tt_val) = tt.get(board_hash, depth - 3, probcut_beta - 1, probcut_beta) {
                if tt_val >= probcut_beta {
                    return tt_val;
                }
            }

            let mut probcut_moves = [ScoredMove::new(ChessMove::new(unsafe { Square::new(0) }, unsafe { Square::new(0) }, None), 0); 256];
            let mut probcut_count = 0;
            for mv in MoveGen::new_legal(board) {
                if (board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some())
                    && see_capture(board, mv, probcut_see_threshold)
                {
                    probcut_moves[probcut_count] = ScoredMove::new(mv, mvv_lva_score(board, mv));
                    probcut_count += 1;
                }
            }

            let probcut_moves_slice = &mut probcut_moves[..probcut_count];
            probcut_moves_slice.sort_by(|a, b| b.score.cmp(&a.score));

            'probcut: for scored_mv in probcut_moves_slice {
                let mv = scored_mv.mv;
                if *timeout_occurred {
                    break 'probcut;
                }

                let new_state = state.make_move(mv);
                let new_position_hash = new_state.position_hash;
                worker.push_repetition(new_position_hash);

                let score = -negamax(
                    &new_state,
                    depth - 4,
                    -probcut_beta,
                    -probcut_beta + 1,
                    start_time,
                    ply + 1,
                    worker,
                    root_color,
                    tt,
                    max_time,
                    timeout_occurred,
                    Some(mv),
                    previous_move,
                    contempt,
                    0,
                    stop_flag,
                );

                worker.pop_repetition();

                if *timeout_occurred {
                    break 'probcut;
                }

                if score >= probcut_beta {
                    tt.store(board_hash, depth - 3, score, TTFlag::Lower, Some(mv));
                    return score;
                }
            }

            if *timeout_occurred {
                return evaluate(board, state.psqt, state.phase, state.pawn_hash, 0, -30000, 30000, &mut worker.pawn_table);
            }
        }
    }

    // IID
    if is_pv && depth >= IID_DEPTH && hash_move.is_none() {
        let iid_depth = depth - 2 - if is_pv { 0 } else { 1 };
        negamax(
            state,
            iid_depth,
            alpha,
            beta,
            start_time,
            ply,
            worker,
            root_color,
            tt,
            max_time,
            timeout_occurred,
            previous_move,
            previous_2_move,
            contempt,
            halfmove_clock,
            stop_flag,
        );
        if !*timeout_occurred {
            hash_move = tt.get_move(board_hash);
        }
    }

    // Singular Extensions
    let mut singular_extension = 0;
    if !is_root 
        && depth >= SINGULAR_DEPTH 
        && hash_move.is_some() 
        && !in_check
        && tt_value.is_some()
    {
        let tt_entry_depth = tt.get_depth(board_hash);
        
        if let Some(entry_depth) = tt_entry_depth {
            if entry_depth >= depth - 3 {
                let hash_mv = hash_move.unwrap();
                let tt_val = tt_value.unwrap();
                
                let singular_beta = tt_val - depth as i32 * SINGULAR_MARGIN_MULTIPLIER;
                let singular_depth = (depth - 1) / 2;
                
                let mut best_singular_score = -32000;
                let mut is_singular = true;
                
                let mut picker = MovePicker::new(board, Some(hash_mv), singular_depth, None, None, worker);
                
                while let Some(mv) = picker.next(board, worker) {
                    if mv == hash_mv {
                        continue;
                    }
                    
                    if *timeout_occurred {
                        is_singular = false;
                        break;
                    }
                    
                    let new_state = state.make_move(mv);
                    let new_position_hash = new_state.position_hash;
                    worker.push_repetition(new_position_hash);
                    
                    let score = -negamax(
                        &new_state,
                        singular_depth,
                        -singular_beta - 1,
                        -singular_beta,
                        start_time,
                        ply + 1,
                        worker,
                        root_color,
                        tt,
                        max_time,
                        timeout_occurred,
                        None,
                        None,
                        contempt,
                        {
                            let is_capture = board.piece_on(mv.get_dest()).is_some();
                            let is_pawn = board.piece_on(mv.get_source()) == Some(chess::Piece::Pawn);
                            if is_capture || is_pawn { 0 } else { halfmove_clock + 1 }
                        },
                        stop_flag,
                    );
                    
                    worker.pop_repetition();
                    
                    if *timeout_occurred {
                        is_singular = false;
                        break;
                    }
                    
                    best_singular_score = best_singular_score.max(score);
                    
                    if score >= singular_beta {
                        is_singular = false;
                        break;
                    }
                }
                
                if !*timeout_occurred && is_singular {
                    singular_extension = 1;
                    
                    if depth >= DOUBLE_EXTENSION_DEPTH 
                        && best_singular_score < singular_beta - DOUBLE_EXTENSION_MARGIN
                    {
                        singular_extension = 2;
                    }
                }
            }
        }
    }

    // Mate Threat Extension
    let mut mate_threat_extension = 0;
    if !is_pv 
        && !in_check 
        && depth >= MATE_THREAT_DEPTH 
        && static_eval < beta - 300
    {
        if let Some(null_state) = state.null_move() {
            let threat_score = -negamax(
                &null_state,
                depth.saturating_sub(3),
                -beta,
                -beta + 1,
                start_time,
                ply + 1,
                worker,
                root_color,
                tt,
                max_time,
                timeout_occurred,
                None,
                None,
                contempt,
                halfmove_clock + 1,
                stop_flag,
            );
            
            if !*timeout_occurred && threat_score >= beta && threat_score.abs() > 29000 {
                mate_threat_extension = 1;
            }
        }
    }

    let mut picker = MovePicker::new(board, hash_move, ply, previous_move, previous_2_move, worker);
    let killer_moves = [picker.killer_1, picker.killer_2];
    let counter_move = picker.counter_move;

    let mut best_value = -31000;
    let mut best_move = None;
    let mut moves_searched = 0;
    let mut quiet_moves_searched = 0;
    let mut searched_moves = [ScoredMove::new(ChessMove::new(unsafe { Square::new(0) }, unsafe { Square::new(0) }, None), 0); 128];

    let prev_info = previous_move.and_then(|prev_mv| {
        board.piece_on(prev_mv.get_dest()).map(|p| (p.to_index(), prev_mv.get_dest().to_index()))
    });

    let prev_2_info = previous_2_move.and_then(|prev_mv| {
        board.piece_on(prev_mv.get_dest()).map(|p| (p.to_index(), prev_mv.get_dest().to_index()))
    });

    let active_weight = 2 + if previous_move.is_some() { 2 } else { 0 } + if previous_2_move.is_some() { 1 } else { 0 };

    while let Some(mv) = picker.next(board, worker) {
        let i = moves_searched;
        if i < 128 {
            searched_moves[i] = ScoredMove::new(mv, 0);
        }
        if *timeout_occurred {
            break;
        }

        let moving_piece = board.piece_on(mv.get_source());
        let is_en_passant = moving_piece == Some(chess::Piece::Pawn) && board.en_passant() == Some(mv.get_dest());
        let is_capture = board.piece_on(mv.get_dest()).is_some() || is_en_passant;
        let is_promotion = mv.get_promotion().is_some();
        let is_quiet = !is_capture && !is_promotion;
        let is_hash_move = hash_move.is_some() && mv == hash_move.unwrap();
        let is_killer = Some(mv) == killer_moves[0] || Some(mv) == killer_moves[1];
        let is_counter = Some(mv) == counter_move;

        let next_halfmove_clock = if is_capture || moving_piece == Some(chess::Piece::Pawn) {
            0
        } else {
            halfmove_clock + 1
        };

        let (hist_score, cmh_score, _fmh_score, h_avg) = if is_quiet {
            let curr_from = mv.get_source().to_index();
            let curr_to = mv.get_dest().to_index();
            let hist_score = worker.history[curr_from][curr_to];
            let curr_p = moving_piece.map(|p| p.to_index());
            let cmh_score = match (prev_info, curr_p) {
                (Some((prev_p, prev_t)), Some(cp)) => {
                    worker.cmh_table[prev_p][prev_t][cp][curr_to] as i32
                }
                _ => 0,
            };
            let fmh_score = match (prev_2_info, curr_p) {
                (Some((p2_p, p2_t)), Some(cp)) => {
                    worker.fmh_table[p2_p][p2_t][cp][curr_to] as i32
                }
                _ => 0,
            };

            let h_avg = (2 * hist_score + 2 * cmh_score + fmh_score) / active_weight;
            (hist_score, cmh_score, fmh_score, h_avg)
        } else {
            (0, 0, 0, 0)
        };

        let new_state = state.make_move(mv);
        let gives_check = *new_state.board.checkers() != BitBoard(0);

        if !is_pv && !in_check && is_quiet && !gives_check {
            // Late move pruning
            if depth <= LMP_DEPTH {
                let mut lmp_threshold = match depth {
                    1 => 3,
                    2 => 6,
                    3 => 12,
                    4 => 18,
                    _ => 25,
                };

                if static_eval + 200 < alpha {
                    lmp_threshold /= 2;
                } else if static_eval > alpha + 200 {
                    lmp_threshold = (lmp_threshold * 3) / 2;
                }

                if quiet_moves_searched >= lmp_threshold && !is_killer && !is_counter {
                    continue;
                }
            }

            // Futility pruning
            if depth <= FUTILITY_DEPTH && i > 0 {
                let history_margin_adj = (h_avg / 50).clamp(-100, 50);
                let futility_margin = FUTILITY_MARGINS[depth.min(4)] + history_margin_adj;

                let futility_value = static_eval + futility_margin;
                if futility_value <= alpha && !is_killer && !is_counter {
                    continue;
                }
            }

            // Futility move count pruning
            if depth <= 4 && moves_searched > 0 {
                let futility_value = static_eval + FUTILITY_MARGINS[depth.min(4)];
                if futility_value <= alpha && !is_killer && !is_counter {
                    let move_count_limit = FUTILITY_MOVE_COUNTS[depth.min(4)];
                    if quiet_moves_searched >= move_count_limit {
                        continue;
                    }
                }
            }

            // CMH Pruning (Continuation History Leaf Pruning)
            if depth <= CMH_PRUNING_DEPTH 
                && moves_searched > 0 
                && !is_hash_move 
                && !is_killer 
                && !is_counter 
            {
                if cmh_score < CMH_PRUNING_THRESHOLD {
                    continue;
                }
            }

            // History leaf pruning
            if depth <= HISTORY_PRUNING_DEPTH 
                && moves_searched > 0 
                && !is_hash_move
            {
                let history_score = hist_score;
                
                if history_score < HISTORY_PRUNING_THRESHOLD && !is_counter && cmh_score <= 0 {
                    continue;
                }
            }
        }

        let new_position_hash = new_state.position_hash;
        worker.push_repetition(new_position_hash);

        // Extensions
        let mut extension = mate_threat_extension;
        
        if is_hash_move && singular_extension > 0 {
            extension = extension.max(singular_extension);
        } else if gives_check && see_capture(board, mv, 0) {
            extension += 1;
        } else if let Some(piece) = moving_piece {
            if piece == chess::Piece::Pawn {
                let dest_rank = mv.get_dest().get_rank();
                if dest_rank == chess::Rank::Seventh || dest_rank == chess::Rank::Second {
                    extension += 1;
                }
            }
        }
        
        extension = extension.min(2);
        
        let mut new_depth = depth - 1 + extension;
        let mut do_full_search = true;

        // LMR
        if depth >= 3 
            && i >= LMR_FULL_DEPTH_MOVES 
            && is_quiet 
            && !gives_check 
            && !is_killer
            && !is_counter
        {
            let depth_idx = depth.min(63);
            let move_idx = i.min(255);
            
            let mut r_scaled = unsafe { crate::engine::constants::LMR_TABLE[depth_idx][move_idx] };

            if !is_pv {
                r_scaled += 716;
            } else {
                r_scaled -= 204;
            }

            let history_bonus = (h_avg * LMR_SCALE) / LMR_HIST_DIVISOR;
            let clamped_bonus = history_bonus.clamp(-1280, 1280);
            
            r_scaled -= clamped_bonus;

            if (static_eval - alpha).abs() > 150 {
                r_scaled -= 256;
            }

            if i > 15 {
                r_scaled += 512;
            }

            let mut r = (r_scaled / LMR_SCALE).max(0) as usize;

            r = r.min(depth.saturating_sub(1));
            new_depth = new_depth.saturating_sub(r);
            do_full_search = false;
        }

        // PVS
        let score = if i == 0 {
            -negamax(
                &new_state,
                new_depth,
                -beta,
                -alpha,
                start_time,
                ply + 1,
                worker,
                root_color,
                tt,
                max_time,
                timeout_occurred,
                Some(mv),
                previous_move,
                contempt,
                next_halfmove_clock,
                stop_flag,
            )
        } else {
            let mut search_score = -negamax(
                &new_state,
                new_depth,
                -alpha - 1,
                -alpha,
                start_time,
                ply + 1,
                worker,
                root_color,
                tt,
                max_time,
                timeout_occurred,
                Some(mv),
                previous_move,
                contempt,
                next_halfmove_clock,
                stop_flag,
            );

            if *timeout_occurred {
                evaluate(board, state.psqt, state.phase, state.pawn_hash, 0, -30000, 30000, &mut worker.pawn_table)
            } else {
                // Re-search at full depth if reduced
                if !do_full_search && search_score > alpha {
                    new_depth = depth - 1 + extension;
                    search_score = -negamax(
                        &new_state,
                        new_depth,
                        -alpha - 1,
                        -alpha,
                        start_time,
                        ply + 1,
                        worker,
                        root_color,
                        tt,
                        max_time,
                        timeout_occurred,
                        Some(mv),
                        previous_move,
                        contempt,
                        next_halfmove_clock,
                        stop_flag,
                    );
                }

                // PV re-search
                if *timeout_occurred {
                    evaluate(board, state.psqt, state.phase, state.pawn_hash, 0, -30000, 30000, &mut worker.pawn_table)
                } else if search_score > alpha {
                    if is_pv || (search_score < beta) {
                        -negamax(
                            &new_state,
                            new_depth,
                            -beta,
                            -alpha,
                            start_time,
                            ply + 1,
                            worker,
                            root_color,
                            tt,
                            max_time,
                            timeout_occurred,
                            Some(mv),
                            previous_move,
                            contempt,
                            next_halfmove_clock,
                            stop_flag,
                        )
                    } else {
                        search_score
                    }
                } else {
                    search_score
                }
            }
        };

        worker.pop_repetition();

        if *timeout_occurred {
            break;
        }

        moves_searched += 1;
        if is_quiet {
            quiet_moves_searched += 1;
        }

        if score > best_value {
            best_value = score;
            best_move = Some(mv);
        }

        if score > alpha {
            alpha = score;

            if is_quiet {
                worker.update_killer(ply, mv);

                let bonus = (depth * depth).min(400) as i32; 
                worker.update_history(mv, bonus);

                let penalty = -bonus;
                let count = i.min(128);
                for prev_scored_mv in &searched_moves[0..count] {
                    let prev_mv = prev_scored_mv.mv;
                    if board.piece_on(prev_mv.get_dest()).is_none()
                        && prev_mv.get_promotion().is_none()
                    {
                        worker.penalize_history(prev_mv, penalty);
                    }
                }
            } else {
                let count = i.min(128);
                let bonus = if count > 0 { (depth * depth + depth * 2) as i32 } else { (depth * depth) as i32 };
                worker.update_capture_history_batch(board, mv, &searched_moves[0..count], bonus);
            }
        }

        if alpha >= beta {
            if is_quiet {
                worker.update_counter_move(board, previous_move, mv);

                let conthist_bonus = continuation_bonus(depth);
                let count = i.min(128);
                let failed_quiets = &searched_moves[0..count];

                if previous_move.is_some() {
                    worker.update_conthist_batch(
                        true,
                        board,
                        previous_move,
                        mv,
                        failed_quiets,
                        conthist_bonus,
                    );
                }

                if previous_2_move.is_some() {
                    worker.update_conthist_batch(
                        false,
                        board,
                        previous_2_move,
                        mv,
                        failed_quiets,
                        conthist_bonus,
                    );
                }
            }
            break;
        }
    }

    if moves_searched == 0 {
        return if in_check { -30000 + ply as i32 } else { 0 };
    }

    if !*timeout_occurred {
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

pub fn iterative_deepening(
    board: &Board,
    max_time: f64,
    contempt: i32,
    is_movetime: bool,
    max_depth: Option<usize>,
    halfmove_clock: u16,
    worker: &mut Worker,
    tt: &TranspositionTable,
    stop_flag: &AtomicBool,
) -> Option<ChessMove> {
    let start_time = Instant::now();
    let mut best_move = None;
    let mut best_score = 0;
    let root_color = board.side_to_move();
    let root_state = SearchState::new(*board);

    worker.stats = SearchStats::default();

    let max_search_depth = max_depth.unwrap_or(MAX_DEPTH);

    for depth in 1..=max_search_depth {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        if max_depth.is_none() {
            let elapsed = start_time.elapsed().as_secs_f64();
            let time_limit = if is_movetime {
                max_time * 0.95
            } else {
                max_time * 0.80
            };
            if elapsed > time_limit {
                break;
            }
        }

        let mut timeout_occurred = false;

        // Aspiration window setup
        let (initial_alpha, initial_beta) = if depth <= 4 || best_move.is_none() {
            (-31000, 31000)
        } else {
            (best_score - INITIAL_WINDOW, best_score + INITIAL_WINDOW)
        };

        let mut alpha = initial_alpha;
        let mut beta = initial_beta;
        let mut window_size = INITIAL_WINDOW;
        let mut search_iterations = 0;

        loop {
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }
            search_iterations += 1;

            let score = negamax(
                &root_state,
                depth,
                alpha,
                beta,
                start_time,
                0,
                worker,
                root_color,
                tt,
                max_time,
                &mut timeout_occurred,
                None,
                None,
                contempt,
                halfmove_clock,
                stop_flag,
            );

            if timeout_occurred || stop_flag.load(Ordering::Relaxed) {
                break;
            }

            if score <= alpha {
                // Fail low: widen window
                window_size = (window_size * 2).min(MAX_WINDOW);

                if window_size >= MAX_WINDOW || search_iterations >= MAX_ASPIRATION_ITERATIONS {
                    alpha = -31000;
                    beta = 31000;
                } else {
                    alpha = (best_score - window_size).max(-31000);
                    beta = (best_score + window_size).min(31000);
                }
            } else if score >= beta {
                // Fail high: widen window
                window_size = (window_size * 2).min(MAX_WINDOW);

                if window_size >= MAX_WINDOW || search_iterations >= MAX_ASPIRATION_ITERATIONS {
                    alpha = -31000;
                    beta = 31000;
                } else {
                    alpha = (best_score - window_size).max(-31000);
                    beta = (best_score + window_size).min(31000);
                }
            } else {
                best_score = score;

                let board_hash = root_state.position_hash;
                if let Some(mv) = tt.get_move(board_hash) {
                    if board.legal(mv) {
                        best_move = Some(mv);

                        if STATS {
                            // Extract PV
                            let mut pv = Vec::new();
                            let mut temp_board = *board;
                            let mut temp_hash = board_hash;
                            let mut pv_hashes = Vec::with_capacity(depth.min(20));
                            let mut temp_clock = halfmove_clock;

                            for _ in 0..depth.min(20) {
                                if temp_clock >= 100 {
                                    break; 
                                }

                                let h = temp_hash;
                                let history_count = worker.repetition_table.iter().filter(|&&x| x == h).count();
                                let mut pv_count = 0;
                                for &x in pv_hashes.iter() {
                                    if x == h { pv_count += 1; }
                                }
                                
                                if history_count + pv_count >= 2 {
                                    break; 
                                }
                                pv_hashes.push(h);

                                if let Some(pv_move) = tt.get_move(h) {
                                    if temp_board.legal(pv_move) {
                                        pv.push(pv_move);
                                        
                                        let is_capture = temp_board.piece_on(pv_move.get_dest()).is_some();
                                        let is_pawn = temp_board.piece_on(pv_move.get_source()) == Some(chess::Piece::Pawn);
                                        if is_capture || is_pawn {
                                            temp_clock = 0;
                                        } else {
                                            temp_clock += 1;
                                        }

                                        let next_board = temp_board.make_move_new(pv_move);
                                        temp_hash = update_zobrist_hash(&temp_board, &next_board, temp_hash);
                                        temp_board = next_board;
                                    } else {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }

                            let time_ms = (start_time.elapsed().as_secs_f64() * 1000.0) as u64;
                            let nps = if time_ms > 0 {
                                worker.stats.nodes * 1000 / time_ms as usize
                            } else {
                                0
                            };
                            use std::fmt::Write;
                            let mut pv_string = String::new();
                            for (i, m) in pv.iter().enumerate() {
                                if i > 0 {
                                    pv_string.push(' ');
                                }
                                write!(&mut pv_string, "{}", m).unwrap();
                            }

                            let hashfull = tt.hashfull();

                            println!(
                                "info depth {} seldepth {} score cp {} nodes {} nps {} time {} hashfull {} pv {}",
                                depth,
                                worker.stats.seldepth,
                                best_score,
                                worker.stats.nodes,
                                nps,
                                time_ms,
                                hashfull,
                                pv_string
                            );
                        }

                        if best_score.abs() > 29000 {
                            break;
                        }
                    }
                }
                break;
            }

            if timeout_occurred || search_iterations >= MAX_ASPIRATION_ITERATIONS * 2 {
                break;
            }
        }

        if timeout_occurred || stop_flag.load(Ordering::Relaxed) {
            break;
        }
    }

    best_move.or_else(|| {
        MoveGen::new_legal(board).next()
    })
}


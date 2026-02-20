use crate::engine::constants::*;
use crate::engine::evaluation::evaluate;
use crate::engine::move_ordering::{order_moves, see_capture, mvv_lva_score, KILLER_MOVES, HISTORY_HEURISTIC, PIECE_VALUES, update_counter_move, update_capture_history, penalize_capture_history};
use crate::engine::transposition_table::{TranspositionTable, TTFlag};
use crate::engine::zobrist::compute_zobrist_hash;
use chess::{BitBoard, Board, BoardStatus, ChessMove, Color, MoveGen, Piece};
use std::collections::HashMap;
use std::cell::RefCell;
use std::time::Instant;

#[derive(Default, Clone)]
pub struct SearchStats {
    pub nodes: usize,
    pub qnodes: usize,
    pub tt_hits: usize,
    pub seldepth: usize,
}

thread_local! {
    pub static REPETITION_TABLE: RefCell<HashMap<u64, usize>> = RefCell::new(HashMap::new());
    pub static TRANSPOSITION_TABLE: RefCell<TranspositionTable> = RefCell::new(TranspositionTable::new(256));
}

fn is_repetition(position_hash: u64) -> bool {
    REPETITION_TABLE.with(|table| {
        *table.borrow().get(&position_hash).unwrap_or(&0) >= 2
    })
}

pub fn update_repetition_table(position_hash: u64) {
    REPETITION_TABLE.with(|table| {
        *table.borrow_mut().entry(position_hash).or_insert(0) += 1;
    });
}

pub fn clear_repetition_table() {
    REPETITION_TABLE.with(|table| {
        table.borrow_mut().clear();
    });
}

fn quiesce(
    board: &Board,
    mut alpha: i32,
    beta: i32,
    start_time: Instant,
    stats: &mut SearchStats,
    root_color: Color,
    max_time: f64,
    timeout_occurred: &mut bool,
    ply: usize,
) -> i32 {
    stats.nodes += 1;
    stats.qnodes += 1;
    stats.seldepth = stats.seldepth.max(ply);
    if start_time.elapsed().as_secs_f64() > max_time {
        *timeout_occurred = true;
        return evaluate(board, 0, None);
    }

    // Pass beta for lazy eval optimization
    let stand_pat = evaluate(board, 0, Some(beta));

    if stand_pat >= beta {
        return beta;
    }

    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Delta pruning: calculate the maximum possible gain
    // We use queen value (900) as the base delta, plus a margin for promotions
    let big_delta = PIECE_VALUES[Piece::Queen.to_index()] + DELTA_MARGIN;

    // If even capturing the opponent's queen cannot raise alpha, prune all moves
    if stand_pat + big_delta < alpha {
        return alpha;
    }

    let mut captures = Vec::with_capacity(32);
    let movegen = MoveGen::new_legal(board);

    for mv in movegen {
        if board.piece_on(mv.get_dest()).is_some() || mv.get_promotion().is_some() {
            let mvv_lva = mvv_lva_score(board, mv);
            captures.push((mv, mvv_lva));
        }
    }

    captures.sort_by(|a, b| b.1.cmp(&a.1));

    for (mv, _) in captures {
        if *timeout_occurred {
            break;
        }
        
        // Delta pruning for individual moves
        // Calculate the value of the captured piece
        let captured_value = if let Some(captured_piece) = board.piece_on(mv.get_dest()) {
            PIECE_VALUES[captured_piece.to_index()]
        } else if mv.get_promotion().is_some() {
            // Promotion - assume queen for simplicity
            PIECE_VALUES[Piece::Queen.to_index()] - PIECE_VALUES[Piece::Pawn.to_index()]
        } else {
            0
        };
        
        // If the material gain plus the stand_pat score and margin cannot raise alpha, skip this move
        if stand_pat + captured_value + DELTA_MARGIN < alpha {
            continue;
        }
        
        // SEE pruning - skip obviously bad captures
        if !see_capture(board, mv, -50) {
            continue;
        }

        let new_board = board.make_move_new(mv);
        let score = -quiesce(
            &new_board,
            -beta,
            -alpha,
            start_time,
            stats,
            root_color,
            max_time,
            timeout_occurred,
            ply + 1,
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
    board: &Board,
    depth: usize,
    mut alpha: i32,
    mut beta: i32,
    start_time: Instant,
    ply: usize,
    stats: &mut SearchStats,
    root_color: Color,
    tt: &mut TranspositionTable,
    max_time: f64,
    timeout_occurred: &mut bool,
    previous_move: Option<ChessMove>,
    contempt: i32,
) -> i32 {
    let is_pv = beta - alpha > 1;
    let is_root = ply == 0;
    let original_alpha = alpha;

    stats.nodes += 1;
    stats.seldepth = stats.seldepth.max(ply);

    // Timeout check
    if start_time.elapsed().as_secs_f64() > max_time {
        *timeout_occurred = true;
        return if ply > 0 { evaluate(board, 0, None) } else { 0 };
    }

    let position_hash = compute_zobrist_hash(board);

    // Draw detection and mate distance pruning
    if ply > 0 {
        if is_repetition(position_hash) {
            return if board.side_to_move() == root_color {
                -contempt
            } else {
                contempt
            };
        }

        if board.status() == BoardStatus::Checkmate {
            return -30000 + ply as i32;
        }

        if board.status() == BoardStatus::Stalemate {
            return if board.side_to_move() == root_color {
                -contempt
            } else {
                contempt
            };
        }

        // Mate distance pruning
        alpha = alpha.max(-30000 + ply as i32);
        beta = beta.min(30000 - ply as i32);
        if alpha >= beta {
            return alpha;
        }
    }

    let board_hash = position_hash;
    let mut hash_move = None;
    let mut tt_value = None;

    // TT probe
    if let Some(stored_value) = tt.get(board_hash, depth, alpha, beta) {
        stats.tt_hits += 1;
        if !is_pv || !is_root {
            return stored_value;
        }
        tt_value = Some(stored_value);
    }

    if hash_move.is_none() {
        hash_move = tt.get_move(board_hash);
    }
    
    let in_check = *board.checkers() != BitBoard(0);
    let static_eval = if in_check {
        -30000 + ply as i32
    } else {
        // Pass beta to trigger lazy eval
        evaluate(board, contempt, Some(beta))
    };


    // Quiescence at leaf
    if depth == 0 {
        return quiesce(
            board,
            alpha,
            beta,
            start_time,
            stats,
            root_color,
            max_time,
            timeout_occurred,
            ply,
        );
    }

    if !is_pv && !in_check {
        // Razoring
        if depth <= RAZOR_DEPTH {
            let razor_margin = RAZOR_MARGIN + 50 * (depth as i32 - 1);
            if static_eval + razor_margin < alpha {
                let razor_alpha = alpha - razor_margin;
                let razor_score = quiesce(
                    board,
                    razor_alpha,
                    razor_alpha + 1,
                    start_time,
                    stats,
                    root_color,
                    max_time,
                    timeout_occurred,
                    ply,
                );
                if *timeout_occurred {
                    return evaluate(board, 0, None);
                }
                if razor_score <= razor_alpha {
                    return razor_score;
                }
            }
        }

        // Reverse futility pruning
        if depth <= REVERSE_FUTILITY_DEPTH {
            let rfp_margin = REVERSE_FUTILITY_MARGIN * depth as i32;
            
            if static_eval - rfp_margin >= beta {
                return static_eval;
            }
        }

        // Null move pruning
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
                        max_time,
                        timeout_occurred,
                        None,
                        contempt,
                    );

                    if *timeout_occurred {
                        return evaluate(board, 0, None);
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

            // TT early exit: if we have a cached shallow result already proving
            if let Some(tt_val) = tt.get(board_hash, depth - 3, probcut_beta - 1, probcut_beta) {
                if tt_val >= probcut_beta {
                    return tt_val;
                }
            }

            // Collect only captures/promotions whose SEE clears the threshold.
            let mut probcut_moves: Vec<ChessMove> = MoveGen::new_legal(board)
                .filter(|mv| {
                    (board.piece_on(mv.get_dest()).is_some()
                        || mv.get_promotion().is_some())
                        && see_capture(board, *mv, probcut_see_threshold)
                })
                .collect();

            // MVV-LVA ordering: try most promising captures first to get
            // a cutoff as early as possible.
            probcut_moves.sort_by(|a, b| {
                mvv_lva_score(board, *b).cmp(&mvv_lva_score(board, *a))
            });

            'probcut: for mv in probcut_moves {
                if *timeout_occurred {
                    break 'probcut;
                }

                let new_board = board.make_move_new(mv);
                let new_position_hash = compute_zobrist_hash(&new_board);
                update_repetition_table(new_position_hash);

                let score = -negamax(
                    &new_board,
                    depth - 4,
                    -probcut_beta,
                    -probcut_beta + 1,
                    start_time,
                    ply + 1,
                    stats,
                    root_color,
                    tt,
                    max_time,
                    timeout_occurred,
                    Some(mv),
                    contempt,
                );

                REPETITION_TABLE.with(|rep_table| {
                    let mut rep_table = rep_table.borrow_mut();
                    if let Some(count) = rep_table.get_mut(&new_position_hash) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            rep_table.remove(&new_position_hash);
                        }
                    }
                });

                if *timeout_occurred {
                    break 'probcut;
                }

                if score >= probcut_beta {
                    // Cache result so future shallower searches skip this work.
                    tt.store(board_hash, depth - 3, score, TTFlag::Lower, Some(mv));
                    return score;
                }
            }

            if *timeout_occurred {
                return evaluate(board, 0, None);
            }
        }
    }

    // IID
    if is_pv && depth >= IID_DEPTH && hash_move.is_none() {
        let iid_depth = depth - 2 - if is_pv { 0 } else { 1 };
        negamax(
            board,
            iid_depth,
            alpha,
            beta,
            start_time,
            ply,
            stats,
            root_color,
            tt,
            max_time,
            timeout_occurred,
            previous_move,
            contempt,
        );
        if !*timeout_occurred {
            hash_move = tt.get_move(board_hash);
        }
    }

    // Singular extensions 
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
                
                let movegen = MoveGen::new_legal(board);
                
                for mv in movegen {
                    if mv == hash_mv {
                        continue;
                    }
                    
                    if *timeout_occurred {
                        is_singular = false;
                        break;
                    }
                    
                    let new_board = board.make_move_new(mv);
                    let new_position_hash = compute_zobrist_hash(&new_board);
                    update_repetition_table(new_position_hash);
                    
                    let score = -negamax(
                        &new_board,
                        singular_depth,
                        -singular_beta - 1,
                        -singular_beta,
                        start_time,
                        ply + 1,
                        stats,
                        root_color,
                        tt,
                        max_time,
                        timeout_occurred,
                        None,
                        contempt,
                    );
                    
                    REPETITION_TABLE.with(|rep_table| {
                        let mut rep_table = rep_table.borrow_mut();
                        if let Some(count) = rep_table.get_mut(&new_position_hash) {
                            *count = count.saturating_sub(1);
                            if *count == 0 {
                                rep_table.remove(&new_position_hash);
                            }
                        }
                    });
                    
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
                    
                    // Double extensions
                    if depth >= DOUBLE_EXTENSION_DEPTH 
                        && best_singular_score < singular_beta - DOUBLE_EXTENSION_MARGIN
                    {
                        singular_extension = 2;
                    }
                }
            }
        }
    }

    // Mate threat extension
    let mut mate_threat_extension = 0;
    if !is_pv 
        && !in_check 
        && depth >= MATE_THREAT_DEPTH 
        && static_eval < beta - 300
    {
        if let Some(null_board) = board.null_move() {
            let threat_score = -negamax(
                &null_board,
                depth.saturating_sub(3),
                -beta,
                -beta + 1,
                start_time,
                ply + 1,
                stats,
                root_color,
                tt,
                max_time,
                timeout_occurred,
                None,
                contempt,
            );
            
            if !*timeout_occurred && threat_score >= beta && threat_score.abs() > 29000 {
                mate_threat_extension = 1;
            }
        }
    }

    let mut moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();

    if moves.is_empty() {
        return if in_check { -30000 + ply as i32 } else { 0 };
    }

    moves = order_moves(board, moves, hash_move, ply, previous_move);

    let mut best_value = -31000;
    let mut best_move = None;
    let mut moves_searched = 0;
    let mut quiet_moves_searched = 0;

    let mut killer_moves = [None, None];
    if ply < 64 {
        KILLER_MOVES.with(|k| {
            let k = k.borrow();
            if k.len() > ply {
                if k[ply].len() > 0 { killer_moves[0] = Some(k[ply][0]); }
                if k[ply].len() > 1 { killer_moves[1] = Some(k[ply][1]); }
            }
        });
    }

    for (i, mv) in moves.iter().enumerate() {
        if *timeout_occurred {
            break;
        }

        let is_capture = board.piece_on(mv.get_dest()).is_some();
        let is_promotion = mv.get_promotion().is_some();
        let is_quiet = !is_capture && !is_promotion;
        let is_hash_move = hash_move.is_some() && *mv == hash_move.unwrap();

        // Late move pruning
        if !is_pv && !in_check && is_quiet && depth <= LMP_DEPTH {
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

            if quiet_moves_searched >= lmp_threshold {
                continue;
            }
        }

        // Futility pruning
        if !is_pv && !in_check && is_quiet && depth <= FUTILITY_DEPTH && i > 0 {
            let futility_margin = FUTILITY_MARGINS[depth.min(4)];

            let futility_value = static_eval + futility_margin;
            if futility_value <= alpha {
                continue;
            }
        }

        // Futility move count pruning
        if !is_pv && !in_check && is_quiet && depth <= 4 && moves_searched > 0 {
            let futility_value = static_eval + FUTILITY_MARGINS[depth.min(4)];
            if futility_value <= alpha {
                let move_count_limit = FUTILITY_MOVE_COUNTS[depth.min(4)];
                if quiet_moves_searched >= move_count_limit {
                    continue;
                }
            }
        }

        // History leaf pruning
        if !is_pv 
            && !in_check 
            && is_quiet 
            && depth <= HISTORY_PRUNING_DEPTH 
            && moves_searched > 0 
            && !is_hash_move
        {
            let history_score = HISTORY_HEURISTIC.with(|history| {
                history.borrow()[mv.get_source().to_index()][mv.get_dest().to_index()]
            });
            
            if history_score < HISTORY_PRUNING_THRESHOLD {
                continue;
            }
        }

        let new_board = board.make_move_new(*mv);
        let new_position_hash = compute_zobrist_hash(&new_board);
        update_repetition_table(new_position_hash);

        let gives_check = *new_board.checkers() != BitBoard(0);

        // Extensions
        let mut extension = mate_threat_extension;
        
        if is_hash_move && singular_extension > 0 {
            extension = extension.max(singular_extension);
        } else if gives_check && see_capture(board, *mv, 0) {
            extension += 1;
        } else if let Some(piece) = board.piece_on(mv.get_source()) {
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
        let is_killer = Some(*mv) == killer_moves[0] || Some(*mv) == killer_moves[1];

        if depth >= 3 
            && i >= LMR_FULL_DEPTH_MOVES 
            && is_quiet 
            && !gives_check 
            && !is_killer
        {
            // Base formula
            let mut r_f = 0.55 + (depth as f32).ln() * (i as f32).ln() / 1.85;

            // PV nodes need less reduction
            if !is_pv {
                r_f += 0.70;
            } else {
                r_f -= 0.20;
            }

            // History-based adjustments with better scaling
            let history_bonus = HISTORY_HEURISTIC.with(|history| {
                let history_score = history.borrow()[mv.get_source().to_index()][mv.get_dest().to_index()];
                (history_score as f32 / 7000.0).clamp(-1.25, 1.25)
            });
            r_f -= history_bonus;

            if (static_eval - alpha).abs() > 150 {
                r_f -= 0.25;
            }

            // Reduce more for very late moves
            if i > 15 {
                r_f += 0.5;
            }

            let mut r = r_f.max(0.0) as usize;

            r = r.min(depth.saturating_sub(1));
            new_depth = new_depth.saturating_sub(r);
            do_full_search = false;
        }

        // PVS
        let score = if i == 0 {
            -negamax(
                &new_board,
                new_depth,
                -beta,
                -alpha,
                start_time,
                ply + 1,
                stats,
                root_color,
                tt,
                max_time,
                timeout_occurred,
                Some(*mv),
                contempt,
            )
        } else {
            let mut search_score = -negamax(
                &new_board,
                new_depth,
                -alpha - 1,
                -alpha,
                start_time,
                ply + 1,
                stats,
                root_color,
                tt,
                max_time,
                timeout_occurred,
                Some(*mv),
                contempt,
            );

            if *timeout_occurred {
                evaluate(board, 0, None)
            } else {
                // Re-search at full depth if reduced
                if !do_full_search && search_score > alpha {
                    new_depth = depth - 1 + extension;
                    search_score = -negamax(
                        &new_board,
                        new_depth,
                        -alpha - 1,
                        -alpha,
                        start_time,
                        ply + 1,
                        stats,
                        root_color,
                        tt,
                        max_time,
                        timeout_occurred,
                        Some(*mv),
                        contempt,
                    );
                }

                // PV re-search
                if *timeout_occurred {
                    evaluate(board, 0, None)
                } else if search_score > alpha {
                    if is_pv || (search_score < beta) {
                        -negamax(
                            &new_board,
                            new_depth,
                            -beta,
                            -alpha,
                            start_time,
                            ply + 1,
                            stats,
                            root_color,
                            tt,
                            max_time,
                            timeout_occurred,
                            Some(*mv),
                            contempt,
                        )
                    } else {
                        search_score
                    }
                } else {
                    search_score
                }
            }
        };

        REPETITION_TABLE.with(|rep_table| {
            let mut rep_table = rep_table.borrow_mut();
            if let Some(count) = rep_table.get_mut(&new_position_hash) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    rep_table.remove(&new_position_hash);
                }
            }
        });

        if *timeout_occurred {
            break;
        }

        moves_searched += 1;
        if is_quiet {
            quiet_moves_searched += 1;
        }

        if score > best_value {
            best_value = score;
            best_move = Some(*mv);
        }

        if score > alpha {
            alpha = score;

            if is_quiet {
                // Killer moves
                KILLER_MOVES.with(|killers| {
                    let mut killers = killers.borrow_mut();
                    if ply < 64 && killers.len() > ply {
                        if killers[ply].is_empty() || killers[ply][0] != *mv {
                            if killers[ply].len() >= 2 {
                                killers[ply][1] = killers[ply][0];
                            } else if killers[ply].len() == 1 {
                                let first_killer = killers[ply][0];
                                killers[ply].push(first_killer);
                            }
                            if killers[ply].is_empty() {
                                killers[ply].push(*mv);
                            } else {
                                killers[ply][0] = *mv;
                            }
                        }
                    }
                });

                // History heuristic
                HISTORY_HEURISTIC.with(|history| {
                    let mut history = history.borrow_mut();
                    
                    // Standard bonus scaled by depth
                    let bonus = (depth * depth).min(400) as i32; 
                    
                    let from_sq = mv.get_source().to_index();
                    let to_sq = mv.get_dest().to_index();
                    
                    let current = history[from_sq][to_sq];
                    
                    // Gravity formula: 
                    // approach the target (bonus) while decaying the current value
                    // 10000 is the arbitrary clamp, used here as the scaling divisor
                    history[from_sq][to_sq] = current + bonus - (current * bonus.abs()) / 10000;
                });

                // Reduce history for failed moves
                for prev_mv in &moves[0..i] {
                    if board.piece_on(prev_mv.get_dest()).is_none()
                        && prev_mv.get_promotion().is_none()
                    {
                        HISTORY_HEURISTIC.with(|history| {
                            let mut history = history.borrow_mut();
                            let prev_from = prev_mv.get_source().to_index();
                            let prev_to = prev_mv.get_dest().to_index();
                            
                            let penalty = -((depth * depth).min(400) as i32);
                            let current = history[prev_from][prev_to];
                            
                            // Same gravity formula for penalty
                            history[prev_from][prev_to] = current + penalty - (current * penalty.abs()) / 10000;
                        });
                    }
                }
            }
            else {
                // Capture succeeded - update capture history
                update_capture_history(*mv, board, depth, i > 0);
                
                // Penalize failed captures
                for prev_mv in &moves[0..i] {
                    if board.piece_on(prev_mv.get_dest()).is_some() || prev_mv.get_promotion().is_some() {
                        penalize_capture_history(*prev_mv, board, depth);
                    }
                }
            }
        }

        if alpha >= beta {
            if is_quiet {
                update_counter_move(previous_move, *mv);
            }
            break;
        }
    }

    // TT store
    if !*timeout_occurred && moves_searched > 0 {
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
) -> Option<ChessMove> {
    let start_time = Instant::now();
    let mut best_move = None;
    let mut best_score = 0;
    let root_color = board.side_to_move();

    TRANSPOSITION_TABLE.with(|tt| {
        let mut tt_guard = tt.borrow_mut();
        let mut stats = SearchStats::default();

        let max_search_depth = max_depth.unwrap_or(MAX_DEPTH);

        for depth in 1..=max_search_depth {
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
                (-31000, 31000)  // Full window for early depths
            } else {
                (best_score - INITIAL_WINDOW, best_score + INITIAL_WINDOW)
            };

            let mut alpha = initial_alpha;
            let mut beta = initial_beta;
            let mut window_size = INITIAL_WINDOW;
            let mut search_iterations = 0;

            // Aspiration window re-search loop
            loop {
                search_iterations += 1;

                let score = negamax(
                    board,
                    depth,
                    alpha,
                    beta,
                    start_time,
                    0,
                    &mut stats,
                    root_color,
                    &mut tt_guard,
                    max_time,
                    &mut timeout_occurred,
                    None,
                    contempt,
                );

                if timeout_occurred {
                    break;
                }

                // Check if we failed low or high
                if score <= alpha {
                    // FAIL LOW: True score is <= alpha
                    // Widen window exponentially
                    window_size = (window_size * 2).min(MAX_WINDOW);

                    if window_size >= MAX_WINDOW || search_iterations >= MAX_ASPIRATION_ITERATIONS {
                        // Give up, use full window
                        alpha = -31000;
                        beta = 31000;
                        if STATS {
                            println!("info string Giving up on aspiration, using full window");
                        }
                    } else {
                        // Adjust window symmetrically around best score
                        alpha = (best_score - window_size).max(-31000);
                        beta = (best_score + window_size).min(31000);

                        if STATS {
                            println!("info string Re-searching with window [{}, {}]", alpha, beta);
                        }
                    }
                } else if score >= beta {
                    // FAIL HIGH: True score is >= beta
                    // Widen window exponentially
                    window_size = (window_size * 2).min(MAX_WINDOW);

                    if window_size >= MAX_WINDOW || search_iterations >= MAX_ASPIRATION_ITERATIONS {
                        // Give up, use full window
                        alpha = -31000;
                        beta = 31000;
                        if STATS {
                            println!("info string Giving up on aspiration, using full window");
                        }
                    } else {
                        // Adjust window symmetrically around best score
                        alpha = (best_score - window_size).max(-31000);
                        beta = (best_score + window_size).min(31000);

                        if STATS {
                            println!("info string Re-searching with window [{}, {}]", alpha, beta);
                        }
                    }
                } else {
                    // SUCCESS: Score is within [alpha, beta]
                    best_score = score;

                    let board_hash = compute_zobrist_hash(board);
                    if let Some(mv) = tt_guard.get_move(board_hash) {
                        let movegen = MoveGen::new_legal(board);
                        if movegen.into_iter().any(|legal_move| legal_move == mv) {
                            best_move = Some(mv);

                            if STATS {
                                // Extract PV
                                let mut pv = Vec::new();
                                let mut temp_board = *board;

                                for _ in 0..depth.min(20) {
                                    let h = compute_zobrist_hash(&temp_board);
                                    if let Some(pv_move) = tt_guard.get_move(h) {
                                        let movegen = MoveGen::new_legal(&temp_board);
                                        if movegen.into_iter().any(|legal_move| legal_move == pv_move) {
                                            pv.push(pv_move);
                                            temp_board = temp_board.make_move_new(pv_move);
                                        } else {
                                            break;
                                        }
                                    } else {
                                        break;
                                    }
                                }

                                let time_ms = (start_time.elapsed().as_secs_f64() * 1000.0) as u64;
                                let nps = if time_ms > 0 {
                                    stats.nodes * 1000 / time_ms as usize
                                } else {
                                    0
                                };
                                let pv_string = pv
                                    .iter()
                                    .map(|m| format!("{}", m))
                                    .collect::<Vec<_>>()
                                    .join(" ");

                                let hashfull = tt_guard.hashfull();

                                println!(
                                    "info depth {} seldepth {} score cp {} nodes {} nps {} time {} hashfull {} pv {}",
                                    depth,
                                    stats.seldepth,
                                    best_score,
                                    stats.nodes,
                                    nps,
                                    time_ms,
                                    hashfull,
                                    pv_string
                                );
                            }

                            // Early exit if mate found
                            if best_score.abs() > 29000 {
                                break;
                            }
                        }
                    }
                    break; // Exit aspiration loop on success
                }

                // Safety check
                if timeout_occurred || search_iterations >= MAX_ASPIRATION_ITERATIONS * 2 {
                    if STATS && search_iterations >= MAX_ASPIRATION_ITERATIONS * 2 {
                        println!(
                            "info string Aspiration window abandoned after {} iterations",
                            search_iterations
                        );
                    }
                    break;
                }
            }

            if timeout_occurred {
                break;
            }
        }

        best_move.or_else(|| {
            let moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
            moves.first().copied()
        })
    })
}

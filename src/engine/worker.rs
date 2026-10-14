use chess::{Board, ChessMove};
use crate::engine::evaluation::PawnHashTable;
use crate::engine::move_ordering::{ContinuationHistoryTable, ScoredMove};

#[derive(Default, Clone)]
pub struct SearchStats {
    pub nodes: usize,
    pub qnodes: usize,
    pub tt_hits: usize,
    pub seldepth: usize,
}

pub struct Worker {
    pub history: [[i32; 64]; 64],
    pub capture_history: [[[i32; 64]; 6]; 6],
    pub counter_moves: [[Option<ChessMove>; 64]; 6],
    pub cmh_table: Box<ContinuationHistoryTable>,
    pub fmh_table: Box<ContinuationHistoryTable>,
    pub killers: [[Option<ChessMove>; 2]; 64],
    pub repetition_table: Vec<u64>,
    pub stats: SearchStats,
    pub pawn_table: PawnHashTable,
}

impl Worker {
    pub fn new(pawn_hash_mb: usize) -> Self {
        Self {
            history: [[0; 64]; 64],
            capture_history: [[[0; 64]; 6]; 6],
            counter_moves: [[None; 64]; 6],
            cmh_table: Box::new([[[[0; 64]; 6]; 64]; 6]),
            fmh_table: Box::new([[[[0; 64]; 6]; 64]; 6]),
            killers: [[None; 2]; 64],
            repetition_table: Vec::with_capacity(1024),
            stats: SearchStats::default(),
            pawn_table: PawnHashTable::new(pawn_hash_mb),
        }
    }

    pub fn clear_for_game(&mut self) {
        self.stats = SearchStats::default();
        self.killers = [[None; 2]; 64];
        self.history = [[0; 64]; 64];
        self.capture_history = [[[0; 64]; 6]; 6];
        self.counter_moves = [[None; 64]; 6];
        *self.cmh_table = [[[[0; 64]; 6]; 64]; 6];
        *self.fmh_table = [[[[0; 64]; 6]; 64]; 6];
        self.pawn_table.clear();
        self.repetition_table.clear();
    }

    pub fn clear_for_search(&mut self) {
        self.stats = SearchStats::default();
        self.killers = [[None; 2]; 64];
        self.repetition_table.clear();
    }

    #[inline(always)]
    pub fn push_repetition(&mut self, hash: u64) {
        self.repetition_table.push(hash);
    }

    #[inline(always)]
    pub fn pop_repetition(&mut self) {
        self.repetition_table.pop();
    }

    #[inline(always)]
    pub fn is_repetition(&self, position_hash: u64) -> bool {
        let mut count = 0;
        for &hash in self.repetition_table.iter().rev() {
            if hash == position_hash {
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
        }
        false
    }

    #[inline(always)]
    pub fn update_killer(&mut self, ply: usize, mv: ChessMove) {
        if ply < 64 {
            if self.killers[ply][0] != Some(mv) {
                self.killers[ply][1] = self.killers[ply][0];
                self.killers[ply][0] = Some(mv);
            }
        }
    }

    #[inline(always)]
    pub fn update_history(&mut self, mv: ChessMove, bonus: i32) {
        let from = mv.get_source().to_index();
        let to = mv.get_dest().to_index();
        let current = self.history[from][to];
        self.history[from][to] = current + bonus - (current * bonus.abs()) / 10000;
    }

    #[inline(always)]
    pub fn penalize_history(&mut self, mv: ChessMove, penalty: i32) {
        let from = mv.get_source().to_index();
        let to = mv.get_dest().to_index();
        let current = self.history[from][to];
        self.history[from][to] = current + penalty - (current * penalty.abs()) / 10000;
    }

    #[inline(always)]
    pub fn update_capture_history(&mut self, board: &Board, mv: ChessMove, bonus: i32) {
        if let (Some(vic), Some(att)) = (board.piece_on(mv.get_dest()), board.piece_on(mv.get_source())) {
            let att_idx = att.to_index();
            let vic_idx = vic.to_index();
            let dest_idx = mv.get_dest().to_index();
            let current = self.capture_history[att_idx][vic_idx][dest_idx];
            self.capture_history[att_idx][vic_idx][dest_idx] = current + bonus - (current * bonus.abs()) / 10000;
        }
    }

    #[inline(always)]
    pub fn update_capture_history_batch(
        &mut self,
        board: &Board,
        cutoff_mv: ChessMove,
        failed_moves: &[ScoredMove],
        bonus: i32,
    ) {
        self.update_capture_history(board, cutoff_mv, bonus);
        let penalty = -bonus;
        for prev_scored_mv in failed_moves {
            self.update_capture_history(board, prev_scored_mv.mv, penalty);
        }
    }

    #[inline(always)]
    pub fn update_counter_move(&mut self, board: &Board, previous_move: Option<ChessMove>, mv: ChessMove) {
        if let Some(prev) = previous_move {
            if let Some(p) = board.piece_on(prev.get_dest()) {
                self.counter_moves[p.to_index()][prev.get_dest().to_index()] = Some(mv);
            }
        }
    }

    #[inline(always)]
    pub fn update_conthist_batch(
        &mut self,
        is_cmh: bool,
        board: &Board,
        prev_mv: Option<ChessMove>,
        cutoff_mv: ChessMove,
        failed_quiets: &[ScoredMove],
        bonus: i32,
    ) {
        let table = if is_cmh { &mut self.cmh_table } else { &mut self.fmh_table };
        let prev = match prev_mv {
            Some(m) => m,
            None => return,
        };
        let prev_piece = match board.piece_on(prev.get_dest()) {
            Some(p) => p.to_index(),
            None => return,
        };
        let prev_to = prev.get_dest().to_index();

        if let Some(curr_piece) = board.piece_on(cutoff_mv.get_source()) {
            let curr_p = curr_piece.to_index();
            let curr_to = cutoff_mv.get_dest().to_index();
            let current = table[prev_piece][prev_to][curr_p][curr_to] as i32;
            let new_val = current + bonus - (current * bonus.abs()) / crate::engine::constants::CONTHIST_GRAVITY;
            table[prev_piece][prev_to][curr_p][curr_to] = new_val.clamp(-crate::engine::constants::CONTHIST_GRAVITY, crate::engine::constants::CONTHIST_GRAVITY) as i16;
        }

        let penalty = -bonus;
        for scored_mv in failed_quiets {
            let failed_mv = scored_mv.mv;
            if board.piece_on(failed_mv.get_dest()).is_none() && failed_mv.get_promotion().is_none() {
                if let Some(curr_piece) = board.piece_on(failed_mv.get_source()) {
                    let curr_p = curr_piece.to_index();
                    let curr_to = failed_mv.get_dest().to_index();
                    let current = table[prev_piece][prev_to][curr_p][curr_to] as i32;
                    let new_val = current + penalty - (current * penalty.abs()) / crate::engine::constants::CONTHIST_GRAVITY;
                    table[prev_piece][prev_to][curr_p][curr_to] = new_val.clamp(-crate::engine::constants::CONTHIST_GRAVITY, crate::engine::constants::CONTHIST_GRAVITY) as i16;
                }
            }
        }
    }
}

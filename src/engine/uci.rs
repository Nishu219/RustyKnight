use crate::engine::constants::*;
use crate::engine::search::{iterative_deepening, clear_repetition_table, update_repetition_table, TRANSPOSITION_TABLE};
use crate::engine::zobrist::compute_zobrist_hash;
use crate::engine::transposition_table::TranspositionTable;
use crate::engine::move_ordering::{KILLER_MOVES, HISTORY_HEURISTIC, clear_counter_moves, clear_capture_history};
use crate::engine::evaluation::{MATERIAL_HASH_TABLE, PAWN_HASH_TABLE, MaterialHashTable};
use chess::{Board, ChessMove, Color};
use std::io::{self, BufRead};
use std::str::FromStr;

pub struct UCIEngine {
    board: Board,
    debug: bool,
    position_history: Vec<u64>,
    hash_size: usize,
    contempt: i32,
}

impl UCIEngine {
    pub fn new() -> Self {
        UCIEngine {
            board: Board::default(),
            debug: false,
            position_history: Vec::new(),
            hash_size: 256,
            contempt: 0,
        }
    }

    fn handle_uci(&self) {
        println!("id name RustKnightv2.2");
        println!("id author Anish");
        println!("option name Hash type spin default 256 min 1 max 4096");
        println!("option name Contempt type spin default 0 min -100 max 100");
        println!("uciok");
    }

    fn handle_isready(&self) {
        println!("readyok");
    }
    fn handle_setoption(&mut self, tokens: &[&str]) {
        if tokens.len() >= 5 && tokens[1] == "name" && tokens[3] == "value" {
            let option_name = tokens[2].to_lowercase();
            match option_name.as_str() {
                "hash" => {
                    if let Ok(size) = tokens[4].parse::<usize>() {
                        let clamped_size = size.max(1).min(4096);
                        self.hash_size = clamped_size;
                        TRANSPOSITION_TABLE.with(|tt| {
                            *tt.borrow_mut() = TranspositionTable::new(clamped_size);
                        });
                    }
                }
                "contempt" => {
                    if let Ok(value) = tokens[4].parse::<i32>() {
                        self.contempt = value.max(-100).min(100);
                    }
                }
                _ => {}
            }
        }
    }
    fn handle_position(&mut self, tokens: &[&str]) {
        if tokens.len() < 2 {
            return;
        }

        self.position_history.clear();

        match tokens[1] {
            "startpos" => {
                self.board = Board::default();
                self.position_history
                    .push(compute_zobrist_hash(&self.board));

                if tokens.len() > 2 && tokens[2] == "moves" {
                    for move_str in &tokens[3..] {
                        if let Ok(chess_move) = ChessMove::from_str(move_str) {
                            self.board = self.board.make_move_new(chess_move);
                            self.position_history
                                .push(compute_zobrist_hash(&self.board));
                        }
                    }
                }
            }
            "fen" => {
                if tokens.len() < 8 {
                    return;
                }
                let fen = tokens[2..8].join(" ");
                if let Ok(board) = Board::from_str(&fen) {
                    self.board = board;
                    self.position_history
                        .push(compute_zobrist_hash(&self.board));

                    let mut moves_start = None;
                    for (i, &token) in tokens.iter().enumerate() {
                        if token == "moves" {
                            moves_start = Some(i + 1);
                            break;
                        }
                    }

                    if let Some(start) = moves_start {
                        for move_str in &tokens[start..] {
                            if let Ok(chess_move) = ChessMove::from_str(move_str) {
                                self.board = self.board.make_move_new(chess_move);
                                self.position_history
                                    .push(compute_zobrist_hash(&self.board));
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        clear_repetition_table();
        for &hash in &self.position_history {
            update_repetition_table(hash);
        }
    }

    fn calculate_time_budget(
        &self,
        wtime: Option<u64>,
        btime: Option<u64>,
        winc: Option<u64>,
        binc: Option<u64>,
        movetime: Option<u64>,
    ) -> f64 {
        if let Some(mt) = movetime {
            return (mt as f64) / 1000.0;
        }

        let (time_left, increment) = match self.board.side_to_move() {
            Color::White => (wtime.unwrap_or(0), winc.unwrap_or(0)),
            Color::Black => (btime.unwrap_or(0), binc.unwrap_or(0)),
        };

        if time_left == 0 {
            return 1.0;
        }

        let base_time = (time_left as f64) / 1000.0;
        let inc_time = (increment as f64) / 1000.0;

        let moves_to_go = 40;
        let time_per_move = (base_time / moves_to_go as f64) + (inc_time * 0.8);

        time_per_move.max(0.01).min(base_time * 0.1)
    }

    fn handle_go(&self, tokens: &[&str]) {
        let mut wtime = None;
        let mut btime = None;
        let mut winc = None;
        let mut binc = None;
        let mut movetime = None;
        let mut infinite = false;

        let mut i = 1;
        while i < tokens.len() {
            match tokens[i] {
                "wtime" => {
                    if i + 1 < tokens.len() {
                        wtime = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "btime" => {
                    if i + 1 < tokens.len() {
                        btime = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "winc" => {
                    if i + 1 < tokens.len() {
                        winc = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "binc" => {
                    if i + 1 < tokens.len() {
                        binc = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "movetime" => {
                    if i + 1 < tokens.len() {
                        movetime = tokens[i + 1].parse().ok();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "infinite" => {
                    infinite = true;
                    i += 1;
                }
                _ => i += 1,
            }
        }

        let max_time = if infinite {
            10000000.0
        } else {
            self.calculate_time_budget(wtime, btime, winc, binc, movetime)
        };

        let is_movetime = movetime.is_some();

        if let Some(best_move) =
            iterative_deepening(&self.board, max_time, self.contempt, is_movetime)
        {
            println!("bestmove {}", best_move);
        } else {
            println!("bestmove 0000");
        }
    }

    fn handle_debug(&mut self, tokens: &[&str]) {
        if tokens.len() > 1 {
            self.debug = tokens[1] == "on";
        }
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(line) => line,
                Err(_) => break,
            };

            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            match tokens[0] {
                "uci" => self.handle_uci(),
                "debug" => self.handle_debug(&tokens),
                "isready" => self.handle_isready(),
                "setoption" => self.handle_setoption(&tokens),
                "register" => {}
                "ucinewgame" => {
                    self.board = Board::default();
                    self.position_history.clear();
                    self.position_history
                        .push(compute_zobrist_hash(&self.board));
                    TRANSPOSITION_TABLE.with(|tt| {
                        let mut tt = tt.borrow_mut();
                        *tt = TranspositionTable::new(self.hash_size);
                        tt.clear();
                    });
                    KILLER_MOVES.with(|killers| {
                        let mut killers = killers.borrow_mut();
                        killers.clear();
                        killers.resize(MAX_DEPTH, Vec::new());
                    });
                    HISTORY_HEURISTIC.with(|history| {
                        *history.borrow_mut() = [[0; 64]; 64];
                    });
                    MATERIAL_HASH_TABLE.with(|material_table| {
                        *material_table.borrow_mut() = MaterialHashTable::new(16);
                    });
                    PAWN_HASH_TABLE.with(|pawn_table| {
                        pawn_table.borrow_mut().clear();
                    });
                    clear_repetition_table();
                    clear_counter_moves();
                    clear_capture_history();
                }
                "position" => self.handle_position(&tokens),
                "go" => self.handle_go(&tokens),
                "stop" => {}
                "ponderhit" => {}
                "quit" => break,
                _ => {
                    if self.debug {
                        eprintln!("Unknown command: {}", tokens[0]);
                    }
                }
            }
        }
    }
}

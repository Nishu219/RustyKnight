use crate::engine::search::iterative_deepening;
use crate::engine::transposition_table::TranspositionTable;
use crate::engine::worker::Worker;
use crate::engine::zobrist::{compute_zobrist_hash, update_zobrist_hash};
use chess::{Board, ChessMove, Color};
use std::io::{self, BufRead};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

enum SearchCommand {
    NewGame,
    SetOptionHash(usize),
    Go {
        board: Board,
        max_time: f64,
        contempt: i32,
        is_movetime: bool,
        depth: Option<usize>,
        halfmove_clock: u16,
        pos_hist: Vec<u64>,
    },
    Quit,
}

pub struct UCIEngine {
    board: Board,
    debug: bool,
    position_history: Vec<u64>,
    hash_size: usize,
    contempt: i32,
    halfmove_clock: u16,
    stop_flag: Arc<AtomicBool>,
    cmd_tx: std::sync::mpsc::Sender<SearchCommand>,
    search_thread: Option<std::thread::JoinHandle<()>>,
}

impl UCIEngine {
    pub fn new() -> Self {
        let hash_size = 256;
        let stop_flag = Arc::new(AtomicBool::new(false));
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<SearchCommand>();
        let stop_flag_clone = Arc::clone(&stop_flag);

        let search_thread = std::thread::spawn(move || {
            let mut worker = Worker::new(16);
            let mut current_tt = TranspositionTable::new(hash_size);

            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    SearchCommand::NewGame => {
                        worker.clear_for_game();
                        current_tt.clear();
                    }
                    SearchCommand::SetOptionHash(size) => {
                        current_tt = TranspositionTable::new(size);
                    }
                    SearchCommand::Go {
                        board,
                        max_time,
                        contempt,
                        is_movetime,
                        depth,
                        halfmove_clock,
                        pos_hist,
                    } => {
                        worker.clear_for_search();
                        for &hash in &pos_hist {
                            worker.push_repetition(hash);
                        }

                        if let Some(best_move) = iterative_deepening(
                            &board,
                            max_time,
                            contempt,
                            is_movetime,
                            depth,
                            halfmove_clock,
                            &mut worker,
                            &current_tt,
                            &stop_flag_clone,
                        ) {
                            println!("bestmove {}", best_move);
                        } else {
                            println!("bestmove 0000");
                        }
                    }
                    SearchCommand::Quit => break,
                }
            }
        });

        UCIEngine {
            board: Board::default(),
            debug: false,
            position_history: Vec::new(),
            hash_size,
            contempt: 0,
            halfmove_clock: 0,
            stop_flag,
            cmd_tx,
            search_thread: Some(search_thread),
        }
    }

    fn stop_search(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    fn handle_uci(&self) {
        println!("id name RustyKnightv2.3");
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
                        self.stop_search();
                        self.hash_size = clamped_size;
                        let _ = self.cmd_tx.send(SearchCommand::SetOptionHash(clamped_size));
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

        self.stop_search();
        self.position_history.clear();

        match tokens[1] {
            "startpos" => {
                self.board = Board::default();
                self.halfmove_clock = 0;
                let mut current_hash = compute_zobrist_hash(&self.board);
                self.position_history.push(current_hash);

                if tokens.len() > 2 && tokens[2] == "moves" {
                    for move_str in &tokens[3..] {
                        if let Ok(chess_move) = ChessMove::from_str(move_str) {
                            let is_capture = self.board.piece_on(chess_move.get_dest()).is_some();
                            let is_pawn = self.board.piece_on(chess_move.get_source()) == Some(chess::Piece::Pawn);
                            if is_capture || is_pawn {
                                self.halfmove_clock = 0;
                            } else {
                                self.halfmove_clock += 1;
                            }

                            let new_board = self.board.make_move_new(chess_move);
                            current_hash = update_zobrist_hash(&self.board, &new_board, current_hash);
                            self.board = new_board;
                            self.position_history.push(current_hash);
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
                    self.halfmove_clock = tokens.get(6).and_then(|s| s.parse::<u16>().ok()).unwrap_or(0);

                    let mut current_hash = compute_zobrist_hash(&self.board);
                    self.position_history.push(current_hash);

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
                                let is_capture = self.board.piece_on(chess_move.get_dest()).is_some();
                                let is_pawn = self.board.piece_on(chess_move.get_source()) == Some(chess::Piece::Pawn);
                                if is_capture || is_pawn {
                                    self.halfmove_clock = 0;
                                } else {
                                    self.halfmove_clock += 1;
                                }

                                let new_board = self.board.make_move_new(chess_move);
                                current_hash = update_zobrist_hash(&self.board, &new_board, current_hash);
                                self.board = new_board;
                                self.position_history.push(current_hash);
                            }
                        }
                    }
                }
            }
            _ => {}
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

    fn handle_go(&mut self, tokens: &[&str]) {
        self.stop_search();
        self.stop_flag.store(false, Ordering::Relaxed);

        let mut wtime = None;
        let mut btime = None;
        let mut winc = None;
        let mut binc = None;
        let mut movetime = None;
        let mut depth = None;
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
                "depth" => {
                    if i + 1 < tokens.len() {
                        depth = tokens[i + 1].parse().ok();
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
        let _ = self.cmd_tx.send(SearchCommand::Go {
            board: self.board,
            max_time,
            contempt: self.contempt,
            is_movetime,
            depth,
            halfmove_clock: self.halfmove_clock,
            pos_hist: self.position_history.clone(),
        });
    }

    fn handle_debug(&mut self, tokens: &[&str]) {
        if tokens.len() > 1 {
            self.debug = tokens[1] == "on";
        }
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        let mut line = String::new();
        let mut reader = stdin.lock();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }

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
                    self.stop_search();
                    self.board = Board::default();
                    self.halfmove_clock = 0;
                    self.position_history.clear();
                    self.position_history.push(compute_zobrist_hash(&self.board));
                    let _ = self.cmd_tx.send(SearchCommand::NewGame);
                }
                "position" => self.handle_position(&tokens),
                "go" => self.handle_go(&tokens),
                "stop" => {
                    self.stop_search();
                }
                "ponderhit" => {}
                "quit" => {
                    self.stop_search();
                    let _ = self.cmd_tx.send(SearchCommand::Quit);
                    if let Some(handle) = self.search_thread.take() {
                        let _ = handle.join();
                    }
                    break;
                }
                _ => {
                    if self.debug {
                        eprintln!("Unknown command: {}", tokens[0]);
                    }
                }
            }
        }

        self.stop_search();
        let _ = self.cmd_tx.send(SearchCommand::Quit);
        if let Some(handle) = self.search_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for UCIEngine {
    fn drop(&mut self) {
        self.stop_search();
        let _ = self.cmd_tx.send(SearchCommand::Quit);
        if let Some(handle) = self.search_thread.take() {
            let _ = handle.join();
        }
    }
}

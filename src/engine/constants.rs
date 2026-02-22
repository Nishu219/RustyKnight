use chess::Piece;

pub const MATE_THREAT_DEPTH: usize = 4;
pub const MAX_DEPTH: usize = 99;
pub const MAX_ASPIRATION_ITERATIONS: usize = 6;
pub const INITIAL_WINDOW: i32 = 50;
pub const MAX_WINDOW: i32 = 400;
pub const STATS: bool = true;
pub const LMR_FULL_DEPTH_MOVES: usize = 4;
pub const NULL_MOVE_DEPTH: usize = 3;
pub const IID_DEPTH: usize = 4;
pub const RAZOR_DEPTH: usize = 4;
pub const RAZOR_MARGIN: i32 = 400;
pub const DELTA_MARGIN: i32 = 200;
pub const REVERSE_FUTILITY_DEPTH: usize = 6;
pub const REVERSE_FUTILITY_MARGIN: i32 = 120;
pub const FUTILITY_DEPTH: usize = 4;
pub const HISTORY_PRUNING_DEPTH: usize = 3;
pub const HISTORY_PRUNING_THRESHOLD: i32 = -800;
pub const FUTILITY_MOVE_COUNTS: [usize; 5] = [0, 3, 5, 8, 12];
pub const FUTILITY_MARGINS: [i32; 5] = [0, 200, 400, 600, 800];
pub const LMP_DEPTH: usize = 4;
pub const TROPISM_WEIGHTS: [(i32, i32); 4] = [(3, 2), (2, 3), (1, 1), (2, 1)];
pub const ATTACK_WEIGHTS: [(i32, i32); 5] = [(4, 2), (3, 4), (2, 1), (3, 1), (1, 0)];
pub const KNIGHT_MOBILITY_MG: [i32; 9] = [-25, -15, -10, -5, 0, 5, 10, 15, 20];
pub const KNIGHT_MOBILITY_EG: [i32; 9] = [-30, -20, -15, -10, -5, 0, 5, 10, 15];
pub const BISHOP_MOBILITY_MG: [i32; 14] = [-20, -15, -10, -5, 0, 5, 8, 12, 15, 18, 20, 22, 24, 25];
pub const BISHOP_MOBILITY_EG: [i32; 14] = [-15, -10, -5, 0, 5, 10, 13, 16, 19, 22, 25, 27, 29, 31];
pub const ROOK_MOBILITY_MG: [i32; 15] = [-15, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 15];
pub const ROOK_MOBILITY_EG: [i32; 15] = [-10, -5, -3, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22];
pub const QUEEN_MOBILITY_MG: [i32; 28] = [-10, -8, -6, -4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
pub const QUEEN_MOBILITY_EG: [i32; 28] = [-5, -3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
pub const PIECE_ORDER: [Piece; 5] = [
    Piece::Queen,
    Piece::Rook,
    Piece::Bishop,
    Piece::Knight,
    Piece::Pawn,
];
pub const SINGULAR_DEPTH: usize = 8;
pub const SINGULAR_MARGIN_MULTIPLIER: i32 = 2;
pub const DOUBLE_EXTENSION_MARGIN: i32 = 16;
pub const DOUBLE_EXTENSION_DEPTH: usize = 6;
pub const WHITE_PHALANX_BONUS: [i32; 8] = [0, 3, 5, 8, 12, 18, 25, 0,];
pub const BLACK_PHALANX_BONUS: [i32; 8] = [0, 25, 18, 12, 8, 5, 3, 0,];
pub const LAZY_EVAL_MARGIN: i32 = 200; 
pub const PAWN_ATTACK_THREAT: [(i32, i32); 4] = [(50, 40), (50, 40), (60, 50), (75, 65)];
pub const TIME_CHECK_INTERVAL: usize = 4096;

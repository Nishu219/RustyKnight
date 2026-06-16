mod engine;
use engine::uci::UCIEngine;

fn main() {
    engine::constants::init_lmr_table();
    let mut engine = UCIEngine::new();
    engine.run();
}

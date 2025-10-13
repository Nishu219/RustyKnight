mod engine;
use engine::uci::UCIEngine;

fn main() {
    let mut engine = UCIEngine::new();
    engine.run();
}

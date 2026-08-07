# RustyKnight Chess Engine

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.3-orange.svg)

RustyKnight is a fast and efficient UCI (Universal Chess Interface) chess engine written in Rust. It focuses on speed and clarity, implementing solid evaluation heuristics and advanced search techniques to provide strong gameplay.

## Features

- **UCI Protocol Support:** Fully compatible with standard chess GUIs via the Universal Chess Interface.
- **Zobrist Hashing:** Efficient position hashing for transposition tables and caching.
- **Advanced Search Algorithms:** Implements Negamax with alpha-beta pruning, Iterative Deepening, Quiescence Search, and various pruning techniques (Null Move Pruning, Late Move Reductions, Late Move Pruning, Futility Pruning, Internal Iterative Deepening).
- **Evaluation Function:** Includes heuristics for material, piece-square tables, pawn structure, rook positioning, and special positions (checkmate/stalemate).

---

## Code Structure

All main logic is implemented in `src/main.rs`.
For details, refer to function definitions such as `evaluate`, `negamax`, `order_moves`, `quiesce`, `evaluate_pawn_structure`, and `iterative_deepening`.

---

## Installation and Usage

### Prerequisites

To build RustyKnight, you need to have Rust and Cargo installed. If you don't have them, you can install them from [rustup.rs](https://rustup.rs/).

### Building

Clone the repository and build the engine using Cargo:

```bash
git clone https://github.com/yourusername/RustyKnight.git
cd RustyKnight
cargo build --release
```

The compiled executable will be located in `target/release/RustKnight`.

### Running with a GUI

RustyKnight is a command-line application that communicates via the UCI protocol. To play against it or have it analyze games, you need to load it into a chess GUI that supports UCI.

1. Open your preferred chess GUI (e.g., Arena, Cute Chess, Lucas Chess).
2. Add a new UCI engine and select the compiled `RustKnight` executable from the `target/release/` directory.

### Running from the Command Line

You can also run RustyKnight directly from the terminal to interact with it using UCI commands manually:

```bash
cargo run --release
```

**Basic UCI Commands:**
- `uci`: Initialize the engine and list supported options.
- `isready`: Check if the engine is ready.
- `position startpos`: Set the board to the starting position.
- `go depth 10`: Tell the engine to search up to depth 10 and return the best move.
- `quit`: Exit the engine.

---

## Evaluation Details

The evaluation function determines the quality of a position using the following terms:

- **Material Evaluation:** Total value of pieces for both sides, with a special score for the bishop pair.
- **Piece-Square Tables (PST):** Evaluates piece activity based on their location on the board, smoothly transitioning from middle-game to end-game weights.
- **Special Position Detection:** Assigns scores for Checkmate and Stalemate.
- **Rook Evaluation:** Bonuses for rooks on open files, semi-open files, and the 7th rank.
- **Pawn Evaluation:** Analyzes pawns for weaknesses (doubled, isolated, backward) and strengths (passed, connected, chains).

## Search Details

The engine uses several search algorithms and pruning techniques:

- **Iterative Deepening:** Searches incrementally deeper to improve move ordering.
- **Negamax Search:** The core recursive search algorithm with alpha-beta pruning.
- **Quiescence Search:** Extends the search at leaf nodes by examining tactical moves.
- **Transposition Table:** Memoizes previously evaluated positions.
- **Pruning and Reductions:**
  - Null Move Pruning
  - Late Move Reductions (LMR)
  - Late Move Pruning (LMP)
  - Futility Pruning
  - Internal Iterative Deepening (IID)
- **Move Ordering:**
  - Hash Move (from Transposition Table)
  - MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for captures
  - Killer Moves
  - History Heuristic


## Contributing

Contributions are welcome! Please open issues or pull requests to suggest improvements, new evaluation terms, or advanced search features.

---

## License

This project is open source and available under the GPL 3.0 License.

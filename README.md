# RustyKnight Chess Engine

RustyKnight is a Rust-based chess engine designed for speed and clarity, implementing solid evaluation heuristics and advanced search techniques. This README provides a comprehensive list of all evaluation terms and search methods used in the engine.

---

## Evaluation Terms

The evaluation function determines the quality of a position using the following terms and heuristics:

### 1. Material Evaluation
- Calculates the value of all pieces for both sides.
- Uses a material hash table to cache computed material values.
- Special score for bishop pairs (+30 for two bishops).

### 2. Piece-Square Tables (PST)
- Evaluates piece activity and placement using middle-game and end-game tables.
- The final piece square value is a blend based on the phase of the game.

### 3. Pawn Structure Evaluation
The engine evaluates many aspects of pawn structure, including:
- **Pawn Islands:** Groups of connected pawns separated by empty files.
- **Passed Pawns:** Pawns with no opposing pawns ahead on the same or adjacent files.
- **Candidate Pawns:** Passed pawns with potential to promote.
- **Hidden Passed Pawns:** Passed pawns blocked by own pieces.
- **Backward Pawns:** Pawns that cannot advance or be supported by other pawns.
- **Connected Pawns:** Pawns adjacent to each other, supporting one another.
- **Isolated Pawns:** Pawns with no friendly pawns on adjacent files.
- **Doubled Pawns:** Multiple pawns on the same file.
- **Faker Pawns:** Pawns that look passed but are easily attacked or blocked.
- **Weak Pawns:** Pawns that are easy to attack and not well-defended.
- **Pawn Majority:** More pawns on one side (queenside or kingside) compared to the opponent.
- **Center Pawns:** Pawns occupying the central files and ranks.
- **Pawn Race Potential:** Assessment of pawn promotion races in endgames.
- **Sentry Pawns:** Pawns controlling key squares (rare, receives special bonuses).
- **Chains:** Pawns supporting each other diagonally.
- **Minority Attacks:** Pawns attacking with fewer in number than their opponent on a given flank.
- **Hanging Pairs:** Pairs of pawns that are not supported and are on adjacent files.

### 4. Special Position Detection
- **Checkmate:** Assigns a large positive/negative score depending on the side to move.
- **Stalemate:** Assigns a contempt value (draw score).

### 5. Caching
- **Evaluation Cache:** Caches previously evaluated positions for efficiency.

### 6. Rook Evaluation
The engine assesses rook positioning and activity with the following heuristics:
- **Rooks on Open Files:** A rook on a file with no pawns is valued highly (+25 for each side).
- **Rooks on Semi-Open Files:** A rook on a file with only friendly pawns is valuable (+15 for each side).
- **Rooks on the Seventh Rank:** A rook on the opponent's second to last rank is a powerful piece, often controlling key squares and attacking pawns (+20 for white on the 7th rank, +20 for black on the 2nd rank).
---

## Search Methods

RustyKnight implements several search algorithms and pruning techniques:

### 1. Negamax Search
- Main recursive search algorithm using the negamax variant of minimax.
- Integrates alpha-beta pruning for efficiency.
- Supports Principal Variation (PV) search.

### 2. Quiescence Search
- Extends search at leaf nodes by examining only tactical moves (captures, promotions) to avoid the horizon effect.

### 3. Iterative Deepening
- Searches incrementally deeper, using results from shallower searches to improve move ordering.

### 4. Move Ordering
- **Hash Move:** Move recommended by the transposition table.
- **Killer Moves:** Quiet moves causing cutoffs in previous searches at the same depth.
- **History Heuristic:** Tracks which moves historically perform well.

### 5. Static Exchange Evaluation (SEE)
- Evaluates the net gain or loss from captures on a square, helping to order moves effectively.

### 6. Transposition Table
- Memoizes positions already evaluated to prevent redundant computation and speed up search.

### 7. Advanced Pruning and Extensions
- **Futility Pruning:** Skips moves unlikely to improve alpha.
- **Late Move Pruning (LMP):** Prunes quiet moves deeper into the move list.
- **Late Move Reductions (LMR):** Reduces search depth for less promising moves.
- **Null Move Pruning:** Searches positions where the side to move passes (makes a "null" move) to detect threats.
- **Internal Iterative Deepening (IID):** Used to find a good move when hash move is unavailable.

---

## Features

- **UCI Protocol Support:** Communicates with chess GUIs via the Universal Chess Interface.
- **Zobrist Hashing:** Efficient position hashing for transposition tables and caching.


---

## Code Structure

All main logic is implemented in [`src/main.rs`](src/main.rs).  
For details, refer to function definitions such as `evaluate`, `negamax`, `order_moves`, `quiesce`, `see`, `evaluate_pawn_structure`, and `iterative_deepening`.

---

## Contributing

Contributions are welcome! Please open issues or pull requests to suggest improvements, new evaluation terms, or advanced search features.

---

## License

This project is open source and available under the MIT License.

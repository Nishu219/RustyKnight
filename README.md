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

### 3. Special Position Detection
- **Checkmate:** Assigns a large positive/negative score depending on the side to move.
- **Stalemate:** Assigns a contempt value (draw score).

### 4. Caching
- **Evaluation Cache:** Caches previously evaluated positions for efficiency.

### 5. Rook Evaluation
The engine assesses rook positioning and activity with the following heuristics:
- **Rooks on Open Files:** A rook on a file with no pawns is valued highly (+25 for each side).
- **Rooks on Semi-Open Files:** A rook on a file with only friendly pawns is valuable (+15 for each side).
- **Rooks on the Seventh Rank:** A rook on the opponent's second to last rank is a powerful piece, often controlling key squares and attacking pawns (+20 for white on the 7th rank, +20 for black on the 2nd rank).

### 6. Pawn Evaluation
Analyzes pawn structure to detect weaknesses and strengths:
- **Doubled Pawns:** Penalized by -20 per extra pawn on the same file.
- **Isolated Pawns:** Penalized by -15 if there are no adjacent pawns on neighboring files.
- **Connected Pawns:** Bonus of +3 for pawns supporting each other diagonally.
- **Passed Pawns:** Bonuses increase with rank advancement:
   Ranks 2–6: +5 to +80 (based on proximity to promotion)
- **Pawn Chains:** Additional +5 bonus for pawns protected from behind by another pawn.
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
For details, refer to function definitions such as `evaluate`, `negamax`, `order_moves`, `quiesce`,  `evaluate_pawn_structure`, and `iterative_deepening`.

---

## Contributing

Contributions are welcome! Please open issues or pull requests to suggest improvements, new evaluation terms, or advanced search features.

---

## License

This project is open source and available under the GPL 3.0 License.

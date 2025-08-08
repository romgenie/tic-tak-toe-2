# Theory

## Minimax with strict tie-breaks

We solve states from the perspective of the side-to-move using exact minimax with memoization. The value v ∈ {+1, 0, −1} denotes win/draw/loss under perfect play. Among moves with equal v we break ties by distance-to-terminal (plies):

- Win ≻ Draw ≻ Loss
- Among wins/draws: prefer the shortest plies-to-end
- Among losses: prefer the longest plies-to-end (delay the loss)

This induces a total preorder over legal actions. Our solver returns per-action q_values ∈ {+1,0,−1} and dtt_action (distance-to-terminal after taking the action), and the set of optimal_moves obeying the policy above.

## Symmetry group (D4) and canonicalization

Tic-tac-toe on a 3×3 board admits the dihedral-4 (D4) symmetry group with 8 elements: {id, rot90, rot180, rot270, hflip, vflip, d1, d2}. We precompute index mappings for each transformation and use them to:

- Canonicalize states by selecting the lexicographically smallest image across all 8 transforms.
- Remap action indices under symmetry for augmentation and invariance checks.

For a board b and transform g ∈ D4, let g·b be the transformed board and g·a the transformed action index. The solver’s value is invariant: V(g·b) = V(b); optimal moves map equivariantly: g·ArgMax(b) = ArgMax(g·b).

## Complexity and state graph

From the empty board, 5478 states are reachable (4520 nonterminal; terminal split: X=626, O=316, draw=16). This is a tiny graph and the exact backward induction + memoization completes in milliseconds on commodity hardware. Our features and datasets are O(#states) and fit comfortably in memory.

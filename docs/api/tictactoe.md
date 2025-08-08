# API Reference: tictactoe

## Quick examples

Solve a position and inspect optimal moves:

```python
from tictactoe import solve_state

# board as string: 0 empty, 1 X, 2 O
board = "100020000"  # simple midgame
res = solve_state(tuple(int(c) for c in board))
print(res["value"], res["plies_to_end"], res["optimal_moves"])  # -> 1/0/-1, dtt, tuple of indices
```

Generate a small canonical-only dataset in memory:

```python
from tictactoe import solve_all_reachable, extract_board_features, generate_state_action_dataset

solved = solve_all_reachable()
states = [extract_board_features([int(c) for c in k], solved, normalize_to_move=True) for k in solved]
state_actions = generate_state_action_dataset(solved, canonical_only=True, epsilons=[0.0, 0.1])
print(len(states), len(state_actions))
```

Export datasets to disk with a manifest:

```python
from pathlib import Path
from tictactoe import ExportArgs, run_export

out = run_export(ExportArgs(out=Path("data_raw/example"), format="csv", canonical_only=True, epsilons=[0.1]))
print(out)  # path to manifest.json
```

---

::: tictactoe

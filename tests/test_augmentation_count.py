from tictactoe.orchestrator import generate_state_action_dataset
from tictactoe.solver import solve_all_reachable


def test_augmentation_emits_more_rows_per_state():
    solved = solve_all_reachable()
    # pick a small subset of states (first 10 keys) to keep test fast
    subset = {k: solved[k] for i, k in enumerate(solved.keys()) if i < 10}
    rows_no_aug = generate_state_action_dataset(subset, include_augmentation=False, canonical_only=False)
    rows_aug = generate_state_action_dataset(subset, include_augmentation=True, canonical_only=False)
    assert len(rows_aug) > len(rows_no_aug)

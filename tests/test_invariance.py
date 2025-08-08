import os
import sys
import importlib.util
import random

# Add src to path for direct import
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

spec = importlib.util.spec_from_file_location(
    "ttt_module",
    os.path.join(SRC, "01_generate_game_states_enhanced.py"),
)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore

transform_board = module.transform_board
apply_action_transform = module.apply_action_transform
ALL_SYMS = module.ALL_SYMS
solve_state = module.solve_state
solve_all_reachable = module.solve_all_reachable
deserialize_board = module.deserialize_board
build_policy_targets = module.build_policy_targets
epsilon_policy_distribution = module.epsilon_policy_distribution


def sample_reachable_states(solved_map, n=64, seed=7):
    keys = list(solved_map.keys())
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    return keys[: min(n, len(keys))]


def test_symmetry_invariance_value_q_dtt():
    solved_map = solve_all_reachable()
    sample = sample_reachable_states(solved_map, n=64)
    for key in sample:
        b = deserialize_board(key)
        s1 = solve_state(tuple(b))
        for sym in ALL_SYMS:
            tb = transform_board(b, sym)
            s2 = solve_state(tuple(tb))
            # Value should be invariant under board symmetries
            assert s2['value'] == s1['value']
            # Q and DTT should permute according to action index mapping
            for i in range(9):
                j = apply_action_transform(i, sym)
                assert s2['q_values'][j] == s1['q_values'][i]
                assert s2['dtt_action'][j] == s1['dtt_action'][i]


def test_policy_distributions_are_normalized_over_legal():
    solved_map = solve_all_reachable()
    sample = sample_reachable_states(solved_map, n=32, seed=11)
    for key in sample:
        b = deserialize_board(key)
        sol = solved_map[key]
        # Skip terminal (no legal moves)
        if all(v != 0 for v in b) or module.get_winner(b) != 0:
            continue
        pol = build_policy_targets(b, sol, lambda_temp=0.5, q_temp=1.0)
        for name in ['policy_optimal_uniform', 'policy_soft_dtt', 'policy_soft_q']:
            arr = pol[name]
            # Illegal moves should have zero mass
            for i, v in enumerate(b):
                if v != 0:
                    assert arr[i] == 0.0
            s = sum(arr)
            # Should sum to 1 within tolerance when there are legal moves
            assert abs(s - 1.0) < 1e-6
        # Epsilon policy also normalized
        for eps in [0.0, 0.1, 0.3]:
            earr = epsilon_policy_distribution(b, sol, eps)
            for i, v in enumerate(b):
                if v != 0:
                    assert earr[i] == 0.0
            assert abs(sum(earr) - 1.0) < 1e-6

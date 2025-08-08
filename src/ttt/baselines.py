"""
Baselines and expectations under random and epsilon-optimal policies.
"""
from functools import lru_cache
from typing import Tuple
from .solver import legal_moves, current_player_t, apply_move_t, winner_t, is_draw_t, solve_state


@lru_cache(maxsize=None)
def random_policy_expectation(board_t: tuple) -> Tuple[float, float]:
    w = winner_t(board_t)
    if w != 0:
        return (1.0 if w == 1 else -1.0, 0.0)
    if is_draw_t(board_t):
        return (0.0, 0.0)
    moves = legal_moves(board_t)
    if not moves:
        return (0.0, 0.0)
    p = current_player_t(board_t)
    vs = []
    ds = []
    for mv in moves:
        child = apply_move_t(board_t, mv, p)
        v_child, d_child = random_policy_expectation(child)
        vs.append(v_child)
        ds.append(d_child)
    return (float(sum(vs)) / len(vs), 1.0 + float(sum(ds)) / len(ds))


@lru_cache(maxsize=None)
def epsilon_optimal_expectation(board_t: tuple, epsilon: float = 0.1) -> Tuple[float, float]:
    w = winner_t(board_t)
    if w != 0:
        return (1.0 if w == 1 else -1.0, 0.0)
    if is_draw_t(board_t):
        return (0.0, 0.0)
    moves = legal_moves(board_t)
    if not moves:
        return (0.0, 0.0)
    p = current_player_t(board_t)
    s = solve_state(board_t)
    optimal = list(s['optimal_moves'])
    nL = len(moves)
    nO = max(1, len(optimal))
    vs = 0.0
    ds = 0.0
    for mv in moves:
        wmv = ((1.0 - epsilon) * (1.0 / nO if mv in optimal else 0.0)) + (epsilon * (1.0 / nL))
        child = apply_move_t(board_t, mv, p)
        v_child, d_child = epsilon_optimal_expectation(child, epsilon)
        vs += wmv * v_child
        ds += wmv * d_child
    return (vs, 1.0 + ds)

"""
Exact game-theoretic solver (minimax with memoization), from the side-to-move perspective.
Tie-break policy:
- Prefer win over draw over loss.
- Among wins/draws, prefer shorter distance (plies) to termination.
- Among losses, prefer longer distance (delay the loss).
"""
from functools import lru_cache
from typing import Dict, List, Optional
from collections import deque
from .game_basics import get_winner, is_draw, serialize_board


def legal_moves(board_t: tuple) -> List[int]:
    return [i for i, v in enumerate(board_t) if v == 0]


def current_player_t(board_t: tuple) -> int:
    x = board_t.count(1)
    o = board_t.count(2)
    return 1 if x == o else 2


def apply_move_t(board_t: tuple, idx: int, player: int) -> tuple:
    lst = list(board_t)
    lst[idx] = player
    return tuple(lst)


def winner_t(board_t: tuple) -> int:
    return get_winner(list(board_t))


def is_draw_t(board_t: tuple) -> bool:
    return is_draw(list(board_t))


def better_of(a: int, b: int) -> int:
    order = {+1: 2, 0: 1, -1: 0}
    return a if order[a] > order[b] else b


@lru_cache(maxsize=None)
def solve_state(board_t: tuple) -> Dict:
    w = winner_t(board_t)
    if w != 0:
        return {
            'value': -1,
            'plies_to_end': 0,
            'optimal_moves': tuple(),
            'q_values': tuple([None] * 9),
            'dtt_action': tuple([None] * 9),
        }
    if is_draw_t(board_t):
        return {
            'value': 0,
            'plies_to_end': 0,
            'optimal_moves': tuple(),
            'q_values': tuple([None] * 9),
            'dtt_action': tuple([None] * 9),
        }
    p = current_player_t(board_t)
    moves = legal_moves(board_t)
    q_vals: List[Optional[int]] = [None] * 9
    dtt_action: List[Optional[int]] = [None] * 9
    best_val: Optional[int] = None
    best_dtt: Optional[int] = None
    best_moves: List[int] = []
    for mv in moves:
        child = apply_move_t(board_t, mv, p)
        s_child = solve_state(child)
        q = -s_child['value']
        q_vals[mv] = q
        dtt_action[mv] = 1 + s_child['plies_to_end']
        if best_val is None:
            best_val = q
            best_dtt = dtt_action[mv]
            best_moves = [mv]
            continue
        if better_of(q, best_val) == q and q != best_val:
            best_val = q
            best_dtt = dtt_action[mv]
            best_moves = [mv]
        elif q == best_val:
            if q == +1:
                if dtt_action[mv] < best_dtt:
                    best_dtt = dtt_action[mv]
                    best_moves = [mv]
                elif dtt_action[mv] == best_dtt:
                    best_moves.append(mv)
            elif q == -1:
                if dtt_action[mv] > best_dtt:
                    best_dtt = dtt_action[mv]
                    best_moves = [mv]
                elif dtt_action[mv] == best_dtt:
                    best_moves.append(mv)
            else:
                if dtt_action[mv] < best_dtt:
                    best_dtt = dtt_action[mv]
                    best_moves = [mv]
                elif dtt_action[mv] == best_dtt:
                    best_moves.append(mv)
    return {
        'value': best_val,
        'plies_to_end': best_dtt,
        'optimal_moves': tuple(sorted(best_moves)),
        'q_values': tuple(q_vals),
        'dtt_action': tuple(dtt_action),
    }


def solve_all_reachable() -> dict:
    """Enumerate and solve all states reachable from the empty board."""
    all_nodes = {}
    q = deque()
    start = tuple([0] * 9)
    q.append(start)
    seen = {start}
    while q:
        s = q.popleft()
        all_nodes[s] = True
        if winner_t(s) != 0 or is_draw_t(s):
            continue
        p = current_player_t(s)
        for mv in legal_moves(s):
            child = apply_move_t(s, mv, p)
            if child not in seen:
                seen.add(child)
                q.append(child)
    solved = {}
    for s in all_nodes:
        res = solve_state(s)
        solved[''.join(map(str, s))] = res
    return solved

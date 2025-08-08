"""
Supervision utilities: pairwise preferences and symmetry-consistency pairs.
"""
from typing import Dict, List
from .symmetry import ALL_SYMS, transform_board


def generate_pairwise_preferences(solved_map: Dict[str, dict]) -> List[Dict]:
    rows: List[Dict] = []
    for key, sol in solved_map.items():
        q = list(sol['q_values'])
        d = list(sol['dtt_action'])
        legal = [i for i, v in enumerate(q) if v is not None]
        if len(legal) < 2:
            continue
        def pref_key(i: int):
            tier_rank = {+1: 2, 0: 1, -1: 0}[q[i]]
            dtt_rank = -d[i] if q[i] == -1 else d[i]
            return (-tier_rank, dtt_rank)
        sorted_legal = sorted(legal, key=pref_key)
        for i in range(len(sorted_legal)):
            for j in range(i + 1, len(sorted_legal)):
                a, b = sorted_legal[i], sorted_legal[j]
                rows.append({
                    'state': key,
                    'action_a': a,
                    'action_b': b,
                    'label': 1,
                    'q_diff': q[a] - q[b],
                    'dtt_diff': (d[a] - d[b]) if (d[a] is not None and d[b] is not None) else None,
                })
    return rows


def generate_symmetry_pairs(solved_map: Dict[str, dict]) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for key in solved_map.keys():
        board = [int(c) for c in key]
        for sym_op in ALL_SYMS[1:]:
            tboard = transform_board(board, sym_op)
            pairs.append({
                'state': key,
                'sym_op': sym_op,
                'transformed_state': ''.join(map(str, tboard)),
            })
    return pairs

"""
Policy targets, rankings, and difficulty signals for learning.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np


def build_policy_targets(board: List[int], sol: Dict, lambda_temp: float = 0.5, q_temp: float = 1.0) -> Dict[str, List[float]]:
    legal = [int(v == 0) for v in board]
    optimal = list(sol['optimal_moves'])
    pol_uniform = [0.0] * 9
    if optimal:
        mass = 1.0 / len(optimal)
        for m in optimal:
            pol_uniform[m] = mass
    dtt = sol['dtt_action']
    weights: List[float] = []
    for i in range(9):
        if not legal[i] or dtt[i] is None:
            weights.append(0.0)
        else:
            q = sol['q_values'][i]
            base = 1.0 if q == 0 else (2.0 if q == +1 else 0.5)
            weights.append(base * np.exp(-lambda_temp * dtt[i]))
    s = sum(weights)
    pol_soft = [w / s if s > 0 else 0.0 for w in weights]
    q_vals = sol['q_values']
    sm_weights: List[float] = []
    for i in range(9):
        if not legal[i] or q_vals[i] is None:
            sm_weights.append(0.0)
        else:
            sm_weights.append(np.exp(q_temp * float(q_vals[i])))
    z = sum(sm_weights)
    pol_soft_q = [w / z if z > 0 else 0.0 for w in sm_weights]
    return {
        'policy_optimal_uniform': pol_uniform,
        'policy_soft_dtt': pol_soft,
        'policy_soft_q': pol_soft_q,
    }


def epsilon_policy_distribution(board: List[int], sol: Dict, epsilon: float) -> List[float]:
    legal = [i for i, v in enumerate(board) if v == 0]
    optimal = list(sol['optimal_moves'])
    nL = len(legal) if legal else 1
    nO = len(optimal) if optimal else 1
    pol = [0.0] * 9
    for i in legal:
        a = ((1.0 - epsilon) * (1.0 / nO if i in optimal else 0.0)) + (epsilon * (1.0 / nL))
        pol[i] = a
    return pol


def compute_action_ranks(sol: Dict) -> Tuple[List[Optional[int]], List[Optional[int]], List[Optional[int]]]:
    order_map = {+1: 2, 0: 1, -1: 0}
    q_values = list(sol['q_values'])
    dtts = list(sol['dtt_action'])
    legal_idxs = [i for i, q in enumerate(q_values) if q is not None]

    def key_for(i: int):
        q = q_values[i]
        if q is None:
            return (float('inf'), float('inf'), i)
        primary = -order_map[q]
        d = dtts[i]
        d_val = 10 ** 9 if d is None else d
        secondary = -d_val if q == -1 else d_val
        return (primary, secondary, i)

    sorted_moves = sorted(legal_idxs, key=key_for)
    ranks = [None] * 9
    value_regret = [None] * 9
    dtt_regret = [None] * 9
    if not sorted_moves:
        return ranks, value_regret, dtt_regret
    best_q = q_values[sorted_moves[0]]
    best_dtts_same_q = [dtts[i] for i in legal_idxs if q_values[i] == best_q and dtts[i] is not None]
    if best_dtts_same_q:
        if best_q == -1:
            best_dtt_pref = max(best_dtts_same_q)
        else:
            best_dtt_pref = min(best_dtts_same_q)
    else:
        best_dtt_pref = None
    prev_key = None
    current_rank = 0
    for i in sorted_moves:
        k = key_for(i)[:2]
        if k != prev_key:
            current_rank += 1
            prev_key = k
        ranks[i] = current_rank
        value_regret[i] = order_map[best_q] - order_map[q_values[i]]
        if q_values[i] == best_q and dtts[i] is not None and best_dtt_pref is not None:
            if best_q == -1:
                dtt_regret[i] = max(0, best_dtt_pref - dtts[i])
            else:
                dtt_regret[i] = max(0, dtts[i] - best_dtt_pref)
    return ranks, value_regret, dtt_regret


def difficulty_score(sol: Dict) -> float:
    q = list(sol['q_values'])
    d = list(sol['dtt_action'])
    legal = [i for i, v in enumerate(q) if v is not None]
    if not legal:
        return 0.0
    best = max(q[i] for i in legal)
    worst = min(q[i] for i in legal)
    tier_gap = best - worst
    close = sum(1 for i in legal if q[i] == best)
    dtt_values = [d[i] for i in legal if d[i] is not None]
    dtt_mean = float(np.mean(dtt_values)) if dtt_values else 0.0
    return float((1.0 if tier_gap == 0 else 0.5) * (1 + np.log1p(close)) * (1 + 0.1 * dtt_mean))

"""
Feature and dataset orchestration that composes tictactoe.* modules.

Perspective note: features are computed on the board as given (no X-to-move
normalization). The "current_player" field is included to disambiguate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# Note: tactics utilities imported selectively elsewhere; none needed here
from .features import (
    calculate_cell_line_potentials,
    calculate_connectivity,
    calculate_control_metrics,
    calculate_game_phase,
    calculate_line_threats,
    calculate_pattern_strength,
    count_two_in_row_open,
)
from .game_basics import (
    deserialize_board,
    get_piece_counts,
    get_winner,
    is_draw,
    is_valid_state,
    serialize_board,
)
from .policy import (
    build_policy_targets,
    difficulty_score,
    epsilon_policy_distribution,
)
from .symmetry import ALL_SYMS, apply_action_transform, symmetry_info, transform_board


def extract_board_features(
    board: List[int],
    solved_map: Dict[str, dict],
    lambda_temp: float = 0.5,
    q_temp: float = 1.0,
    epsilons: Optional[List[float]] = None,
    normalize_to_move: bool = False,
) -> Dict[str, Any]:
    """Extract features for a board.

    By default, features are computed on the board as given.
    If normalize_to_move=True, we remap pieces so the side-to-move becomes X=1.
    Note: Policies and q-values are derived from the original board's solver
    output. Normalization only swaps labels 1<->2; legality and q-values align
    by construction because moves are on indices, not on piece IDs.
    """
    if epsilons is None:
        epsilons = [0.1]
    to_move = 1 if board.count(1) == board.count(2) else 2
    if normalize_to_move and to_move == 2:
        # swap X and O labels to make current player X
        normalized_board = [0 if v == 0 else (1 if v == 2 else 2) for v in board]
        current_player = 1
    else:
        normalized_board = board[:]
        current_player = to_move
    x_count, o_count = get_piece_counts(normalized_board)
    winner_raw = get_winner(board)
    winner_norm = get_winner(normalized_board)
    key = serialize_board(board)
    norm_key = serialize_board(normalized_board)
    reachable = key in solved_map

    if reachable:
        sol = solved_map[key]
        value_current = sol['value']
        plies_to_end = sol['plies_to_end']
        optimal_moves = set(sol['optimal_moves'])
        qvals = list(sol['q_values'])
        dtt_a = list(sol['dtt_action'])
        policy_targets = build_policy_targets(normalized_board, sol, lambda_temp=lambda_temp, q_temp=q_temp)
        pol_uniform = policy_targets['policy_optimal_uniform']
        pol_soft = policy_targets['policy_soft_dtt']
        pol_soft_q = policy_targets['policy_soft_q']
        eps_policies = {}
        for eps in epsilons:
            tag = f"{int(round(eps*100)):03d}"
            eps_policies[tag] = epsilon_policy_distribution(normalized_board, sol, eps)
        pol_entropy = -sum(p * np.log(p + 1e-10) for p in pol_uniform if p > 0)
        pol_soft_dtt_entropy = -sum(p * np.log(p + 1e-10) for p in pol_soft if p > 0)
        child_tiers = {
            'child_wins': sum(1 for v in qvals if v == +1),
            'child_draws': sum(1 for v in qvals if v == 0),
            'child_losses': sum(1 for v in qvals if v == -1),
        }
        difficulty = difficulty_score(sol)
        # optional: compute reply branching factor as the average number of
        # legal replies for the opponent after optimal moves. Keep lightweight
        # and deterministic. If no optimal moves, 0.0. Terminal children
        # contribute 0 by definition (no replies).
        legal_reply_counts: List[int] = []
        for mv in optimal_moves:
            child = board[:]
            child[mv] = (1 if board.count(1) == board.count(2) else 2)
            if get_winner(child) != 0 or is_draw(child):
                legal_reply_counts.append(0)
            else:
                legal_reply_counts.append(sum(1 for v in child if v == 0))
        reply_branching = float(sum(legal_reply_counts) / len(legal_reply_counts)) if legal_reply_counts else 0.0
    else:
        value_current = None
        plies_to_end = None
        optimal_moves = set()
        qvals = [None]*9
        dtt_a = [None]*9
        pol_uniform = [None]*9
        pol_soft = [None]*9
        pol_soft_q = [None]*9
        eps_policies = {}
        pol_entropy = None
        pol_soft_dtt_entropy = None
        child_tiers = {'child_wins': 0, 'child_draws': 0, 'child_losses': 0}
        difficulty = 0.0
        reply_branching = 0.0

    sym = symmetry_info(board)
    legal = [normalized_board[i] == 0 for i in range(9)]
    best_mask = [(i in optimal_moves) for i in range(9)] if reachable else [False]*9
    cell_pot = calculate_cell_line_potentials(normalized_board)

    # Extended positional/strategic features (deterministic and cheap)
    # Use the normalized_board for player-relative metrics
    x_threats = calculate_line_threats(normalized_board, 1)
    o_threats = calculate_line_threats(normalized_board, 2)
    x_conn = calculate_connectivity(normalized_board, 1)
    o_conn = calculate_connectivity(normalized_board, 2)
    control = calculate_control_metrics(normalized_board)
    x_patterns = calculate_pattern_strength(normalized_board, 1)
    o_patterns = calculate_pattern_strength(normalized_board, 2)
    game_phase = calculate_game_phase(normalized_board)
    x_two_open = count_two_in_row_open(normalized_board, 1)
    o_two_open = count_two_in_row_open(normalized_board, 2)

    features: Dict[str, Any] = {
        'board_state': key,
        'normalized_board_state': norm_key,
    'swapped_color': int(normalize_to_move and to_move == 2),
        'x_count': x_count,
        'o_count': o_count,
        'empty_count': normalized_board.count(0),
        'move_number': x_count + o_count,
        'current_player': current_player,
        'is_terminal': winner_raw != 0 or is_draw(board),
        'winner': winner_raw,
        'winner_normalized': winner_norm,
        'is_draw': is_draw(board),
        'is_valid': is_valid_state(board),
        'reachable_from_start': reachable,
        'canonical_form': sym['canonical_form'],
        'canonical_op': sym['canonical_op'],
        'orbit_size': sym['orbit_size'],
        'horizontal_symmetric': sym['horizontal_symmetric'],
        'vertical_symmetric': sym['vertical_symmetric'],
        'diagonal_symmetric': sym['diagonal_symmetric'],
        'rotational_symmetric': sym['rotational_symmetric'],
        'any_symmetric': sym['any_symmetric'],
        'orbit_index': sym['orbit_index'],
        'value_current': value_current,
        'plies_to_end': plies_to_end,
        'optimal_moves_count': len(optimal_moves),
        'optimal_policy_entropy': pol_entropy,
        'policy_soft_dtt_entropy': pol_soft_dtt_entropy,
        'policy_soft_q_entropy': (
            -sum(p * np.log(p + 1e-10) for p in pol_soft_q if p > 0) if reachable else None
        ),
    # Scalar difficulty per state; 0.0 when not reachable
    'difficulty_score': difficulty,
        'reply_branching_factor': reply_branching,
        **child_tiers,
    # control metrics (symmetric)
    **control,
    # player-relative threats/connectivity/patterns
    'x_row_threats': x_threats['row_threats'],
    'x_col_threats': x_threats['col_threats'],
    'x_diag_threats': x_threats['diag_threats'],
    'x_total_threats': x_threats['total_threats'],
    'o_row_threats': o_threats['row_threats'],
    'o_col_threats': o_threats['col_threats'],
    'o_diag_threats': o_threats['diag_threats'],
    'o_total_threats': o_threats['total_threats'],
    'x_connected_pairs': x_conn['connected_pairs'],
    'x_total_connections': x_conn['total_connections'],
    'x_isolated_pieces': x_conn['isolated_pieces'],
    'x_cluster_count': x_conn['cluster_count'],
    'x_largest_cluster': x_conn['largest_cluster'],
    'o_connected_pairs': o_conn['connected_pairs'],
    'o_total_connections': o_conn['total_connections'],
    'o_isolated_pieces': o_conn['isolated_pieces'],
    'o_cluster_count': o_conn['cluster_count'],
    'o_largest_cluster': o_conn['largest_cluster'],
    'x_open_lines': x_patterns['open_lines'],
    'x_semi_open_lines': x_patterns['semi_open_lines'],
    'x_blocked_lines': x_patterns['blocked_lines'],
    'x_potential_lines': x_patterns['potential_lines'],
    'o_open_lines': o_patterns['open_lines'],
    'o_semi_open_lines': o_patterns['semi_open_lines'],
    'o_blocked_lines': o_patterns['blocked_lines'],
    'o_potential_lines': o_patterns['potential_lines'],
    'x_two_in_row_open': x_two_open,
    'o_two_in_row_open': o_two_open,
    'game_phase': game_phase,
    }

    for i in range(9):
        features[f'cell_{i}'] = normalized_board[i]
        features[f'legal_{i}'] = int(legal[i])
        features[f'best_{i}'] = int(best_mask[i])
        features[f'q_value_{i}'] = qvals[i]
        features[f'dtt_action_{i}'] = dtt_a[i]
        features[f'canonical_action_map_{i}'] = apply_action_transform(i, sym['canonical_op'])
        features[f'policy_uniform_{i}'] = pol_uniform[i] if reachable else None
        features[f'policy_soft_{i}'] = pol_soft[i] if reachable else None
        features[f'policy_soft_q_{i}'] = pol_soft_q[i] if reachable else None
        for tag, pol in (eps_policies.items() if reachable else []):
            features[f'epsilon_policy_{tag}_{i}'] = pol[i]
        features[f'x_cell_open_lines_{i}'] = cell_pot['x_cell_open_lines'][i]
        features[f'o_cell_open_lines_{i}'] = cell_pot['o_cell_open_lines'][i]

    return features


def generate_state_action_dataset(
    solved_map: Dict[str, dict],
    include_augmentation: bool = False,
    canonical_only: bool = False,
    lambda_temp: float = 0.5,
    q_temp: float = 1.0,
    epsilons: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    if epsilons is None:
        epsilons = [0.1]
    rows: List[Dict[str, Any]] = []
    def emit_rows(state_board: List[int], state_key: str, sol_ref: Dict, pol_ref: Dict[str, List[float]], eps_ref: Optional[Dict[str, List[float]]] = None):
        p_local = 1 if state_board.count(1) == state_board.count(2) else 2
        sym_local = symmetry_info(state_board)
        canonical_local = sym_local['canonical_form']
        for i in range(9):
            if state_board[i] != 0:
                continue
            child = state_board[:]
            child[i] = p_local
            w = get_winner(child)
            done = (w != 0) or is_draw(child)
            # +1 win, -1 loss, 0 draw or non-terminal
            r = 1 if w == p_local else (-1 if w != 0 else 0)
            base_row = {
                'board_state': state_key,
                'canonical_form': canonical_local,
                'player_to_move': p_local,
                'action': i,
                'q_value': sol_ref['q_values'][i],
                'dtt_action': sol_ref['dtt_action'][i],
                'optimal_action': int(i in sol_ref['optimal_moves']),
                'policy_optimal_uniform': pol_ref['policy_optimal_uniform'][i],
                'policy_soft_dtt': pol_ref['policy_soft_dtt'][i],
                'policy_soft_q': pol_ref['policy_soft_q'][i],
                'reward_if_terminal': r,
                'done': int(done),
            }
            if eps_ref:
                # One column per epsilon, value for this action
                base_row.update({f'epsilon_policy_{tag}': eps_ref[tag][i] for tag in eps_ref})
            rows.append(base_row)

    for key, sol in solved_map.items():
        b = deserialize_board(key)
        sym = symmetry_info(b)
        canonical = sym['canonical_form']
        if canonical_only and key != canonical:
            continue
        # Skip terminal states entirely for state-action dataset
        if get_winner(b) != 0 or is_draw(b):
            continue
        pol = build_policy_targets(b, sol, lambda_temp=lambda_temp, q_temp=q_temp)
        # Precompute epsilon-greedy policies for the base state. Each is a 9-length
        # distribution over actions. We'll emit per-row columns: epsilon_policy_XXX
        # equals the probability of the row's action under that epsilon.
        eps_pols = {f"{int(round(e*100)):03d}": epsilon_policy_distribution(b, sol, e) for e in epsilons}

        # base rows
        emit_rows(b, key, sol, pol, eps_pols)

        # symmetry augmentation (minimal: transform board + remap action indices)
        # If canonical_only is True, skip augmentation to honor the constraint.
        if include_augmentation and not canonical_only:
            for op in ALL_SYMS[1:]:
                tb = transform_board(b, op)
                # Skip transformed terminal states as well
                if get_winner(tb) != 0 or is_draw(tb):
                    continue
                tkey = serialize_board(tb)
                # Remap policies and move stats by op
                pol_t = {
                    'policy_optimal_uniform': [0.0]*9,
                    'policy_soft_dtt': [0.0]*9,
                    'policy_soft_q': [0.0]*9,
                }
                q_t = [None]*9
                dtt_t = [None]*9
                optimal_t = set()
                # Prepare transformed epsilon policies with the same remap
                eps_pols_t: Dict[str, List[float]] = {tag: [0.0]*9 for tag in eps_pols}
                for j in range(9):
                    j_t = apply_action_transform(j, op)
                    pol_t['policy_optimal_uniform'][j_t] = pol['policy_optimal_uniform'][j]
                    pol_t['policy_soft_dtt'][j_t] = pol['policy_soft_dtt'][j]
                    pol_t['policy_soft_q'][j_t] = pol['policy_soft_q'][j]
                    q_t[j_t] = sol['q_values'][j]
                    dtt_t[j_t] = sol['dtt_action'][j]
                    for tag in eps_pols:
                        eps_pols_t[tag][j_t] = eps_pols[tag][j]
                for m in sol['optimal_moves']:
                    optimal_t.add(apply_action_transform(m, op))

                sol_t = {
                    'q_values': q_t,
                    'dtt_action': dtt_t,
                    'optimal_moves': tuple(sorted(optimal_t)),
                }
                emit_rows(tb, tkey, sol_t, pol_t, eps_pols_t)
    return rows

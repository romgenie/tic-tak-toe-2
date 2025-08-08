"""
Feature and dataset orchestration that composes ttt.* modules.
This keeps the monolith optional and enables a clean teaching pipeline.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .game_basics import (
    serialize_board, deserialize_board, get_winner, is_draw,
    get_piece_counts, is_valid_state,
)
from .symmetry import symmetry_info, apply_action_transform, transform_board, ALL_SYMS
from .solver import solve_state
from .tactics import (
    immediate_winning_moves, fork_moves, gives_opponent_immediate_win,
)
from .features_positional import (
    calculate_line_threats, calculate_connectivity, calculate_control_metrics,
    calculate_pattern_strength, calculate_cell_line_potentials, calculate_game_phase,
    count_two_in_row_open,
)
from .policy_targets import (
    build_policy_targets, epsilon_policy_distribution, compute_action_ranks, difficulty_score,
)
from .baselines import random_policy_expectation, epsilon_optimal_expectation


# ---------- Helper computations (moved from monolith) ----------

def creates_immediate_threat_oneply(b: List[int], player: int, m: int) -> bool:
    if b[m] != 0:
        return False
    b2 = b[:]
    b2[m] = player
    return len(immediate_winning_moves(b2, player)) > 0

def compute_margin_features(sol: Dict) -> Dict[str, Optional[float]]:
    q_values = list(sol['q_values'])
    dtts = list(sol['dtt_action'])
    order_map = {+1: 2, 0: 1, -1: 0}
    legal = [i for i, q in enumerate(q_values) if q is not None]
    if not legal:
        return {
            'best_q_value': None,
            'best_dtt': None,
            'second_q_value': None,
            'second_dtt': None,
            'value_gap_levels': None,
            'dtt_gap': None,
            'dominated_count': 0,
            'legal_count': 0,
        }

    def key_for(i: int):
        q = q_values[i]
        d = dtts[i]
        d_val = 10**9 if d is None else d
        primary = -order_map[q]
        secondary = (-d_val) if q == -1 else d_val
        return (primary, secondary, i)

    sorted_legal = sorted(legal, key=key_for)
    best = sorted_legal[0]
    best_q = q_values[best]
    best_d = dtts[best]
    second = sorted_legal[1] if len(sorted_legal) > 1 else None
    second_q = q_values[second] if second is not None else None
    second_d = dtts[second] if second is not None else None

    gap_levels = None if second_q is None else ({+1: 2, 0: 1, -1: 0}[best_q] - {+1: 2, 0: 1, -1: 0}[second_q])
    if second_q is None:
        dtt_gap = None
    elif best_q == second_q:
        dtt_gap = ((best_d or 0) - (second_d or 0)) if best_q == -1 else ((second_d or 0) - (best_d or 0))
    else:
        dtt_gap = None

    dominated = sum(1 for i in legal if {+1: 2, 0: 1, -1: 0}[q_values[i]] < {+1: 2, 0: 1, -1: 0}[best_q])

    return {
        'best_q_value': float(best_q) if best_q is not None else None,
        'best_dtt': float(best_d) if best_d is not None else None,
        'second_q_value': float(second_q) if second_q is not None else None,
        'second_dtt': float(second_d) if second_d is not None else None,
        'value_gap_levels': float(gap_levels) if gap_levels is not None else None,
        'dtt_gap': float(dtt_gap) if dtt_gap is not None else None,
        'dominated_count': int(dominated),
        'legal_count': int(len(legal)),
    }


def count_trap_in_2(board: List[int], player: int, move: int) -> bool:
    if board[move] != 0:
        return False
    new_board = board[:]
    new_board[move] = player
    if get_winner(new_board) == player:
        return False
    opp = 2 if player == 1 else 1
    threats = immediate_winning_moves(new_board, player)
    if len(threats) >= 2:
        return True
    if len(threats) == 1:
        block_board = new_board[:]
        block_board[threats[0]] = opp
        for next_move in range(9):
            if block_board[next_move] == 0:
                test_board = block_board[:]
                test_board[next_move] = player
                if get_winner(test_board) == player:
                    return True
                if len(immediate_winning_moves(test_board, player)) >= 2:
                    return True
    return False


def action_tactical_labels(board: List[int], move: int, sol: Optional[Dict] = None) -> Dict[str, int]:
    p = 1 if board.count(1) == board.count(2) else 2
    opp = 2 if p == 1 else 1
    labels = {
        'legal': int(board[move] == 0),
        'is_center': int(move == 4),
        'is_corner': int(move in [0, 2, 6, 8]),
        'is_edge': int(move in [1, 3, 5, 7]),
        'is_immediate_win': 0,
        'is_immediate_block': 0,
        'creates_fork': 0,
        'blocks_opponent_fork': 0,
        'gives_opponent_immediate_win': 0,
        'creates_immediate_threat': 0,
        'creates_trap_in_2': 0,
        'is_forced_move': 0,
        'is_safe_move': 0,
    }
    if labels['legal'] == 0:
        return labels
    labels['is_immediate_win'] = int(move in immediate_winning_moves(board, p))
    labels['is_immediate_block'] = int(move in immediate_winning_moves(board, opp))
    labels['creates_fork'] = int(move in fork_moves(board, p))
    # blocks_opponent_fork heuristic: if opponent had a fork, check it eliminates
    nb = board[:]
    nb[move] = p
    labels['blocks_opponent_fork'] = int(len(fork_moves(board, opp)) > 0 and len(fork_moves(nb, opp)) == 0)
    labels['gives_opponent_immediate_win'] = int(gives_opponent_immediate_win(board, p, move))
    if labels['legal'] and labels['is_immediate_win'] == 0:
        labels['creates_immediate_threat'] = int(creates_immediate_threat_oneply(board, p, move))
    if sol is not None and sol['q_values'][move] is not None:
        labels['creates_trap_in_2'] = int(sol['q_values'][move] == +1 and sol['dtt_action'][move] == 2)
    else:
        labels['creates_trap_in_2'] = int(count_trap_in_2(board, p, move))
    opp_wins_now = immediate_winning_moves(board, opp)
    if opp_wins_now:
        legal_moves_now = [i for i, v in enumerate(board) if v == 0]
        safe_blockers = []
        for m in legal_moves_now:
            nb2 = board[:]
            nb2[m] = p
            if len(immediate_winning_moves(nb2, opp)) == 0:
                safe_blockers.append(m)
        labels['is_forced_move'] = int(len(safe_blockers) == 1 and move == safe_blockers[0])
    labels['is_safe_move'] = int(labels['legal'] and (not labels['gives_opponent_immediate_win']))
    return labels


def action_robustness_metrics(board: List[int], move: int, solved_map: Dict[str, dict]) -> Dict[str, Optional[int]]:
    if board[move] != 0:
        return {
            'winning_reply_count': None,
            'drawing_reply_count': None,
            'losing_reply_count': None,
            'worst_reply_q': None,
            'safe_1ply': None,
            'safe_2ply': None,
        }
    b2 = board[:]
    p = 1 if board.count(1) == board.count(2) else 2
    b2[move] = p
    if get_winner(b2) != 0 or is_draw(b2):
        q_terminal = +1 if get_winner(b2) == p else (0 if is_draw(b2) else -1)
        return {
            'winning_reply_count': 0,
            'drawing_reply_count': 0,
            'losing_reply_count': 0,
            'worst_reply_q': q_terminal,
            'safe_1ply': 1,
            'safe_2ply': 1,
        }
    opp = 2 if p == 1 else 1
    replies = [i for i, v in enumerate(b2) if v == 0]
    win_cnt = draw_cnt = loss_cnt = 0
    worst_q = +1
    for r in replies:
        b3 = b2[:]
        b3[r] = opp
        k3 = serialize_board(b3)
        if k3 not in solved_map:
            continue
        q_after = solved_map[k3]['value']
        if q_after == +1:
            win_cnt += 1
        elif q_after == 0:
            draw_cnt += 1
        else:
            loss_cnt += 1
        worst_q = min(worst_q, q_after)
    safe_1ply = int(loss_cnt == 0)
    # 2-ply safety check (conservative)
    safe_2ply = 1
    for r in replies:
        b3 = b2[:]
        b3[r] = opp
        k3 = serialize_board(b3)
        if k3 not in solved_map:
            continue
        val3 = solved_map[k3]['value']
        dtt3 = solved_map[k3]['plies_to_end']
        if val3 == -1 and dtt3 is not None and dtt3 <= 2:
            safe_2ply = 0
            break
    return {
        'winning_reply_count': win_cnt,
        'drawing_reply_count': draw_cnt,
        'losing_reply_count': loss_cnt,
        'worst_reply_q': worst_q,
        'safe_1ply': safe_1ply,
        'safe_2ply': safe_2ply,
    }


def orbit_int_id(canonical_form: str, canonical_action: int) -> int:
    val = 0
    for c in canonical_form:
        val = val * 3 + ord(c) - ord('0')
    return (val << 4) | (canonical_action & 0xF)


def child_qtier_histogram(sol: Dict) -> Dict[str, int]:
    q = [v for v in sol['q_values'] if v is not None]
    return {
        'child_wins': sum(1 for v in q if v == +1),
        'child_draws': sum(1 for v in q if v == 0),
        'child_losses': sum(1 for v in q if v == -1),
    }


def reply_branching_factor_after_best(board: List[int], sol: Dict) -> float:
    optimal = sol['optimal_moves']
    if not optimal:
        return 0.0
    p = 1 if board.count(1) == board.count(2) else 2
    total_branches = 0
    count = 0
    for move in optimal:
        new_board = board[:]
        new_board[move] = p
        legal_after = sum(1 for i in range(9) if new_board[i] == 0)
        total_branches += legal_after
        count += 1
    return total_branches / count if count > 0 else 0.0


def reply_branching_factor_after_best_reply(board: List[int], sol: Dict, solved_map: Dict[str, dict]) -> float:
    optimal = sol['optimal_moves']
    if not optimal:
        return 0.0
    p = 1 if board.count(1) == board.count(2) else 2
    opp = 2 if p == 1 else 1
    totals: List[float] = []
    for move in optimal:
        b2 = board[:]
        b2[move] = p
        key2 = serialize_board(b2)
        if key2 not in solved_map:
            continue
        sol2 = solved_map[key2]
        opp_opts = list(sol2['optimal_moves'])
        if not opp_opts:
            totals.append(0)
            continue
        branch_sum = 0
        for r in opp_opts:
            b3 = b2[:]
            b3[r] = opp
            branch_sum += sum(1 for i in range(9) if b3[i] == 0)
        totals.append(branch_sum / len(opp_opts))
    return float(np.mean(totals)) if totals else 0.0


# ---------- Feature extraction (from current-player perspective) ----------

def extract_board_features(
    board: List[int],
    solved_map: Dict[str, dict],
    lambda_temp: float = 0.5,
    q_temp: float = 1.0,
    epsilons: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if epsilons is None:
        epsilons = [0.1]
    normalized_board = board[:]  # side-to-move normalization is implicit in solver/targets here
    x_count, o_count = get_piece_counts(normalized_board)
    winner_raw = get_winner(board)
    winner_norm = get_winner(normalized_board)
    current_player = 1 if board.count(1) == board.count(2) else 2
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
        # Targets and policies
        policy_targets = build_policy_targets(normalized_board, sol, lambda_temp=lambda_temp, q_temp=q_temp)
        pol_uniform = policy_targets['policy_optimal_uniform']
        pol_soft = policy_targets['policy_soft_dtt']
        pol_soft_q = policy_targets['policy_soft_q']
        eps_policies = {}
        eps_values = {}
        for eps in epsilons:
            tag = f"{int(round(eps*100)):03d}"
            eps_policies[tag] = epsilon_policy_distribution(normalized_board, sol, eps)
            vx, dlen = epsilon_optimal_expectation(tuple(board), eps)
            eps_values[tag] = (vx, dlen)
        pol_entropy = -sum(p * np.log(p + 1e-10) for p in pol_uniform if p > 0)
        pol_soft_dtt_entropy = -sum(p * np.log(p + 1e-10) for p in pol_soft if p > 0)
        child_tiers = child_qtier_histogram(sol)
        difficulty = difficulty_score(sol)
        reply_branching = reply_branching_factor_after_best(board, sol)
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
        eps_values = {}
        pol_entropy = None
        pol_soft_dtt_entropy = None
        child_tiers = {'child_wins': 0, 'child_draws': 0, 'child_losses': 0}
        difficulty = 0.0
        reply_branching = 0.0

    sym = symmetry_info(board)
    legal = [normalized_board[i] == 0 for i in range(9)]
    legal_count = sum(legal)
    best_mask = [(i in optimal_moves) for i in range(9)] if reachable else [False]*9

    x_threats = calculate_line_threats(normalized_board, 1)
    o_threats = calculate_line_threats(normalized_board, 2)
    x_connectivity = calculate_connectivity(normalized_board, 1)
    o_connectivity = calculate_connectivity(normalized_board, 2)
    x_patterns = calculate_pattern_strength(normalized_board, 1)
    o_patterns = calculate_pattern_strength(normalized_board, 2)
    control = calculate_control_metrics(normalized_board)
    cell_pot = calculate_cell_line_potentials(normalized_board)

    current_immediate_wins = len(immediate_winning_moves(normalized_board, current_player))
    opponent_immediate_wins = len(immediate_winning_moves(normalized_board, 2 if current_player == 1 else 1))
    must_block_now = opponent_immediate_wins > 0
    x_two_open = count_two_in_row_open(normalized_board, 1)
    o_two_open = count_two_in_row_open(normalized_board, 2)

    safe_moves = 0
    create_threat_moves = 0
    for i in range(9):
        if normalized_board[i] != 0:
            continue
        if not gives_opponent_immediate_win(normalized_board, current_player, i):
            safe_moves += 1
    if creates_immediate_threat_oneply(normalized_board, current_player, i):
            create_threat_moves += 1

    rv_x, rlen = random_policy_expectation(tuple(board))
    side_to_move = 1 if board.count(1) == board.count(2) else 2
    rv_current = rv_x if side_to_move == 1 else -rv_x

    margin = compute_margin_features(sol) if reachable else {
        'best_q_value': None, 'best_dtt': None, 'second_q_value': None, 'second_dtt': None,
        'value_gap_levels': None, 'dtt_gap': None, 'dominated_count': None, 'legal_count': legal_count,
    }

    features: Dict[str, Any] = {
        'board_state': key,
        'normalized_board_state': norm_key,
        'swapped_color': 0,  # normalized simplification
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
        'game_phase': calculate_game_phase(board),
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
        'winner_perfect': 0 if value_current == 0 else (current_player if value_current == +1 else (2 if current_player == 1 else 1)) if value_current is not None else None,
        'plies_to_end': plies_to_end,
        'optimal_moves_count': len(optimal_moves),
        'optimal_policy_entropy': pol_entropy,
        'policy_soft_dtt_entropy': pol_soft_dtt_entropy,
        'policy_soft_q_entropy': (
            -sum(p * np.log(p + 1e-10) for p in pol_soft_q if p > 0) if reachable else None
        ),
        'best_q_value': margin['best_q_value'],
        'best_dtt': margin['best_dtt'],
        'second_q_value': margin['second_q_value'],
        'second_dtt': margin['second_dtt'],
        'value_gap_levels': margin['value_gap_levels'],
        'dtt_gap': margin['dtt_gap'],
        'dominated_moves_count': margin['dominated_count'],
        'plies_to_win_current': (plies_to_end if (value_current == +1) else None) if reachable else None,
        'plies_to_lose_current': (plies_to_end if (value_current == -1) else None) if reachable else None,
        'plies_to_win_X': None,
        'plies_to_win_O': None,
        'legal_move_count': legal_count,
        'current_player_immediate_wins': current_immediate_wins,
        'opponent_immediate_wins': opponent_immediate_wins,
        'must_block_now': must_block_now,
        'safe_move_count': safe_moves,
        'creates_immediate_threat_moves': create_threat_moves,
        'x_two_in_row_open': x_two_open,
        'o_two_in_row_open': o_two_open,
        'center_occupied': normalized_board[4] != 0,
        'center_owner': normalized_board[4],
        'corners_x': sum(1 for i in [0,2,6,8] if normalized_board[i] == 1),
        'corners_o': sum(1 for i in [0,2,6,8] if normalized_board[i] == 2),
        'edges_x': sum(1 for i in [1,3,5,7] if normalized_board[i] == 1),
        'edges_o': sum(1 for i in [1,3,5,7] if normalized_board[i] == 2),
        'x_row_threats': x_threats['row_threats'],
        'x_col_threats': x_threats['col_threats'],
        'x_diag_threats': x_threats['diag_threats'],
        'x_total_threats': x_threats['total_threats'],
        'o_row_threats': o_threats['row_threats'],
        'o_col_threats': o_threats['col_threats'],
        'o_diag_threats': o_threats['diag_threats'],
        'o_total_threats': o_threats['total_threats'],
        'x_connected_pairs': x_connectivity['connected_pairs'],
        'x_isolated_pieces': x_connectivity['isolated_pieces'],
        'x_cluster_count': x_connectivity['cluster_count'],
        'x_largest_cluster': x_connectivity['largest_cluster'],
        'o_connected_pairs': o_connectivity['connected_pairs'],
        'o_isolated_pieces': o_connectivity['isolated_pieces'],
        'o_cluster_count': o_connectivity['cluster_count'],
        'o_largest_cluster': o_connectivity['largest_cluster'],
        'x_open_lines': x_patterns['open_lines'],
        'x_potential_lines': x_patterns['potential_lines'],
        'o_open_lines': o_patterns['open_lines'],
        'o_potential_lines': o_patterns['potential_lines'],
        'x_control_score': control['x_control_score'],
        'o_control_score': control['o_control_score'],
        'x_control_pct': control.get('x_control_percentage', 0.0),
        'o_control_pct': control.get('o_control_percentage', 0.0),
        'control_difference': control['control_difference'],
        'board_density': (x_count + o_count) / 9,
        'random_value_x': rv_x,
        'random_value_current': rv_current,
        'random_expected_plies_to_end': rlen,
        'difficulty_score': difficulty,
        'reply_branching_factor': reply_branching,
        'reply_branching_after_best_reply': (
            reply_branching_factor_after_best_reply(board, sol, solved_map) if reachable else 0.0
        ),
        **child_tiers,
    }

    for i in range(9):
        features[f'cell_{i}'] = normalized_board[i]
        features[f'x_plane_{i}'] = int(normalized_board[i] == 1)
        features[f'o_plane_{i}'] = int(normalized_board[i] == 2)
        features[f'empty_plane_{i}'] = int(normalized_board[i] == 0)
        features[f'to_move_plane_{i}'] = 1 if (board.count(1) == board.count(2)) else 0
        features[f'legal_{i}'] = int(legal[i])
        features[f'best_{i}'] = int(best_mask[i])
        features[f'q_value_{i}'] = qvals[i]
        features[f'dtt_action_{i}'] = dtt_a[i]
        features[f'canonical_action_map_{i}'] = apply_action_transform(i, sym['canonical_op'])

    for i in range(9):
        features[f'policy_uniform_{i}'] = pol_uniform[i] if reachable else None
        features[f'policy_soft_{i}'] = pol_soft[i] if reachable else None
        features[f'policy_soft_q_{i}'] = pol_soft_q[i] if reachable else None
        for tag, pol in (eps_policies.items() if reachable else []):
            features[f'epsilon_policy_{tag}_{i}'] = pol[i]

    for tag, (vx, dlen) in (eps_values.items() if reachable else []):
        features[f'epsilon_value_x_{tag}'] = vx
        features[f'epsilon_expected_plies_{tag}'] = dlen

    for i in range(9):
        features[f'x_cell_open_lines_{i}'] = cell_pot['x_cell_open_lines'][i]
        features[f'o_cell_open_lines_{i}'] = cell_pot['o_cell_open_lines'][i]

    return features


# ---------- State-action rows ----------

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
    seen_canonical = set()
    for key, sol in solved_map.items():
        b = deserialize_board(key)
        p = 1 if b.count(1) == b.count(2) else 2
        sym = symmetry_info(b)
        canonical = sym['canonical_form']
        if canonical_only:
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
        pol = build_policy_targets(b, sol, lambda_temp=lambda_temp, q_temp=q_temp)
        ranks, v_regret, dtt_regret = compute_action_ranks(sol)
        eps_pols = {f"{int(round(e*100)):03d}": epsilon_policy_distribution(b, sol, e) for e in epsilons}
        base_actions: List[Dict[str, Any]] = []
        for i in range(9):
            if b[i] != 0:
                continue
            child = b[:]
            child[i] = p
            r = 0
            w = get_winner(child)
            done = (w != 0) or is_draw(child)
            if done:
                r = 1 if w == p else (0 if w == 0 else -1)
            tactical = action_tactical_labels(b, i, sol=sol)
            robust = action_robustness_metrics(b, i, solved_map)
            canonical_action = apply_action_transform(i, sym['canonical_op'])
            orbit_id = f"{canonical}:{canonical_action}"
            next_key = serialize_board(child)
            next_val = solved_map[next_key]['value'] if next_key in solved_map else None
            next_dtt = solved_map[next_key]['plies_to_end'] if next_key in solved_map else None
            base_actions.append({
                'board_state': key,
                'canonical_form': canonical,
                'player_to_move': p,
                'action': i,
                'next_state': next_key,
                'reward_if_terminal': r,
                'done': int(done),
                'q_value': sol['q_values'][i],
                'dtt_action': sol['dtt_action'][i],
                'optimal_action': int(i in sol['optimal_moves']),
                'policy_optimal_uniform': pol['policy_optimal_uniform'][i],
                'policy_soft_dtt': pol['policy_soft_dtt'][i],
                'policy_soft_q': pol['policy_soft_q'][i],
                **{f'epsilon_policy_{tag}': eps_pols[tag][i] for tag in eps_pols},
                'rank': ranks[i],
                'value_regret': v_regret[i],
                'dtt_regret': dtt_regret[i],
                'canonical_action': int(canonical_action),
                'action_orbit_id': orbit_id,
                'orbit_int_id': orbit_int_id(canonical, canonical_action),
                'advantage': None if sol['q_values'][i] is None else float(sol['q_values'][i] - sol['value']),
                'within_tier_pref': None,  # omitted for brevity
                'within_tier_margin': None,
                'next_value_current': next_val,
                'next_plies_to_end': next_dtt,
                **tactical,
                **robust,
            })
        rows.extend(base_actions)
        if include_augmentation:
            for sym_op in ALL_SYMS[1:]:
                transformed_board = transform_board(b, sym_op)
                transformed_key = serialize_board(transformed_board)
                transformed_optimal = {apply_action_transform(m, sym_op) for m in sol['optimal_moves']}
                transformed_pol = {k: [0.0]*9 for k in ['policy_optimal_uniform','policy_soft_dtt','policy_soft_q']}
                for orig_idx in range(9):
                    j = apply_action_transform(orig_idx, sym_op)
                    transformed_pol['policy_optimal_uniform'][j] = pol['policy_optimal_uniform'][orig_idx]
                    transformed_pol['policy_soft_dtt'][j] = pol['policy_soft_dtt'][orig_idx]
                    transformed_pol['policy_soft_q'][j] = pol['policy_soft_q'][orig_idx]
                transformed_eps = {}
                for tag in eps_pols:
                    transformed_eps[tag] = [0.0]*9
                    for orig_idx in range(9):
                        j = apply_action_transform(orig_idx, sym_op)
                        transformed_eps[tag][j] = eps_pols[tag][orig_idx]
                base_by_action = {a['action']: a for a in base_actions}
                for i in range(9):
                    if transformed_board[i] != 0:
                        continue
                    orig_action = None
                    for j in range(9):
                        if apply_action_transform(j, sym_op) == i:
                            orig_action = j
                            break
                    if orig_action is None:
                        continue
                    orig_data = base_by_action.get(orig_action)
                    if orig_data is None:
                        continue
                    child = transformed_board[:]
                    child[i] = p
                    sym_t = symmetry_info(transformed_board)
                    canonical_action_t = apply_action_transform(i, sym_t['canonical_op'])
                    orbit_id_t = f"{sym_t['canonical_form']}:{canonical_action_t}"
                    child_key = serialize_board(child)
                    next_val_t = solved_map[child_key]['value'] if child_key in solved_map else None
                    next_dtt_t = solved_map[child_key]['plies_to_end'] if child_key in solved_map else None
                    rows.append({
                        'board_state': transformed_key,
                        'canonical_form': sym_t['canonical_form'],
                        'player_to_move': p,
                        'action': i,
                        'next_state': child_key,
                        'reward_if_terminal': orig_data['reward_if_terminal'],
                        'done': orig_data['done'],
                        'q_value': orig_data['q_value'] if orig_data['q_value'] is not None else None,
                        'dtt_action': orig_data['dtt_action'] if orig_data['dtt_action'] is not None else None,
                        'optimal_action': int(i in transformed_optimal),
                        'policy_optimal_uniform': transformed_pol['policy_optimal_uniform'][i],
                        'policy_soft_dtt': transformed_pol['policy_soft_dtt'][i],
                        'policy_soft_q': transformed_pol['policy_soft_q'][i],
                        **{f'epsilon_policy_{tag}': transformed_eps[tag][i] for tag in transformed_eps},
                        'rank': orig_data['rank'],
                        'value_regret': orig_data['value_regret'],
                        'dtt_regret': orig_data['dtt_regret'],
                        'legal': int(transformed_board[i] == 0),
                        'is_center': int(i == 4),
                        'is_corner': int(i in [0, 2, 6, 8]),
                        'is_edge': int(i in [1, 3, 5, 7]),
                        'is_immediate_win': orig_data['is_immediate_win'],
                        'is_immediate_block': orig_data['is_immediate_block'],
                        'creates_fork': orig_data['creates_fork'],
                        'blocks_opponent_fork': orig_data['blocks_opponent_fork'],
                        'gives_opponent_immediate_win': orig_data['gives_opponent_immediate_win'],
                        'creates_immediate_threat': orig_data['creates_immediate_threat'],
                        'creates_trap_in_2': orig_data.get('creates_trap_in_2', 0),
                        'is_forced_move': orig_data['is_forced_move'],
                        'is_safe_move': orig_data['is_safe_move'],
                        'augmentation_op': sym_op,
                        'canonical_action': int(canonical_action_t),
                        'action_orbit_id': orbit_id_t,
                        'orbit_int_id': orbit_int_id(sym_t['canonical_form'], canonical_action_t),
                        'advantage': None,
                        'within_tier_pref': None,
                        'within_tier_margin': None,
                        'next_value_current': next_val_t,
                        'next_plies_to_end': next_dtt_t,
                    })
    return rows

"""
Positional features and control utilities for Tic-Tac-Toe.
"""
from typing import Dict, List
from .game_basics import WIN_PATTERNS

WIN_PATTERNS_BY_TYPE = {
    'row': [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    'col': [[0, 3, 6], [1, 4, 7], [2, 5, 8]],
    'diag': [[0, 4, 8], [2, 4, 6]],
}


def calculate_line_threats(board: List[int], player: int) -> Dict[str, int]:
    threats = {
        'row_threats': 0,
        'col_threats': 0,
        'diag_threats': 0,
        'total_threats': 0,
    }
    for line_type, patterns in WIN_PATTERNS_BY_TYPE.items():
        for pattern in patterns:
            player_count = sum(1 for i in pattern if board[i] == player)
            opponent_count = sum(1 for i in pattern if board[i] != 0 and board[i] != player)
            if player_count > 0 and opponent_count == 0:
                threat_level = player_count
                if line_type == 'row':
                    threats['row_threats'] += threat_level
                elif line_type == 'col':
                    threats['col_threats'] += threat_level
                else:
                    threats['diag_threats'] += threat_level
                threats['total_threats'] += threat_level
    return threats


def calculate_connectivity(board: List[int], player: int) -> Dict[str, int]:
    adjacency = {
        0: [1, 3, 4], 1: [0, 2, 3, 4, 5], 2: [1, 4, 5],
        3: [0, 1, 4, 6, 7], 4: [0, 1, 2, 3, 5, 6, 7, 8], 5: [1, 2, 4, 7, 8],
        6: [3, 4, 7], 7: [3, 4, 5, 6, 8], 8: [4, 5, 7],
    }
    player_positions = [i for i in range(9) if board[i] == player]
    connectivity = {
        'connected_pairs': 0,
        'total_connections': 0,
        'isolated_pieces': 0,
        'cluster_count': 0,
        'largest_cluster': 0,
    }
    for pos in player_positions:
        connections = 0
        for adj in adjacency[pos]:
            if board[adj] == player:
                connectivity['connected_pairs'] += 1
                connections += 1
        connectivity['total_connections'] += connections
        if connections == 0:
            connectivity['isolated_pieces'] += 1
    connectivity['connected_pairs'] //= 2
    connectivity['total_connections'] //= 2
    visited = set()
    clusters = []
    for start_pos in player_positions:
        if start_pos not in visited:
            cluster = []
            queue = [start_pos]
            visited.add(start_pos)
            while queue:
                pos = queue.pop(0)
                cluster.append(pos)
                for adj in adjacency[pos]:
                    if board[adj] == player and adj not in visited:
                        visited.add(adj)
                        queue.append(adj)
            clusters.append(cluster)
    connectivity['cluster_count'] = len(clusters)
    connectivity['largest_cluster'] = max((len(c) for c in clusters), default=0)
    return connectivity


def calculate_control_metrics(board: List[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    position_weights = {4: 3, 0: 2, 2: 2, 6: 2, 8: 2, 1: 1, 3: 1, 5: 1, 7: 1}
    x_control = sum(position_weights[i] for i in range(9) if board[i] == 1)
    o_control = sum(position_weights[i] for i in range(9) if board[i] == 2)
    total_control = sum(position_weights.values())
    metrics['x_control_score'] = x_control
    metrics['o_control_score'] = o_control
    metrics['x_control_percentage'] = x_control / total_control if total_control > 0 else 0.0
    metrics['o_control_percentage'] = o_control / total_control if total_control > 0 else 0.0
    metrics['control_difference'] = x_control - o_control
    return metrics


def calculate_pattern_strength(board: List[int], player: int) -> Dict[str, int]:
    patterns = {
        'open_lines': 0,
        'semi_open_lines': 0,
        'blocked_lines': 0,
        'potential_lines': 0,
    }
    opponent = 2 if player == 1 else 1
    for pattern in WIN_PATTERNS:
        player_count = sum(1 for i in pattern if board[i] == player)
        opponent_count = sum(1 for i in pattern if board[i] == opponent)
        empty_count = sum(1 for i in pattern if board[i] == 0)
        if player_count > 0 and opponent_count == 0:
            patterns['open_lines'] += 1
            if empty_count > 0:
                patterns['potential_lines'] += 1
        elif player_count > 0 and opponent_count == 1:
            patterns['semi_open_lines'] += 1
        elif player_count > 0 and opponent_count > 1:
            patterns['blocked_lines'] += 1
    return patterns


def calculate_cell_line_potentials(board: List[int]) -> Dict[str, List[int]]:
    """Per-cell potential open lines if the player played here (empties only).

    For each empty cell i, x_cell_open_lines[i] counts how many winning lines
    remain open for X if X were to play at i (i.e., no O on the line), and
    similarly for O. For occupied cells, the values are 0.
    """
    x_pot = [0] * 9
    o_pot = [0] * 9
    for i in range(9):
        if board[i] == 0:
            cnt_x = 0
            cnt_o = 0
            for pat in WIN_PATTERNS:
                if i in pat:
                    if not any(board[j] == 2 for j in pat):
                        cnt_x += 1
                    if not any(board[j] == 1 for j in pat):
                        cnt_o += 1
            x_pot[i] = cnt_x
            o_pot[i] = cnt_o
        # occupied cells remain 0 by definition
    return {'x_cell_open_lines': x_pot, 'o_cell_open_lines': o_pot}


def calculate_game_phase(board: List[int]) -> str:
    move_count = sum(1 for cell in board if cell != 0)
    if move_count <= 2:
        return 'opening'
    elif move_count <= 5:
        return 'midgame'
    else:
        return 'endgame'


def count_two_in_row_open(board: List[int], player: int) -> int:
    cnt = 0
    for pat in WIN_PATTERNS:
        p = sum(1 for i in pat if board[i] == player)
        e = sum(1 for i in pat if board[i] == 0)
        if p == 2 and e == 1:
            cnt += 1
    return cnt

"""
Trajectory generation under specified teacher policy.
"""
from typing import Dict, List
import numpy as np
from .solver import solve_state, legal_moves, current_player_t, apply_move_t, winner_t, is_draw_t


def generate_trajectories(policy: str = 'optimal', epsilon: float = 0.1, max_games: int = 1000, seed: int = 42) -> List[List[Dict]]:
    rng = np.random.default_rng(seed)
    games: List[List[Dict]] = []
    for _ in range(max_games):
        s = tuple([0] * 9)
        game: List[Dict] = []
        while True:
            w = winner_t(s)
            if w != 0 or is_draw_t(s):
                break
            p = current_player_t(s)
            key = ''.join(map(str, s))
            sol = solve_state(s)
            moves = legal_moves(s)
            if policy == 'optimal':
                choices = list(sol['optimal_moves'])
                mv = int(rng.choice(choices))
            elif policy == 'random':
                mv = int(rng.choice(moves))
            else:
                if rng.random() < epsilon:
                    mv = int(rng.choice(moves))
                else:
                    mv = int(rng.choice(list(sol['optimal_moves'])))
            entry = {
                'state': key,
                'player_to_move': p,
                'action': mv,
                'value': sol['value'],
                'dtt_action': sol['dtt_action'][mv],
            }
            game.append(entry)
            s = apply_move_t(s, mv, p)
        games.append(game)
    return games

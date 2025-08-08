"""
Tactics and simple motifs: immediate wins/blocks, forks, safety checks.
Teaching notes:
- Local motifs provide strong signals for learning before deep search.
"""
from typing import List
from .game_basics import get_winner
from .solver import current_player_t

WIN_PATTERNS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6],
]


def immediate_winning_moves(board: List[int], player: int) -> List[int]:
    wins: List[int] = []
    for i, v in enumerate(board):
        if v != 0:
            continue
        b = board[:]
        b[i] = player
        if get_winner(b) == player:
            wins.append(i)
    return wins


def fork_moves(board: List[int], player: int) -> List[int]:
    forks: List[int] = []
    for i, v in enumerate(board):
        if v != 0:
            continue
        b = board[:]
        b[i] = player
        if len(immediate_winning_moves(b, player)) >= 2:
            forks.append(i)
    return forks


def gives_opponent_immediate_win(board: List[int], player: int, move: int) -> bool:
    if board[move] != 0:
        return False
    opp = 2 if player == 1 else 1
    b = board[:]
    b[move] = player
    return len(immediate_winning_moves(b, opp)) > 0

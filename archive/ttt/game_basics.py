"""
Game basics: board representation, serialization, rules, winner/draw checks, validity.
Teaching notes:
- State is a list of 9 cells: 0=empty, 1=X, 2=O. X always starts.
- A "ply" is a half-move (one player's turn).
- Valid states have counts either equal (O to move) or X has one more (X just moved).
"""
from typing import List, Tuple

WIN_PATTERNS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]


def serialize_board(board: List[int]) -> str:
    return ''.join(str(cell) for cell in board)


def deserialize_board(board_str: str) -> List[int]:
    return [int(cell) for cell in board_str]


def get_winner(board: List[int]) -> int:
    for pattern in WIN_PATTERNS:
        a, b, c = pattern
        v = board[a]
        if v != 0 and v == board[b] and v == board[c]:
            return v
    return 0


def is_draw(board: List[int]) -> bool:
    return 0 not in board and get_winner(board) == 0


def get_piece_counts(board: List[int]) -> Tuple[int, int]:
    return board.count(1), board.count(2)


def is_valid_state(board: List[int]) -> bool:
    x_count, o_count = get_piece_counts(board)
    if not (x_count == o_count or x_count == o_count + 1):
        return False
    w = get_winner(board)
    if w == 1 and x_count != o_count + 1:
        return False
    if w == 2 and x_count != o_count:
        return False
    # no double winners
    def count_wins(p: int) -> int:
        return sum(1 for pat in WIN_PATTERNS if all(board[i] == p for i in pat))
    if count_wins(1) > 0 and count_wins(2) > 0:
        return False
    return True


def current_player(board: List[int]) -> int:
    x, o = get_piece_counts(board)
    return 1 if x == o else 2

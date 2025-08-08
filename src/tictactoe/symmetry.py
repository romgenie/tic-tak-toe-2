"""
Symmetry and canonicalization for Tic-Tac-Toe.
Teaching notes:
- There are 8 symmetries (the dihedral group of the square). Using them reduces redundancy.
- We canonicalize a board by taking the lexicographically smallest image among all symmetries.
- Actions (cell indices) transform with the board; we precompute index maps.
"""
from functools import lru_cache
from typing import Dict, List

from .game_basics import serialize_board

ALL_SYMS = ['id', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip', 'd1', 'd2']


def transform_board(board: List[int], kind: str) -> List[int]:
    b = board
    if kind == 'id':
        return b[:]
    elif kind == 'rot90':
        return [b[6], b[3], b[0], b[7], b[4], b[1], b[8], b[5], b[2]]
    elif kind == 'rot180':
        return [b[8], b[7], b[6], b[5], b[4], b[3], b[2], b[1], b[0]]
    elif kind == 'rot270':
        return [b[2], b[5], b[8], b[1], b[4], b[7], b[0], b[3], b[6]]
    elif kind == 'hflip':
        return [b[2], b[1], b[0], b[5], b[4], b[3], b[8], b[7], b[6]]
    elif kind == 'vflip':
        return [b[6], b[7], b[8], b[3], b[4], b[5], b[0], b[1], b[2]]
    elif kind == 'd1':
        return [b[0], b[3], b[6], b[1], b[4], b[7], b[2], b[5], b[8]]
    elif kind == 'd2':
        return [b[8], b[5], b[2], b[7], b[4], b[1], b[6], b[3], b[0]]
    else:
        raise ValueError(f"Unknown transformation: {kind}")


def sym_index_map(kind: str) -> List[int]:
    mapping = [None] * 9  # type: ignore
    for i in range(9):
        b = [0] * 9
        b[i] = 1
        tb = transform_board(b, kind)
        mapping[i] = tb.index(1)
    return mapping  # type: ignore


SYMM_INDEX_MAPS = {k: sym_index_map(k) for k in ALL_SYMS}


def apply_action_transform(action: int, kind: str) -> int:
    return SYMM_INDEX_MAPS[kind][action]


@lru_cache(maxsize=None)
def _symmetry_info_tuple(board_t: tuple) -> Dict:
    board = list(board_t)
    images = []
    for k in ALL_SYMS:
        transformed = transform_board(board, k)
        images.append((serialize_board(transformed), k))
    images_sorted = sorted(images, key=lambda x: x[0])
    canonical_str, canonical_op = images_sorted[0]
    unique_set = sorted(set(s for s, _ in images))
    unique_images = len(unique_set)

    board_str = serialize_board(board)
    # horizontal_symmetric: symmetric across a horizontal axis -> use hflip
    # vertical_symmetric: symmetric across a vertical axis -> use vflip
    horizontal_symmetric = serialize_board(transform_board(board, 'hflip')) == board_str
    vertical_symmetric = serialize_board(transform_board(board, 'vflip')) == board_str
    main_diag_symmetric = serialize_board(transform_board(board, 'd1')) == board_str
    anti_diag_symmetric = serialize_board(transform_board(board, 'd2')) == board_str
    rot_symmetric_180 = serialize_board(transform_board(board, 'rot180')) == board_str

    any_symmetric = any([
        horizontal_symmetric, vertical_symmetric,
        main_diag_symmetric, anti_diag_symmetric,
        rot_symmetric_180,
    ])
    orbit_index = unique_set.index(board_str)

    return {
        'canonical_form': canonical_str,
        'canonical_op': canonical_op,
        'orbit_size': unique_images,
        'orbit_index': orbit_index,
        'horizontal_symmetric': horizontal_symmetric,
        'vertical_symmetric': vertical_symmetric,
        'diagonal_symmetric': main_diag_symmetric or anti_diag_symmetric,
        'rotational_symmetric': rot_symmetric_180,
        'any_symmetric': any_symmetric,
    }


def symmetry_info(board: List[int]) -> Dict:
    return _symmetry_info_tuple(tuple(board))

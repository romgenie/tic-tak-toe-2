import os, sys, importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

spec = importlib.util.spec_from_file_location(
    "ttt_module",
    os.path.join(SRC, "01_generate_game_states_enhanced.py"),
)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore

transform_board = module.transform_board
symmetry_info = module.symmetry_info
apply_action_transform = module.apply_action_transform
ALL_SYMS = module.ALL_SYMS


def test_action_transform_roundtrip():
    # For each symmetry, applying transform then inverse should recover index
    # In dihedral group, each element is its own inverse except 90/270 rotations.
    inverse = {
        'id': 'id',
        'rot90': 'rot270',
        'rot180': 'rot180',
        'rot270': 'rot90',
        'hflip': 'hflip',
        'vflip': 'vflip',
        'd1': 'd1',
        'd2': 'd2',
    }
    for k, inv in inverse.items():
        for i in range(9):
            j = apply_action_transform(i, k)
            i2 = apply_action_transform(j, inv)
            assert i2 == i


def test_canonical_is_lexicographically_minimum():
    board = [1,0,2, 0,1,0, 2,0,0]
    images = ["".join(map(str, transform_board(board, k))) for k in ALL_SYMS]
    canonical = min(images)
    sym = symmetry_info(board)
    assert sym['canonical_form'] == canonical

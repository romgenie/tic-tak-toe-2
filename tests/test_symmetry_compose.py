from tictactoe.symmetry import SYMM_INDEX_MAPS, transform_board


def test_symmetry_maps_compose_identity():
    # rot90 composed 4 times should be identity on indices
    m = SYMM_INDEX_MAPS["rot90"]
    idx = list(range(9))
    for _ in range(4):
        idx = [m[i] for i in idx]
    assert idx == list(range(9))


def test_transform_roundtrip_identity():
    b = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    b2 = transform_board(transform_board(b, "rot90"), "rot270")
    assert b2 == b

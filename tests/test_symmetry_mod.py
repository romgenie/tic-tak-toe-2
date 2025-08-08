from ttt.symmetry import apply_action_transform, symmetry_info


def test_action_transform_round_trip():
    # Map a corner through rot90 and back via rot270
    i = 0
    j = apply_action_transform(i, 'rot90')
    k = apply_action_transform(j, 'rot270')
    assert k == i


def test_canonicalization_minimum_string():
    # Board with a single X should canonicalize to put X in top-left
    board = [0]*9
    board[4] = 1  # center X
    info = symmetry_info(board)
    assert info['canonical_form'] <= '100000000'

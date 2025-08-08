
import pytest

from tictactoe.game_basics import get_winner, is_draw
from tictactoe.orchestrator import extract_board_features
from tictactoe.solver import solve_all_reachable


@pytest.fixture(scope="module")
def solved_map():
    return solve_all_reachable()


def _pick_midgame_reachable_nonterminal(solved_map):
    for k in solved_map.keys():
        b = [int(c) for c in k]
        if get_winner(b) == 0 and not is_draw(b) and sum(1 for x in b if x != 0) >= 3:
            return b
    raise RuntimeError("No suitable midgame state found")


def test_features_difficulty_and_child_tiers_nonzero_for_reachable(solved_map):
    b = _pick_midgame_reachable_nonterminal(solved_map)
    feats = extract_board_features(b, solved_map)
    assert feats['reachable_from_start'] is True
    # Difficulty computed and non-negative
    assert isinstance(feats['difficulty_score'], float)
    assert feats['difficulty_score'] >= 0.0
    total_children = feats['child_wins'] + feats['child_draws'] + feats['child_losses']
    assert total_children > 0


def test_reply_branching_factor_positive_when_legal_replies_exist(solved_map):
    b = _pick_midgame_reachable_nonterminal(solved_map)
    feats = extract_board_features(b, solved_map)
    assert isinstance(feats['reply_branching_factor'], float)
    assert feats['reply_branching_factor'] >= 0.0


def test_extended_feature_columns_present_and_types(solved_map):
    b = _pick_midgame_reachable_nonterminal(solved_map)
    feats = extract_board_features(b, solved_map, normalize_to_move=True)
    # Spot-check extended columns
    keys = [
        'x_row_threats','o_row_threats','x_connected_pairs','o_connected_pairs',
        'x_open_lines','o_open_lines','x_two_in_row_open','o_two_in_row_open',
        'x_control_score','o_control_score','control_difference','game_phase'
    ]
    for k in keys:
        assert k in feats, f"missing feature {k}"
    assert feats['game_phase'] in {'opening','midgame','endgame'}

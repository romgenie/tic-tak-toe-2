#!/usr/bin/env python3
"""
Thin shim: the original monolithic generator has been archived under `archive/`.
This shim re-exports its public API so tests and existing imports keep working.
For new workflows, prefer `src/10_cli.py export` or `ttt.datasets` with --use-orchestrator.
"""
from pathlib import Path
import importlib.util
import sys as _sys

_ARCHIVED = Path(__file__).resolve().parents[1] / 'archive' / '01_generate_game_states_enhanced.py'
_SRC_COPY = Path(__file__).resolve()  # this file; try to load original if still present via archive

def _load_module_from(path: Path):
    spec = importlib.util.spec_from_file_location('ttt_monolith_archived', str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

_mod = None
if _ARCHIVED.exists():
    _mod = _load_module_from(_ARCHIVED)
else:
    # Fallback: if someone removed archive, try to load ourselves as a module (no-op)
    _mod = _load_module_from(_SRC_COPY)

# Re-export all public attributes
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith('_')})

def main(argv=None):  # delegate CLI behavior if present
    if hasattr(_mod, 'main'):
        return _mod.main(argv)
    # If archived module expects argparse in __main__, mimic call
    if hasattr(_mod, 'generate_tic_tac_toe_dataset'):
        import argparse
        parser = argparse.ArgumentParser(description='Generate Tic-Tac-Toe datasets (archived monolith)')
        parser.add_argument('--no-augmentation', action='store_true')
        parser.add_argument('--no-canonical-only', action='store_true')
        parser.add_argument('--all-states', action='store_true')
        parser.add_argument('--csv', action='store_true')
        parser.add_argument('--no-npz', action='store_true')
        parser.add_argument('--lambda-temp', type=float, default=0.5)
        parser.add_argument('--q-temp', type=float, default=1.0)
        parser.add_argument('--epsilons', type=str, default='0.1')
        parser.add_argument('--states-canonical-only', action='store_true')
        parser.add_argument('--seed', type=int, default=42)
        args = parser.parse_args(argv)
        eps = [float(x) for x in args.epsilons.split(',') if x.strip()]
        return _mod.generate_tic_tac_toe_dataset(
            include_augmentation=not args.no_augmentation,
            canonical_only=not args.no_canonical_only,
            reachable_only=not args.all_states,
            valid_only=not args.all_states,
            export_parquet=not args.csv,
            export_npz=not args.no_npz,
            lambda_temp=args.lambda_temp,
            q_temp=args.q_temp,
            states_canonical_only=args.states_canonical_only,
            epsilons=eps,
            seed=args.seed,
        )
    raise SystemExit('Archived monolith not found or unsupported entry point.')

if __name__ == '__main__':
    raise SystemExit(main())


# ========== Symmetry Transformations ==========

def transform_board(board: List[int], kind: str) -> List[int]:
    """Apply symmetry transformation to board."""
    b = board
    if kind == 'id':      # identity
        return b[:]
    elif kind == 'rot90':
        return [b[6], b[3], b[0], b[7], b[4], b[1], b[8], b[5], b[2]]
    elif kind == 'rot180':
        return [b[8], b[7], b[6], b[5], b[4], b[3], b[2], b[1], b[0]]
    elif kind == 'rot270':
        return [b[2], b[5], b[8], b[1], b[4], b[7], b[0], b[3], b[6]]
    elif kind == 'hflip':  # mirror left-right (vertical axis)
        return [b[2], b[1], b[0], b[5], b[4], b[3], b[8], b[7], b[6]]
    elif kind == 'vflip':  # mirror top-bottom (horizontal axis)
        return [b[6], b[7], b[8], b[3], b[4], b[5], b[0], b[1], b[2]]
    elif kind == 'd1':     # main diagonal
        return [b[0], b[3], b[6], b[1], b[4], b[7], b[2], b[5], b[8]]
    elif kind == 'd2':     # anti-diagonal
        return [b[8], b[5], b[2], b[7], b[4], b[1], b[6], b[3], b[0]]
    else:
        raise ValueError(f"Unknown transformation: {kind}")


ALL_SYMS = ['id', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip', 'd1', 'd2']


def sym_index_map(kind: str) -> List[int]:
    """Map old index -> new index under transform 'kind'."""
    mapping = [None]*9
    for i in range(9):
        b = [0]*9
        b[i] = 1
        tb = transform_board(b, kind)
        mapping[i] = tb.index(1)
    return mapping


# Precompute index maps for all symmetries
SYMM_INDEX_MAPS = {k: sym_index_map(k) for k in ALL_SYMS}


def apply_action_transform(action: int, kind: str) -> int:
    """Transform an action index under a given symmetry."""
    return SYMM_INDEX_MAPS[kind][action]


@lru_cache(maxsize=None)
def _symmetry_info_tuple(board_t: tuple) -> Dict:
    """Cached symmetry information for an immutable board (tuple)."""
    board = list(board_t)
    images = []
    for k in ALL_SYMS:
        transformed = transform_board(board, k)
        images.append((serialize_board(transformed), k))

    images_sorted = sorted(images, key=lambda x: x[0])
    canonical_str, canonical_op = images_sorted[0]
    unique_set = sorted(set(s for s, _ in images))
    unique_images = len(unique_set)

    # Symmetry booleans (naming follows geometry; see transform_board comments)
    board_str = serialize_board(board)
    # Note: hflip/vflip names refer to mirror axis; booleans below use geometric orientation:
    # - horizontal_symmetric == symmetry across a horizontal line (top-bottom mirror, 'vflip')
    # - vertical_symmetric   == symmetry across a vertical line (left-right mirror, 'hflip')
    horizontal_symmetric = serialize_board(transform_board(board, 'vflip')) == board_str
    vertical_symmetric = serialize_board(transform_board(board, 'hflip')) == board_str
    main_diag_symmetric = serialize_board(transform_board(board, 'd1')) == board_str
    anti_diag_symmetric = serialize_board(transform_board(board, 'd2')) == board_str
    rot_symmetric_180 = serialize_board(transform_board(board, 'rot180')) == board_str

    any_symmetric = any([
        horizontal_symmetric, vertical_symmetric,
        main_diag_symmetric, anti_diag_symmetric,
        rot_symmetric_180
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
        'any_symmetric': any_symmetric
    }


def symmetry_info(board: List[int]) -> Dict:
    """Calculate comprehensive symmetry information for board (with caching)."""
    return _symmetry_info_tuple(tuple(board))


# ========== Exact Solver Utilities ==========

def board_to_tuple(board: List[int]) -> tuple:
    """Convert board list to tuple for hashing."""
    return tuple(board)


def tuple_to_list(t: tuple) -> List[int]:
    """Convert tuple back to list."""
    return list(t)


def legal_moves(board_t: tuple) -> List[int]:
    """Get list of legal move indices."""
    return [i for i, v in enumerate(board_t) if v == 0]


def current_player_t(board_t: tuple) -> int:
    """Determine current player from board state."""
    x = board_t.count(1)
    o = board_t.count(2)
    return 1 if x == o else 2


def apply_move_t(board_t: tuple, idx: int, player: int) -> tuple:
    """Apply a move to the board and return new board tuple."""
    lst = list(board_t)
    lst[idx] = player
    return tuple(lst)


def winner_t(board_t: tuple) -> int:
    """Get winner for tuple board."""
    return get_winner(list(board_t))


def is_draw_t(board_t: tuple) -> bool:
    """Check if tuple board is a draw."""
    return is_draw(list(board_t))


# ========== Color Normalization (current-player perspective) ==========
def swap_colors(board: List[int]) -> List[int]:
    """Swap X(1) and O(2) on the board, keep empties."""
    out = []
    for v in board:
        if v == 1:
            out.append(2)
        elif v == 2:
            out.append(1)
        else:
            out.append(0)
    return out


def normalize_to_current_player(board: List[int]) -> Tuple[List[int], bool]:
    """Return (normalized_board, swapped) so that X is always the side to move.

    If original side to move is X, returns board and swapped=False.
    If original side to move is O, returns color-swapped board and swapped=True.
    Action indices are unchanged by color swap.
    """
    x = board.count(1)
    o = board.count(2)
    p = 1 if x == o else 2
    if p == 1:
        return board[:], False
    else:
        return swap_colors(board), True


# ========== Exact Game-Theoretic Solver (BUG FIXED) ==========

def better_of(a: int, b: int) -> int:
    """Compare values from current side's perspective: +1 > 0 > -1."""
    order = {+1: 2, 0: 1, -1: 0}
    return a if order[a] > order[b] else b


@lru_cache(maxsize=None)
def solve_state(board_t: tuple) -> Dict:
    """
    Solve a state exactly using memoized minimax.
    Returns dict with:
    - value: {-1, 0, 1} from side to move perspective
    - plies_to_end: distance to game end under perfect play
    - optimal_moves: tuple of best move indices
    - q_values: tuple of 9 values (or None for illegal moves)
    - dtt_action: tuple of 9 plies-to-end values per action
    """
    w = winner_t(board_t)
    if w != 0:
        # Terminal: winner is the previous player; side-to-move has already lost
        return {
            'value': -1,
            'plies_to_end': 0,
            'optimal_moves': tuple(),
            'q_values': tuple([None]*9),
            'dtt_action': tuple([None]*9)
        }
    
    if is_draw_t(board_t):
        return {
            'value': 0,
            'plies_to_end': 0,
            'optimal_moves': tuple(),
            'q_values': tuple([None]*9),
            'dtt_action': tuple([None]*9)
        }
    
    p = current_player_t(board_t)
    moves = legal_moves(board_t)
    q_vals = [None]*9
    dtt_action = [None]*9
    
    best_val = None
    best_dtt = None
    best_moves = []
    
    for mv in moves:
        child = apply_move_t(board_t, mv, p)
        s_child = solve_state(child)
        # child.value is from opponent's perspective; action value for current player is -child.value
        q = -s_child['value']
        q_vals[mv] = q
        dtt_action[mv] = 1 + s_child['plies_to_end']
        
        # Handle first move or update best
        if best_val is None:
            best_val = q
            best_dtt = dtt_action[mv]
            best_moves = [mv]
            continue
        
        # Update best according to value then distance tie-break
        if better_of(q, best_val) == q and q != best_val:
            # new strictly better value
            best_val = q
            best_dtt = dtt_action[mv]
            best_moves = [mv]
        elif q == best_val:
            # tie-break on distances
            if q == +1:
                # For wins, prefer shorter distance
                if dtt_action[mv] < best_dtt:
                    best_dtt = dtt_action[mv]
                    best_moves = [mv]
                elif dtt_action[mv] == best_dtt:
                    best_moves.append(mv)
            elif q == -1:
                # For losses, prefer longer distance (delay loss)
                if dtt_action[mv] > best_dtt:
                    best_dtt = dtt_action[mv]
                    best_moves = [mv]
                elif dtt_action[mv] == best_dtt:
                    best_moves.append(mv)
            else:  # draw (q == 0)
                # For draws, prefer shorter games
                if dtt_action[mv] < best_dtt:
                    best_dtt = dtt_action[mv]
                    best_moves = [mv]
                elif dtt_action[mv] == best_dtt:
                    best_moves.append(mv)
    
    return {
        'value': best_val,
        'plies_to_end': best_dtt,
        'optimal_moves': tuple(sorted(best_moves)),
        'q_values': tuple(q_vals),
        'dtt_action': tuple(dtt_action)
    }


def solve_all_reachable() -> Dict[str, dict]:
    """Build and solve all reachable states."""
    all_nodes = {}
    q = deque()
    start = tuple([0]*9)
    q.append(start)
    seen = {start}
    
    # Build reachable states
    while q:
        s = q.popleft()
        all_nodes[s] = True
        if winner_t(s) != 0 or is_draw_t(s):
            continue
        p = current_player_t(s)
        for mv in legal_moves(s):
            child = apply_move_t(s, mv, p)
            if child not in seen:
                seen.add(child)
                q.append(child)
    
    # Force solve via memoized function
    solved = {}
    for s in all_nodes:
        res = solve_state(s)
        solved[''.join(map(str, s))] = res
    
    return solved


# ========== Tactical Utilities ==========

def immediate_winning_moves(board: List[int], player: int) -> List[int]:
    """Find all moves that immediately win for the player."""
    wins = []
    for i, v in enumerate(board):
        if v != 0:
            continue
        b = board[:]
        b[i] = player
        if get_winner(b) == player:
            wins.append(i)
    return wins


def count_immediate_wins(board: List[int], player: int) -> int:
    """Count number of immediate winning moves for player."""
    return len(immediate_winning_moves(board, player))


def fork_moves(board: List[int], player: int) -> List[int]:
    """Find moves that create a fork (two or more ways to win next turn)."""
    forks = []
    for i, v in enumerate(board):
        if v != 0:
            continue
        b = board[:]
        b[i] = player
        # Count distinct next winning moves created
        wins_next = immediate_winning_moves(b, player)
        if len(wins_next) >= 2:
            forks.append(i)
    return forks


def blocks_opponent_immediate_win(board: List[int], player: int, move: int) -> bool:
    """Check if a move blocks an opponent's immediate win."""
    if board[move] != 0:
        return False
    opp = 2 if player == 1 else 1
    opp_wins = set(immediate_winning_moves(board, opp))
    if not opp_wins:
        return False
    # Check if this move blocks at least one winning move
    return move in opp_wins


def gives_opponent_immediate_win(board: List[int], player: int, move: int) -> bool:
    """Check if a move gives the opponent an immediate win on their next turn."""
    if board[move] != 0:
        return False
    opp = 2 if player == 1 else 1
    b = board[:]
    b[move] = player
    return len(immediate_winning_moves(b, opp)) > 0


def count_two_in_row_open(board: List[int], player: int) -> int:
    """Count lines where player has two marks and the third is empty."""
    cnt = 0
    for pat in WIN_PATTERNS:
        p = sum(1 for i in pat if board[i] == player)
        e = sum(1 for i in pat if board[i] == 0)
        if p == 2 and e == 1:
            cnt += 1
    return cnt


def creates_immediate_threat(board: List[int], player: int, move: int) -> bool:
    """Whether playing 'move' creates an immediate winning threat (excluding direct wins)."""
    if board[move] != 0:
        return False
    nb = board[:]
    nb[move] = player
    if get_winner(nb) == player:
        return False
    return len(immediate_winning_moves(nb, player)) > 0


def compute_action_ranks(sol: Dict) -> Tuple[List[Optional[int]], List[Optional[int]], List[Optional[int]]]:
    """Compute rank, value_regret, and dtt_regret for each action given solver output.
    - rank: 1 is best, higher is worse; None for illegal moves
    - value_regret: difference in ordered value levels vs best (0 best), None for illegal
    - dtt_regret: tie-break regret within same value class (non-negative), None if different value or illegal
    """
    order_map = {+1: 2, 0: 1, -1: 0}

    q_values = list(sol['q_values'])
    dtts = list(sol['dtt_action'])

    legal_idxs = [i for i, q in enumerate(q_values) if q is not None]

    def key_for(i: int):
        q = q_values[i]
        if q is None:
            return (float('inf'), float('inf'), i)
        primary = -order_map[q]
        d = dtts[i]
        d_val = 10**9 if d is None else d
        if q == -1:
            secondary = -d_val
        else:
            secondary = d_val
        return (primary, secondary, i)

    sorted_moves = sorted(legal_idxs, key=key_for)

    ranks = [None] * 9
    value_regret = [None] * 9
    dtt_regret = [None] * 9

    if not sorted_moves:
        return ranks, value_regret, dtt_regret

    best_q = q_values[sorted_moves[0]]
    best_dtts_same_q = [dtts[i] for i in legal_idxs if q_values[i] == best_q and dtts[i] is not None]
    if best_dtts_same_q:
        if best_q == -1:
            # prefer longer for losses
            best_dtt_pref = max(best_dtts_same_q)
        else:
            best_dtt_pref = min(best_dtts_same_q)
    else:
        best_dtt_pref = None

    # Assign dense ranks (ties share the same rank)
    prev_key = None
    current_rank = 0
    for i in sorted_moves:
        k = key_for(i)[:2]
        if k != prev_key:
            current_rank += 1
            prev_key = k
        ranks[i] = current_rank
        value_regret[i] = order_map[best_q] - order_map[q_values[i]]
        if q_values[i] == best_q and dtts[i] is not None and best_dtt_pref is not None:
            if best_q == -1:
                dtt_regret[i] = max(0, best_dtt_pref - dtts[i])
            else:
                dtt_regret[i] = max(0, dtts[i] - best_dtt_pref)

    return ranks, value_regret, dtt_regret


def blocks_opponent_fork(board: List[int], player: int, move: int) -> bool:
    """True if playing 'move' eliminates all opponent fork threats next turn."""
    if board[move] != 0:
        return False
    opp = 2 if player == 1 else 1

    # If the opponent has NO fork threat available already, it's not a 'forced block'.
    opp_forks_now = fork_moves(board, opp)
    if not opp_forks_now:
        return False

    # After we move, check if the opponent still has any fork move.
    nb = board[:]
    nb[move] = player
    return len(fork_moves(nb, opp)) == 0


def action_tactical_labels(board: List[int], move: int, sol: Optional[Dict] = None) -> Dict[str, int]:
    """Generate tactical labels for a specific action.

    If sol is provided (solver output for the board), creates_trap_in_2 uses exact labels
    via (q==+1 and dtt_action==2); otherwise a heuristic fallback is used.
    """
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
        'is_safe_move': 0
    }
    
    if labels['legal'] == 0:
        return labels
    
    # Immediate win
    labels['is_immediate_win'] = int(move in immediate_winning_moves(board, p))
    # Immediate block
    labels['is_immediate_block'] = int(move in immediate_winning_moves(board, opp))
    # Forks
    labels['creates_fork'] = int(move in fork_moves(board, p))
    labels['blocks_opponent_fork'] = int(blocks_opponent_fork(board, p, move))
    # Safety
    labels['gives_opponent_immediate_win'] = int(gives_opponent_immediate_win(board, p, move))
    labels['is_safe_move'] = int(
        labels['legal'] and
        (not labels['gives_opponent_immediate_win']) and
        (blocks_opponent_fork(board, p, move) or len(fork_moves(board, opp)) == 0)
    )

    # Create an immediate winning threat (excluding direct immediate wins)
    if labels['legal'] and labels['is_immediate_win'] == 0:
        nb = board[:]
        nb[move] = p
        labels['creates_immediate_threat'] = int(len(immediate_winning_moves(nb, p)) > 0)
    
    # Add trap-in-2 detection
    if sol is not None and sol['q_values'][move] is not None:
        labels['creates_trap_in_2'] = int(sol['q_values'][move] == +1 and sol['dtt_action'][move] == 2)
    else:
        labels['creates_trap_in_2'] = int(count_trap_in_2(board, p, move))

    # Forced move: only one legal move avoids an immediate opponent win next ply
    opp_wins_now = immediate_winning_moves(board, opp)
    if opp_wins_now:
        legal_moves_now = [i for i, v in enumerate(board) if v == 0]
        safe_blockers = []
        for m in legal_moves_now:
            nb2 = board[:]
            nb2[m] = p
            # After our move, opponent should have no immediate winning reply
            if len(immediate_winning_moves(nb2, opp)) == 0:
                safe_blockers.append(m)
        labels['is_forced_move'] = int(len(safe_blockers) == 1 and move == safe_blockers[0])

    return labels


def action_robustness_metrics(board: List[int], move: int, solved_map: Dict[str, dict]) -> Dict[str, Optional[int]]:
    """Robustness metrics under opponent best replies.

    Returns counts of opponent replies by Q tier (win/draw/loss for current player),
    the worst_reply_q (min Q after opponent reply), and simple safe_k flags.
    """
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
        # Opponent has no replies; trivially robust
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
        # Value is from the side to move in b3 (us)
        q_after = solved_map[k3]['value']
        if q_after == +1:
            win_cnt += 1
        elif q_after == 0:
            draw_cnt += 1
        else:
            loss_cnt += 1
        worst_q = min(worst_q, q_after)
    safe_1ply = int(loss_cnt == 0)
    
    # Calculate safe_2ply: no opponent reply that forces our loss within 2 plies
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


# ========== ML-Friendly Policy Targets ==========

def build_policy_targets(
    board: List[int],
    sol: Dict,
    lambda_temp: float = 0.5,
    q_temp: float = 1.0,
) -> Dict[str, List[float]]:
    """Build normalized policy distribution targets.

    Returns three distributions (all length 9, summing to 1 over legal moves):
    - policy_optimal_uniform: uniform over optimal moves
    - policy_soft_dtt: soft weighting by distance-to-termination within value tiers
    - policy_soft_q: softmax over Q-values (from side-to-move perspective)
    """
    # Legal mask
    legal = [int(v == 0) for v in board]
    
    # Uniform over optimal moves
    optimal = list(sol['optimal_moves'])
    pol_uniform = [0.0] * 9
    if optimal:
        mass = 1.0 / len(optimal)
        for m in optimal:
            pol_uniform[m] = mass
    
    # Soft by dtt_action (shorter distance more probable)
    dtt = sol['dtt_action']
    weights = []
    for i in range(9):
        if not legal[i] or dtt[i] is None:
            weights.append(0.0)
        else:
            # Favor wins strongly, neutral for draws, de-emphasize losing moves
            q = sol['q_values'][i]
            base = 1.0 if q == 0 else (2.0 if q == +1 else 0.5)
            weights.append(base * np.exp(-lambda_temp * dtt[i]))
    
    s = sum(weights)
    pol_soft = [w/s if s > 0 else 0.0 for w in weights]

    # Softmax over Q-values only (no dtt), controlled by q_temp
    # Map Q in {-1,0,+1} to scores and apply exp
    q_vals = sol['q_values']
    sm_weights = []
    for i in range(9):
        if not legal[i] or q_vals[i] is None:
            sm_weights.append(0.0)
        else:
            sm_weights.append(np.exp(q_temp * float(q_vals[i])))
    z = sum(sm_weights)
    pol_soft_q = [w/z if z > 0 else 0.0 for w in sm_weights]
    
    return {
        'policy_optimal_uniform': pol_uniform,
        'policy_soft_dtt': pol_soft,
        'policy_soft_q': pol_soft_q,
    }


def epsilon_policy_distribution(board: List[int], sol: Dict, epsilon: float) -> List[float]:
    """Per-action epsilon-optimal policy: mix optimal-uniform with legal-uniform.
    mass = (1-eps)/|optimal| if in optimal else 0 + eps/|legal| if legal else 0.
    """
    legal = [i for i, v in enumerate(board) if v == 0]
    optimal = list(sol['optimal_moves'])
    nL = len(legal) if legal else 1
    nO = len(optimal) if optimal else 1
    pol = [0.0] * 9
    for i in legal:
        a = ((1.0 - epsilon) * (1.0 / nO if i in optimal else 0.0)) + (epsilon * (1.0 / nL))
        pol[i] = a
    return pol


# ========== Constant-Perspective Values ==========

def value_from_x_perspective(board: List[int], solved_map: Dict[str, dict]) -> Optional[int]:
    """Get value from X's perspective regardless of current player."""
    key = serialize_board(board)
    if key not in solved_map:
        return None
    val_current = solved_map[key]['value']
    p = 1 if board.count(1) == board.count(2) else 2
    # If current player is X, value_current is already X-perspective
    # If current player is O, flip sign to get X perspective
    return val_current if p == 1 else -val_current


def perfect_value_current(board: List[int], solved_map: Dict[str, dict]) -> Optional[int]:
    """Get perfect-play value from current player's perspective."""
    key = serialize_board(board)
    return solved_map[key]['value'] if key in solved_map else None


def perfect_optimal_moves(board: List[int], solved_map: Dict[str, dict]) -> List[int]:
    """Get list of optimal moves under perfect play."""
    key = serialize_board(board)
    return list(solved_map[key]['optimal_moves']) if key in solved_map else []


def perfect_q_values(board: List[int], solved_map: Dict[str, dict]) -> List:
    """Get Q-values for all actions."""
    key = serialize_board(board)
    return list(solved_map[key]['q_values']) if key in solved_map else [None]*9


def perfect_dtt(board: List[int], solved_map: Dict[str, dict]) -> Optional[int]:
    """Get plies to end under perfect play."""
    key = serialize_board(board)
    return solved_map[key]['plies_to_end'] if key in solved_map else None


def perfect_winner(board: List[int], solved_map: Dict[str, dict]) -> Optional[int]:
    """Determine winner under perfect play (0=draw, 1=X, 2=O)."""
    p = 1 if board.count(1) == board.count(2) else 2
    val = perfect_value_current(board, solved_map)
    if val is None:
        return None
    if val == 0:
        return 0  # draw
    return p if val == +1 else (2 if p == 1 else 1)


def player_distance_to_win(board: List[int], player: int, solved_map: Dict[str, dict]) -> int:
    """Get distance to win for a specific player under perfect play."""
    w = perfect_winner(board, solved_map)
    if w == player:
        return perfect_dtt(board, solved_map)
    return -1


# ========== Random-policy expectations (useful baselines for ML) ==========
@lru_cache(maxsize=None)
def random_policy_expectation(board_t: tuple) -> Tuple[float, float]:
    """Expected (value_x, plies_to_end) under uniform random play from this state.

    value_x is from X's perspective: +1 X win, 0 draw, -1 O win.
    plies_to_end is the expected number of moves remaining.
    """
    w = winner_t(board_t)
    if w != 0:
        # Terminal board already has a winner; no more plies
        return (1.0 if w == 1 else -1.0, 0.0)
    if is_draw_t(board_t):
        return (0.0, 0.0)

    moves = legal_moves(board_t)
    if not moves:
        return (0.0, 0.0)
    p = current_player_t(board_t)
    # Under random play, the player to move selects uniformly at random
    vs = []
    ds = []
    for mv in moves:
        child = apply_move_t(board_t, mv, p)
        v_child, d_child = random_policy_expectation(child)
        vs.append(v_child)
        ds.append(d_child)
    # Average children; add 1 ply for the move we make here
    return (float(sum(vs)) / len(vs), 1.0 + float(sum(ds)) / len(ds))


@lru_cache(maxsize=None)
def epsilon_optimal_expectation(board_t: tuple, epsilon: float = 0.1) -> Tuple[float, float]:
    """Expected (value_x, plies_to_end) under epsilon-optimal play.

    With prob 1-epsilon choose uniformly among optimal moves, else uniformly among legal.
    Value from X perspective.
    """
    w = winner_t(board_t)
    if w != 0:
        return (1.0 if w == 1 else -1.0, 0.0)
    if is_draw_t(board_t):
        return (0.0, 0.0)
    p = current_player_t(board_t)
    moves = legal_moves(board_t)
    s = solve_state(board_t)
    optimal = list(s['optimal_moves'])
    if not moves:
        return (0.0, 0.0)
    nL = len(moves)
    nO = max(1, len(optimal))
    vs = 0.0
    ds = 0.0
    for mv in moves:
        wmv = ((1.0 - epsilon) * (1.0 / nO if mv in optimal else 0.0)) + (epsilon * (1.0 / nL))
        child = apply_move_t(board_t, mv, p)
        v_child, d_child = epsilon_optimal_expectation(child, epsilon)
        vs += wmv * v_child
        ds += wmv * d_child
    return (vs, 1.0 + ds)


# ========== Additional Metrics ==========

def calculate_line_threats(board: List[int], player: int) -> Dict[str, int]:
    """Calculate threat levels for each type of line."""
    threats = {
        'row_threats': 0,
        'col_threats': 0,
        'diag_threats': 0,
        'total_threats': 0
    }
    
    for line_type, patterns in WIN_PATTERNS_BY_TYPE.items():
        for pattern in patterns:
            player_count = sum(1 for i in pattern if board[i] == player)
            opponent_count = sum(1 for i in pattern if board[i] != 0 and board[i] != player)
            
            # Threat exists if player has pieces and opponent doesn't block
            if player_count > 0 and opponent_count == 0:
                threat_level = player_count  # 1 piece = low, 2 pieces = high threat
                if line_type == 'row':
                    threats['row_threats'] += threat_level
                elif line_type == 'col':
                    threats['col_threats'] += threat_level
                else:
                    threats['diag_threats'] += threat_level
                threats['total_threats'] += threat_level
    
    return threats


def calculate_connectivity(board: List[int], player: int) -> Dict[str, int]:
    """Calculate connectivity metrics for a player's pieces."""
    adjacency = {
        0: [1, 3, 4],
        1: [0, 2, 3, 4, 5],
        2: [1, 4, 5],
        3: [0, 1, 4, 6, 7],
        4: [0, 1, 2, 3, 5, 6, 7, 8],
        5: [1, 2, 4, 7, 8],
        6: [3, 4, 7],
        7: [3, 4, 5, 6, 8],
        8: [4, 5, 7]
    }
    
    player_positions = [i for i in range(9) if board[i] == player]
    
    connectivity = {
        'connected_pairs': 0,
        'total_connections': 0,
        'isolated_pieces': 0,
        'cluster_count': 0,
        'largest_cluster': 0
    }
    
    # Count connected pairs
    for pos in player_positions:
        connections = 0
        for adj in adjacency[pos]:
            if board[adj] == player:
                connectivity['connected_pairs'] += 1
                connections += 1
        connectivity['total_connections'] += connections
        if connections == 0:
            connectivity['isolated_pieces'] += 1
    
    # Divide by 2 because we counted each connection twice
    connectivity['connected_pairs'] //= 2
    connectivity['total_connections'] //= 2
    
    # Find clusters using BFS
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
    connectivity['largest_cluster'] = max(len(c) for c in clusters) if clusters else 0
    
    return connectivity


def calculate_control_metrics(board: List[int]) -> Dict[str, float]:
    """Calculate board control metrics."""
    metrics = {}
    
    # Position weights (center > corners > edges)
    position_weights = {
        4: 3,  # Center
        0: 2, 2: 2, 6: 2, 8: 2,  # Corners
        1: 1, 3: 1, 5: 1, 7: 1   # Edges
    }
    
    x_control = sum(position_weights[i] for i in range(9) if board[i] == 1)
    o_control = sum(position_weights[i] for i in range(9) if board[i] == 2)
    total_control = sum(position_weights.values())
    
    metrics['x_control_score'] = x_control
    metrics['o_control_score'] = o_control
    metrics['x_control_percentage'] = x_control / total_control if total_control > 0 else 0
    metrics['o_control_percentage'] = o_control / total_control if total_control > 0 else 0
    metrics['control_difference'] = x_control - o_control
    
    return metrics


def compute_margin_features(sol: Dict) -> Dict[str, Optional[float]]:
    """Compute best/second-best summaries and gaps among actions for a solved state.

    Uses the same ordering as compute_action_ranks (Q level, then DTT tie-break).
    Returns a dict with keys:
      - best_q_value, best_dtt
      - second_q_value, second_dtt
      - value_gap_levels (0 if same Q tier, 1 if best beats draw, 2 if beats loss)
      - dtt_gap (within-tier tie-break gap; positive means best preferred)
      - dominated_count (# legal moves with a strictly worse Q tier than best)
      - legal_count
    """
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
        if q == -1:
            secondary = -d_val  # prefer longer losses
        else:
            secondary = d_val   # prefer shorter wins or shorter draws
        return (primary, secondary, i)

    sorted_legal = sorted(legal, key=key_for)
    best = sorted_legal[0]
    best_q = q_values[best]
    best_d = dtts[best]
    second = sorted_legal[1] if len(sorted_legal) > 1 else None
    second_q = q_values[second] if second is not None else None
    second_d = dtts[second] if second is not None else None

    # Gap in value tiers; within-tier dtt gap definition depends on tier
    gap_levels = None if second_q is None else (order_map[best_q] - order_map[second_q])
    if second_q is None:
        dtt_gap = None
    elif best_q == second_q:
        if best_q == -1:
            dtt_gap = (best_d or 0) - (second_d or 0)  # positive if best delays more
        else:
            dtt_gap = (second_d or 0) - (best_d or 0)  # positive if best is faster
    else:
        dtt_gap = None

    dominated = sum(1 for i in legal if order_map[q_values[i]] < order_map[best_q])

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


def calculate_pattern_strength(board: List[int], player: int) -> Dict[str, int]:
    """Calculate various pattern strengths."""
    patterns = {
        'open_lines': 0,  # Lines with only player pieces and empty cells
        'semi_open_lines': 0,  # Lines with one opponent piece
        'blocked_lines': 0,  # Lines with multiple opponent pieces
        'potential_lines': 0  # Lines that could be completed
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
    """Per-cell count of open winning lines for each player.

    For each cell i and player p, count the number of winning lines that:
      - include cell i, and
      - contain no opponent stones (i.e., only player's stones and/or empty cells)

    If a cell is occupied by the opponent, its potential for the player is 0.
    """
    x_pot = [0] * 9
    o_pot = [0] * 9

    for i in range(9):
        # X potentials through i
        if board[i] != 2:  # not blocked by O at i
            cnt = 0
            for pat in WIN_PATTERNS:
                if i in pat:
                    if not any(board[j] == 2 for j in pat):
                        cnt += 1
            x_pot[i] = cnt
        # O potentials through i
        if board[i] != 1:  # not blocked by X at i
            cnt = 0
            for pat in WIN_PATTERNS:
                if i in pat:
                    if not any(board[j] == 1 for j in pat):
                        cnt += 1
            o_pot[i] = cnt

    return {
        'x_cell_open_lines': x_pot,
        'o_cell_open_lines': o_pot,
    }


def calculate_game_phase(board: List[int]) -> str:
    """Determine the phase of the game."""
    move_count = sum(1 for cell in board if cell != 0)
    
    if move_count <= 2:
        return 'opening'
    elif move_count <= 5:
        return 'midgame'
    else:
        return 'endgame'


# ========== Enhanced Feature Extraction ==========

def extract_board_features(
    board: List[int],
    solved_map: Dict[str, dict],
    lambda_temp: float = 0.5,
    q_temp: float = 1.0,
    epsilons: Optional[List[float]] = None,
) -> Dict:
    """Extract comprehensive features for a given board state."""
    if epsilons is None:
        epsilons = [0.1]
    # Normalize to current-player perspective
    normalized_board, swapped = normalize_to_current_player(board)
    x_count, o_count = get_piece_counts(normalized_board)
    winner_raw = get_winner(board)  # Winner on original board (1=X, 2=O, 0=none/draw)
    winner_norm = get_winner(normalized_board)  # Winner after color swap
    current_player = 1  # by normalization
    
    # Perfect-play labels if reachable
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
        winner_perf = perfect_winner(board, solved_map)
        x_dist = player_distance_to_win(board, 1, solved_map)
        o_dist = player_distance_to_win(board, 2, solved_map)
        value_x = value_from_x_perspective(board, solved_map)

        # Build policy targets
        policy_targets = build_policy_targets(normalized_board, sol, lambda_temp=lambda_temp, q_temp=q_temp)
        pol_uniform = policy_targets['policy_optimal_uniform']
        pol_soft = policy_targets['policy_soft_dtt']
        pol_soft_q = policy_targets['policy_soft_q']
        # Epsilon policies per action and expectations per state
        eps_policies = {}
        eps_values = {}
        for eps in epsilons:
            tag = f"{int(round(eps*100)):03d}"
            eps_policies[tag] = epsilon_policy_distribution(normalized_board, sol, eps)
            vx, dlen = epsilon_optimal_expectation(tuple(board), eps)
            eps_values[tag] = (vx, dlen)

        # Calculate policy entropy
        pol_entropy = -sum(p * np.log(p + 1e-10) for p in pol_uniform if p > 0)
        pol_soft_dtt_entropy = -sum(p * np.log(p + 1e-10) for p in pol_soft if p > 0)

        # Add new supervision targets
        pol_soft_optimal = policy_softmax_over_optimal(sol, tau=1.0)
        child_tiers = child_qtier_histogram(sol)
        difficulty = difficulty_score(sol)
        reply_branching = reply_branching_factor_after_best(board, sol)
    else:
        value_current = None
        plies_to_end = None
        optimal_moves = set()
        qvals = [None]*9
        dtt_a = [None]*9
        winner_perf = None
        x_dist = -1
        o_dist = -1
        value_x = None
        pol_uniform = [None]*9
        pol_soft = [None]*9
        pol_soft_q = [None]*9
        eps_policies = {}
        eps_values = {}
        pol_entropy = None
        pol_soft_dtt_entropy = None
        pol_soft_optimal = [None]*9
        child_tiers = {'child_wins': 0, 'child_draws': 0, 'child_losses': 0}
        difficulty = 0.0
        reply_branching = 0.0
    
    # Symmetry info
    sym = symmetry_info(board)
    
    # Legal/optimal masks
    legal = [normalized_board[i] == 0 for i in range(9)]
    legal_count = sum(legal)
    best_mask = [(i in optimal_moves) for i in range(9)] if reachable else [False]*9
    
    # Calculate threat metrics for both players
    x_threats = calculate_line_threats(normalized_board, 1)
    o_threats = calculate_line_threats(normalized_board, 2)
    
    # Calculate connectivity for both players
    x_connectivity = calculate_connectivity(normalized_board, 1)
    o_connectivity = calculate_connectivity(normalized_board, 2)
    
    # Calculate pattern strength for both players
    x_patterns = calculate_pattern_strength(normalized_board, 1)
    o_patterns = calculate_pattern_strength(normalized_board, 2)
    
    # Control metrics
    control = calculate_control_metrics(normalized_board)

    # Per-cell open line potentials
    # Use normalized perspective consistently for per-cell potentials
    cell_pot = calculate_cell_line_potentials(normalized_board)

    # Immediate win availability
    x_immediate_wins = count_immediate_wins(normalized_board, 1)
    o_immediate_wins = count_immediate_wins(normalized_board, 2)
    current_immediate_wins = count_immediate_wins(normalized_board, current_player)
    opponent_immediate_wins = count_immediate_wins(normalized_board, 2 if current_player == 1 else 1)
    must_block_now = opponent_immediate_wins > 0

    # Two-in-a-row open counts
    x_two_open = count_two_in_row_open(normalized_board, 1)
    o_two_open = count_two_in_row_open(normalized_board, 2)

    # Safe/forcing move counts
    safe_moves = 0
    create_threat_moves = 0
    for i in range(9):
        if normalized_board[i] != 0:
            continue
        if not gives_opponent_immediate_win(normalized_board, current_player, i):
            safe_moves += 1
        if creates_immediate_threat(normalized_board, current_player, i):
            create_threat_moves += 1
    
    # Tactical counts
    x_immediate_wins = count_immediate_wins(normalized_board, 1)
    o_immediate_wins = count_immediate_wins(normalized_board, 2)
    x_fork_count = len(fork_moves(normalized_board, 1))
    o_fork_count = len(fork_moves(normalized_board, 2))
    
    # Random-policy expectations (baseline difficulty signals)
    rv_x, rlen = random_policy_expectation(tuple(board))
    # From current player's perspective - use actual side to move BEFORE normalization
    side_to_move = 1 if board.count(1) == board.count(2) else 2
    rv_current = rv_x if side_to_move == 1 else -rv_x

    # If we have solver output, compute margin/gap features
    margin = compute_margin_features(sol) if reachable else {
        'best_q_value': None,
        'best_dtt': None,
        'second_q_value': None,
        'second_dtt': None,
        'value_gap_levels': None,
        'dtt_gap': None,
        'dominated_count': None,
        'legal_count': legal_count,
    }

    features = {
        # Basic state information
        'board_state': key,
        'normalized_board_state': norm_key,
        'swapped_color': int(swapped),
        'x_count': x_count,
        'o_count': o_count,
        'empty_count': normalized_board.count(0),
        'move_number': x_count + o_count,
        'current_player': current_player,
        'is_terminal': winner_raw != 0 or is_draw(board),
        'winner': winner_raw,  # Always 1=X, 2=O on raw board
        'winner_normalized': winner_norm,  # Winner after normalization
        'is_draw': is_draw(board),
        'is_valid': is_valid_state(board),
        'reachable_from_start': reachable,
        'game_phase': calculate_game_phase(board),
        
        # Symmetry information
        'canonical_form': sym['canonical_form'],
        'canonical_op': sym['canonical_op'],
        'orbit_size': sym['orbit_size'],
        'horizontal_symmetric': sym['horizontal_symmetric'],
        'vertical_symmetric': sym['vertical_symmetric'],
        'diagonal_symmetric': sym['diagonal_symmetric'],
        'rotational_symmetric': sym['rotational_symmetric'],
        'any_symmetric': sym['any_symmetric'],
        'orbit_index': sym['orbit_index'],
        
        # Perfect-play targets
        'value_current': value_current,
        'value_x_perspective': value_x,
        'winner_perfect': winner_perf,
        'plies_to_end': plies_to_end,
        'optimal_moves_count': len(optimal_moves),
        'x_distance_to_win': x_dist if x_dist >= 0 else None,
        'o_distance_to_win': o_dist if o_dist >= 0 else None,
        'optimal_policy_entropy': pol_entropy,
    'policy_soft_dtt_entropy': pol_soft_dtt_entropy,
        'policy_soft_q_entropy': (
            -sum(p * np.log(p + 1e-10) for p in pol_soft_q if p > 0)
            if reachable else None
        ),
        'best_q_value': margin['best_q_value'],
        'best_dtt': margin['best_dtt'],
        'second_q_value': margin['second_q_value'],
        'second_dtt': margin['second_dtt'],
        'value_gap_levels': margin['value_gap_levels'],
        'dtt_gap': margin['dtt_gap'],
        'dominated_moves_count': margin['dominated_count'],
        
        # Distance-to-termination shaping
        'plies_to_win_current': (plies_to_end if (value_current == +1) else None) if reachable else None,
        'plies_to_lose_current': (plies_to_end if (value_current == -1) else None) if reachable else None,
        'plies_to_win_X': x_dist if (x_dist is not None and x_dist >= 0) else None,
        'plies_to_win_O': o_dist if (o_dist is not None and o_dist >= 0) else None,
        
        # Mobility
        'legal_move_count': legal_count,
        
        # Tactical features
        'x_immediate_wins': x_immediate_wins,
        'o_immediate_wins': o_immediate_wins,
        'current_player_immediate_wins': current_immediate_wins,
        'opponent_immediate_wins': opponent_immediate_wins,
        'must_block_now': must_block_now,
        'safe_move_count': safe_moves,
        'creates_immediate_threat_moves': create_threat_moves,
        'x_fork_count': x_fork_count,
        'o_fork_count': o_fork_count,
        
        # Positional features
        'center_occupied': normalized_board[4] != 0,
        'center_owner': normalized_board[4],
        'corners_x': sum(1 for i in [0, 2, 6, 8] if normalized_board[i] == 1),
        'corners_o': sum(1 for i in [0, 2, 6, 8] if normalized_board[i] == 2),
        'edges_x': sum(1 for i in [1, 3, 5, 7] if normalized_board[i] == 1),
        'edges_o': sum(1 for i in [1, 3, 5, 7] if normalized_board[i] == 2),
        'x_two_in_row_open': x_two_open,
        'o_two_in_row_open': o_two_open,
        
        # Threat analysis
        'x_row_threats': x_threats['row_threats'],
        'x_col_threats': x_threats['col_threats'],
        'x_diag_threats': x_threats['diag_threats'],
        'x_total_threats': x_threats['total_threats'],
        'o_row_threats': o_threats['row_threats'],
        'o_col_threats': o_threats['col_threats'],
        'o_diag_threats': o_threats['diag_threats'],
        'o_total_threats': o_threats['total_threats'],
        
        # Connectivity metrics
        'x_connected_pairs': x_connectivity['connected_pairs'],
        'x_isolated_pieces': x_connectivity['isolated_pieces'],
        'x_cluster_count': x_connectivity['cluster_count'],
        'x_largest_cluster': x_connectivity['largest_cluster'],
        'o_connected_pairs': o_connectivity['connected_pairs'],
        'o_isolated_pieces': o_connectivity['isolated_pieces'],
        'o_cluster_count': o_connectivity['cluster_count'],
        'o_largest_cluster': o_connectivity['largest_cluster'],
        
        # Pattern strength
        'x_open_lines': x_patterns['open_lines'],
        'x_potential_lines': x_patterns['potential_lines'],
    'x_semi_open_lines': x_patterns.get('semi_open_lines', 0),
    'x_blocked_lines': x_patterns.get('blocked_lines', 0),
        'o_open_lines': o_patterns['open_lines'],
    'o_potential_lines': o_patterns['potential_lines'],
    'o_semi_open_lines': o_patterns.get('semi_open_lines', 0),
    'o_blocked_lines': o_patterns.get('blocked_lines', 0),
        
        # Control metrics
        'x_control_score': control['x_control_score'],
        'o_control_score': control['o_control_score'],
    'x_control_pct': control.get('x_control_percentage', 0.0),
    'o_control_pct': control.get('o_control_percentage', 0.0),
        'control_difference': control['control_difference'],
        
        # Board density
        'board_density': (x_count + o_count) / 9,
        
        # Random-policy expectations
        'random_value_x': rv_x,
        'random_value_current': rv_current,
        'random_expected_plies_to_end': rlen,
        
        # New supervision targets
        'difficulty_score': difficulty,
        'reply_branching_factor': reply_branching,
        'reply_branching_after_best_reply': (
            reply_branching_factor_after_best_reply(board, sol, solved_map) if reachable else 0.0
        ),
        **child_tiers,  # Adds child_wins, child_draws, child_losses
    }
    
    # Add individual cell states
    for i in range(9):
        features[f'cell_{i}'] = normalized_board[i]
    # Binary planes for ML models
    for i in range(9):
        features[f'x_plane_{i}'] = int(normalized_board[i] == 1)
        features[f'o_plane_{i}'] = int(normalized_board[i] == 2)
        features[f'empty_plane_{i}'] = int(normalized_board[i] == 0)
        features[f'to_move_plane_{i}'] = 1  # normalized perspective: always X to move
    
    # Add legal move masks
    for i in range(9):
        features[f'legal_{i}'] = int(legal[i])
    
    # Add optimal move masks
    for i in range(9):
        features[f'best_{i}'] = int(best_mask[i])
    
    # Add Q-values
    for i in range(9):
        features[f'q_value_{i}'] = qvals[i]
    
    # Add distance-to-termination per action
    for i in range(9):
        features[f'dtt_action_{i}'] = dtt_a[i]
    
    # Add policy targets
    for i in range(9):
        features[f'policy_uniform_{i}'] = pol_uniform[i]
        features[f'policy_soft_{i}'] = pol_soft[i]
        features[f'policy_soft_q_{i}'] = pol_soft_q[i]
        features[f'policy_soft_optimal_{i}'] = pol_soft_optimal[i]
        for tag, pol in eps_policies.items():
            features[f'epsilon_policy_{tag}_{i}'] = pol[i]

    # Add epsilon expectations
    for tag, (vx, dlen) in eps_values.items():
        features[f'epsilon_value_x_{tag}'] = vx
        features[f'epsilon_expected_plies_{tag}'] = dlen

    # Add plane encodings for legal and optimal actions (useful for CNNs)
    for i in range(9):
        features[f'legal_plane_{i}'] = int(legal[i])
        features[f'best_plane_{i}'] = int(best_mask[i])

    # Add per-cell open line potentials
    for i in range(9):
        features[f'x_cell_open_lines_{i}'] = cell_pot['x_cell_open_lines'][i]
        features[f'o_cell_open_lines_{i}'] = cell_pot['o_cell_open_lines'][i]

    # Add canonical action mapping (index mapping from this state to its canonical representative)
    canon_op = sym['canonical_op']
    for i in range(9):
        features[f'canonical_action_map_{i}'] = apply_action_transform(i, canon_op)

    return features


# ========== Enhanced Dataset Generation ==========

def generate_state_action_dataset(
    solved_map: Dict[str, dict],
    include_augmentation: bool = False,
    canonical_only: bool = False,
    lambda_temp: float = 0.5,
    q_temp: float = 1.0,
    epsilons: Optional[List[float]] = None,
) -> List[Dict]:
    """Generate per-action dataset for reachable states with enhanced features."""
    rows = []
    seen_canonical = set()
    if epsilons is None:
        epsilons = [0.1]
    
    for key, sol in solved_map.items():
        b = deserialize_board(key)
        p = 1 if b.count(1) == b.count(2) else 2
        
        # Get symmetry info
        sym = symmetry_info(b)
        canonical = sym['canonical_form']
        
        # Skip if canonical-only and we've seen this canonical form
        if canonical_only:
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
        
        # Build policy targets
        pol = build_policy_targets(b, sol, lambda_temp=lambda_temp, q_temp=q_temp)
        # Ranks and regret metrics
        ranks, v_regret, dtt_regret = compute_action_ranks(sol)
        # Epsilon policy distributions (per state)
        eps_pols = {f"{int(round(e*100)):03d}": epsilon_policy_distribution(b, sol, e) for e in epsilons}
        
        # Helper to compute within-tier preference sign relative to best DTT
        def within_tier_margin(sol_local: Dict, idx: int) -> Optional[float]:
            """Calculate DTT margin within same Q-tier (continuous signal)."""
            qvals = list(sol_local['q_values'])
            dtts = list(sol_local['dtt_action'])
            if qvals[idx] is None:
                return None
            best_q = sol_local['value']
            if qvals[idx] != best_q:
                return 0.0  # Not in best tier
            # compute preferred dtt among best tier
            same = [dtts[i] for i, q in enumerate(qvals) if q == best_q and dtts[i] is not None]
            if not same or dtts[idx] is None:
                return 0.0
            best_pref = max(same) if best_q == -1 else min(same)
            if best_q == -1:
                # For losses: positive margin = better (longer)
                return float(dtts[idx] - best_pref)
            else:
                # For wins/draws: positive margin = better (shorter)
                return float(best_pref - dtts[idx])
        
        def within_tier_sign(sol_local: Dict, idx: int) -> Optional[int]:
            """Direction tag relative to tier preference: 0=equal, -1=worse (never +1 vs the best)."""
            qvals = list(sol_local['q_values'])
            dtts = list(sol_local['dtt_action'])
            q = qvals[idx]
            d = dtts[idx]
            if q is None or d is None:
                return None
            best_q = sol_local['value']
            if q != best_q:
                return 0
            # preferred dtt among best tier
            same = [dtts[i] for i, qv in enumerate(qvals) if qv == best_q and dtts[i] is not None]
            if not same:
                return 0
            pref = max(same) if best_q == -1 else min(same)
            # 0 if equal, -1 if worse than preferred; (never +1 vs the best)
            if d == pref:
                return 0
            if best_q == -1:
                return -1 if d < pref else 0
            else:
                return -1 if d > pref else 0

        # Generate base actions
        base_actions = []
        for i in range(9):
            if b[i] != 0:
                continue
            
            child = b[:]
            child[i] = p
            
            # Calculate immediate reward
            r = 0
            w = get_winner(child)
            done = (w != 0) or is_draw(child)
            
            if done:
                if w == p:
                    r = 1
                elif w == 0:
                    r = 0
                else:
                    r = -1
            
            # Get tactical labels
            tactical = action_tactical_labels(b, i, sol=sol)
            robust = action_robustness_metrics(b, i, solved_map)
            
            # Advantage and canonical action/orbit id
            canonical_action = apply_action_transform(i, sym['canonical_op'])
            orbit_id = f"{canonical}:{canonical_action}"
            advantage = None if sol['q_values'][i] is None else float(sol['q_values'][i] - sol['value'])
            next_key = serialize_board(child)
            next_val = solved_map[next_key]['value'] if next_key in solved_map else None
            next_dtt = solved_map[next_key]['plies_to_end'] if next_key in solved_map else None

            base_actions.append({
                'board_state': key,
                'canonical_form': canonical,
                'player_to_move': p,
                'action': i,
                'next_state': serialize_board(child),
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
                'advantage': advantage,
                'within_tier_pref': within_tier_sign(sol, i),
                'within_tier_margin': within_tier_margin(sol, i),
                'next_value_current': next_val,
                'next_plies_to_end': next_dtt,
                **tactical,
                **robust,
            })
        
        rows.extend(base_actions)
        
        # Generate augmented versions if requested
        if include_augmentation:
            # Note: Tactical labels are symmetry-invariant, so we reuse them from the base action.
            for sym_op in ALL_SYMS[1:]:  # Skip identity
                # Transform the board
                transformed_board = transform_board(b, sym_op)
                transformed_key = serialize_board(transformed_board)
                
                # Transform optimal moves and Q-values
                transformed_optimal = {apply_action_transform(m, sym_op) for m in sol['optimal_moves']}
                transformed_q = [None] * 9
                transformed_dtt = [None] * 9
                transformed_pol_uniform = [0.0] * 9
                transformed_pol_soft = [0.0] * 9
                
                for orig_idx in range(9):
                    new_idx = apply_action_transform(orig_idx, sym_op)
                    transformed_q[new_idx] = sol['q_values'][orig_idx]
                    transformed_dtt[new_idx] = sol['dtt_action'][orig_idx]
                    transformed_pol_uniform[new_idx] = pol['policy_optimal_uniform'][orig_idx]
                    transformed_pol_soft[new_idx] = pol['policy_soft_dtt'][orig_idx]
                # policy_soft_q transforms identically
                transformed_pol_soft_q = [0.0] * 9
                for orig_idx in range(9):
                    new_idx = apply_action_transform(orig_idx, sym_op)
                    transformed_pol_soft_q[new_idx] = pol['policy_soft_q'][orig_idx]
                    
                # Transform epsilon policies
                transformed_eps_pols = {}
                for tag in eps_pols:
                    transformed_eps_pols[tag] = [0.0] * 9
                    for orig_idx in range(9):
                        new_idx = apply_action_transform(orig_idx, sym_op)
                        transformed_eps_pols[tag][new_idx] = eps_pols[tag][orig_idx]
                # Transform ranks and regrets by action index mapping
                transformed_ranks = [None] * 9
                transformed_vreg = [None] * 9
                transformed_dttreg = [None] * 9
                for orig_idx in range(9):
                    new_idx = apply_action_transform(orig_idx, sym_op)
                    transformed_ranks[new_idx] = ranks[orig_idx]
                    transformed_vreg[new_idx] = v_regret[orig_idx]
                    transformed_dttreg[new_idx] = dtt_regret[orig_idx]
                
                # Build a quick lookup for original action rows by action index
                base_by_action = {a['action']: a for a in base_actions}

                # Generate actions for transformed board
                for i in range(9):
                    if transformed_board[i] != 0:
                        continue
                    
                    # Find original action index
                    orig_action = None
                    for j in range(9):
                        if apply_action_transform(j, sym_op) == i:
                            orig_action = j
                            break
                    
                    if orig_action is None:
                        continue
                    
                    # Use original base action data
                    orig_data = base_by_action.get(orig_action)
                    if orig_data is None:
                        continue
                    
                    child = transformed_board[:]
                    child[i] = p
                    
                    # Compute symmetry-specific canonical mapping/info for transformed board
                    sym_t = symmetry_info(transformed_board)
                    canonical_action_t = apply_action_transform(i, sym_t['canonical_op'])
                    orbit_id_t = f"{sym_t['canonical_form']}:{canonical_action_t}"
                    child_key = serialize_board(child)
                    next_val_t = solved_map[child_key]['value'] if child_key in solved_map else None
                    next_dtt_t = solved_map[child_key]['plies_to_end'] if child_key in solved_map else None
                    advantage_t = None if transformed_q[i] is None else float(transformed_q[i] - (solved_map[transformed_key]['value'] if transformed_key in solved_map else 0))

                    rows.append({
                        'board_state': transformed_key,
                        'canonical_form': sym_t['canonical_form'],
                        'player_to_move': p,
                        'action': i,
                        'next_state': child_key,
                        'reward_if_terminal': orig_data['reward_if_terminal'],
                        'done': orig_data['done'],
                        'q_value': transformed_q[i],
                        'dtt_action': transformed_dtt[i],
                        'optimal_action': int(i in transformed_optimal),
                        'policy_optimal_uniform': transformed_pol_uniform[i],
                        'policy_soft_dtt': transformed_pol_soft[i],
                        'policy_soft_q': transformed_pol_soft_q[i],
                        **{f'epsilon_policy_{tag}': transformed_eps_pols[tag][i] for tag in transformed_eps_pols},
                        'rank': transformed_ranks[i],
                        'value_regret': transformed_vreg[i],
                        'dtt_regret': transformed_dttreg[i],
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
                        'advantage': advantage_t,
                        'within_tier_pref': within_tier_sign(solved_map[transformed_key], i) if transformed_key in solved_map else None,
                        'within_tier_margin': within_tier_margin(solved_map[transformed_key], i) if transformed_key in solved_map else None,
                        'next_value_current': next_val_t,
                        'next_plies_to_end': next_dtt_t,
                    })
    
    return rows


def generate_pairwise_preferences(solved_map: Dict[str, dict]) -> List[Dict]:
    """Generate pairwise ranking data for RankNet/Plackett-Luce training."""
    rows = []
    for key, sol in solved_map.items():
        q = list(sol['q_values'])
        d = list(sol['dtt_action'])
        legal = [i for i, v in enumerate(q) if v is not None]
        if len(legal) < 2:
            continue
        # preference: higher tier wins; tie-break by DTT (shorter for win/draw; longer for loss)
        def pref_key(i):
            tier_rank = {+1: 2, 0: 1, -1: 0}[q[i]]
            dtt_rank = -d[i] if q[i] == -1 else d[i]
            return (-tier_rank, dtt_rank)
        sorted_legal = sorted(legal, key=pref_key)
        # produce all ordered pairs respecting the ranking
        for i in range(len(sorted_legal)):
            for j in range(i+1, len(sorted_legal)):
                a, b = sorted_legal[i], sorted_legal[j]
                rows.append({
                    'state': key, 
                    'action_a': a, 
                    'action_b': b, 
                    'label': 1,  # a preferred to b
                    'q_diff': q[a] - q[b],
                            'dtt_diff': (d[a] - d[b]) if (d[a] is not None and d[b] is not None) else None
                })
    return rows


def child_qtier_histogram(sol: Dict) -> Dict[str, int]:
    """Count child states by Q-tier."""
    q = [v for v in sol['q_values'] if v is not None]
    return {
        'child_wins': sum(1 for v in q if v == +1),
        'child_draws': sum(1 for v in q if v == 0),
        'child_losses': sum(1 for v in q if v == -1),
    }


def policy_softmax_over_optimal(sol: Dict, tau: float = 1.0) -> List[float]:
    """Max-entropy policy constrained to optimal tier."""
    q = sol['q_values']
    d = sol['dtt_action']
    val = sol['value']
    # legal & same tier
    idxs = [i for i, v in enumerate(q) if v is not None and v == val]
    if not idxs:
        return [0.0] * 9
    # score = -DTT for win/draw (shorter better), +DTT for loss (longer better)
    scores = []
    for i in range(9):
        if i in idxs and d[i] is not None:
            s = (-d[i]) if val in (+1, 0) else (+d[i])
            scores.append((i, s))
    if not scores:
        return [0.0] * 9
    mx = max(s for _, s in scores)
    exps = {i: np.exp((s - mx) / max(1e-6, tau)) for i, s in scores}
    Z = sum(exps.values())
    out = [exps[i] / Z if i in exps else 0.0 for i in range(9)]
    return out


def difficulty_score(sol: Dict) -> float:
    """Higher = harder decision (small tier gap + many near-ties + long lookahead)."""
    q = list(sol['q_values'])
    d = list(sol['dtt_action'])
    legal = [i for i, v in enumerate(q) if v is not None]
    if not legal:
        return 0.0
    best = max(q[i] for i in legal)
    worst = min(q[i] for i in legal)
    tier_gap = best - worst  # in {0, 1, 2}
    close = sum(1 for i in legal if q[i] == best)
    dtt_values = [d[i] for i in legal if d[i] is not None]
    dtt_mean = np.mean(dtt_values) if dtt_values else 0
    return float((1.0 if tier_gap == 0 else 0.5) * (1 + np.log1p(close)) * (1 + 0.1 * dtt_mean))


def orbit_int_id(canonical_form: str, canonical_action: int) -> int:
    """Pack canonical form and action into a single integer for deterministic sharding."""
    # 3^9 fits in 20 bits; pack (canonical form base-3) and action (4 bits)
    val = 0
    for c in canonical_form:  # '0' '1' '2'
        val = val * 3 + ord(c) - ord('0')
    return (val << 4) | (canonical_action & 0xF)


def count_trap_in_2(board: List[int], player: int, move: int) -> bool:
    """Check if move creates a forced win in 2 plies (threat > forced block cycles)."""
    if board[move] != 0:
        return False
    
    # Make the move
    new_board = board[:]
    new_board[move] = player
    
    # If it's an immediate win, not a trap-in-2
    if get_winner(new_board) == player:
        return False
    
    opp = 2 if player == 1 else 1
    
    # Check if we created multiple threats that can't all be blocked
    threats = immediate_winning_moves(new_board, player)
    if len(threats) >= 2:
        return True  # Fork - opponent can only block one
    
    # Check if we have a single threat that leads to another forced win
    if len(threats) == 1:
        # Simulate opponent blocking
        block_board = new_board[:]
        block_board[threats[0]] = opp
        
        # Check if we still have a winning continuation
        for next_move in range(9):
            if block_board[next_move] == 0:
                test_board = block_board[:]
                test_board[next_move] = player
                if get_winner(test_board) == player:
                    return True
                # Check if this creates another fork
                if len(immediate_winning_moves(test_board, player)) >= 2:
                    return True
    
    return False


def reply_branching_factor_after_best(board: List[int], sol: Dict) -> float:
    """Average legal move count after our optimal move (before opponent reply)."""
    optimal = sol['optimal_moves']
    if not optimal:
        return 0.0
    
    p = 1 if board.count(1) == board.count(2) else 2
    total_branches = 0
    count = 0
    
    for move in optimal:
        new_board = board[:]
        new_board[move] = p
        
        # Count legal moves after this optimal move
        legal_after = sum(1 for i in range(9) if new_board[i] == 0)
        total_branches += legal_after
        count += 1
    
    return total_branches / count if count > 0 else 0.0


def reply_branching_factor_after_best_reply(board: List[int], sol: Dict, solved_map: Dict[str, dict]) -> float:
    """Average legal move count after our optimal move and opponent's optimal reply."""
    optimal = sol['optimal_moves']
    if not optimal:
        return 0.0
    p = 1 if board.count(1) == board.count(2) else 2
    opp = 2 if p == 1 else 1
    totals = []
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


def generate_symmetry_pairs(solved_map: Dict[str, dict]) -> List[Dict[str, str]]:
    """Generate (state, transformed_state, sym_op) triplets for symmetry-consistency training.

    For each reachable state, include its transforms under all non-identity symmetries.
    Returns a list of dicts with keys: state, sym_op, transformed_state.
    """
    pairs: List[Dict[str, str]] = []
    for key in solved_map.keys():
        board = deserialize_board(key)
        for sym_op in ALL_SYMS[1:]:
            tboard = transform_board(board, sym_op)
            pairs.append({
                'state': key,
                'sym_op': sym_op,
                'transformed_state': serialize_board(tboard),
            })
    return pairs


def generate_trajectories(
    solved_map: Dict[str, dict],
    policy: str = 'optimal',
    epsilon: float = 0.1,
    max_games: int = 1000,
    seed: int = 42,
) -> List[List[Dict]]:
    """Generate sequential trajectories using a specified teacher policy.

    policy: 'optimal' | 'random' | 'epsilon'
    Returns a list of games; each game is a list of step dicts with keys:
      - state (board string), action (int), player_to_move (1/2), value (side-to-move), dtt_action[action]
    """
    rng = np.random.default_rng(seed)
    games: List[List[Dict]] = []
    for _ in range(max_games):
        s = tuple([0]*9)
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


def generate_tic_tac_toe_dataset(
    include_augmentation: bool = True,
    canonical_only: bool = True,
    reachable_only: bool = True,
    valid_only: bool = True,
    export_parquet: bool = True,
    export_npz: bool = True,
    lambda_temp: float = 0.5,
    q_temp: float = 1.0,
    states_canonical_only: bool = False,
    epsilons: Optional[List[float]] = None,
    normalize_perspective: bool = True,
    seed: int = 42,
):
    """Main function to generate comprehensive ML-friendly Tic-Tac-Toe datasets."""
    if pd is None:
        raise RuntimeError(
            "pandas import failed in this environment. Dataset export requires pandas. "
            "You can still import this module and use the pure-Python solver/utilities."
        )
    logger.info("Solving all reachable states with exact game-theoretic solution...")
    solved_map = solve_all_reachable()
    logger.info(f"Solved {len(solved_map)} reachable states")
    
    logger.info("Generating all 3^9 states with perfect-play labels...")
    if epsilons is None:
        epsilons = [0.1]
    all_states = []
    seen_state_canon = set()
    for state in itertools.product([0, 1, 2], repeat=9):
        board = list(state)
        
        # Apply filters if requested
        if valid_only and not is_valid_state(board):
            continue
        if reachable_only and serialize_board(board) not in solved_map:
            continue
        
        if states_canonical_only:
            canon = symmetry_info(board)['canonical_form']
            if canon in seen_state_canon:
                continue
            seen_state_canon.add(canon)

        feats = extract_board_features(board, solved_map, lambda_temp=lambda_temp, q_temp=q_temp, epsilons=epsilons)
        all_states.append(feats)
    
    # Create data_raw directory if it doesn't exist
    os.makedirs('data_raw', exist_ok=True)
    
    # Convert to DataFrame for better handling
    df_states = pd.DataFrame(all_states)
    
    # Write states dataset
    if export_parquet:
        out_states = 'data_raw/ttt_states.parquet'
        df_states.to_parquet(out_states, index=False)
    else:
        out_states = 'data_raw/ttt_states.csv'
        df_states.to_csv(out_states, index=False)
    
    logger.info(f"Generated {len(all_states)} total states")
    logger.info(f"Valid states: {sum(1 for s in all_states if s['is_valid'])}")
    logger.info(f"Reachable states: {sum(1 for s in all_states if s['reachable_from_start'])}")
    logger.info(f"Terminal states: {sum(1 for s in all_states if s['is_terminal'])}")
    logger.info(f"X wins: {sum(1 for s in all_states if s['winner'] == 1)}")
    logger.info(f"O wins: {sum(1 for s in all_states if s['winner'] == 2)}")
    logger.info(f"Draws: {sum(1 for s in all_states if s['is_draw'])}")
    logger.info(f"States file saved as: {out_states}")
    logger.info(f"Total features per state: {len(df_states.columns)}")
    
    # Generate and write state-action dataset
    logger.info("Generating state-action dataset...")
    action_rows = generate_state_action_dataset(
        solved_map,
        include_augmentation=include_augmentation,
        canonical_only=canonical_only,
        lambda_temp=lambda_temp,
        q_temp=q_temp,
        epsilons=epsilons,
    )
    
    df_actions = pd.DataFrame(action_rows)
    
    if export_parquet:
        out_actions = 'data_raw/ttt_state_actions.parquet'
        df_actions.to_parquet(out_actions, index=False)
    else:
        out_actions = 'data_raw/ttt_state_actions.csv'
        df_actions.to_csv(out_actions, index=False)
    
    logger.info(f"Generated {len(action_rows)} state-action pairs")
    logger.info(f"Optimal actions: {sum(1 for r in action_rows if r['optimal_action'])}")
    if include_augmentation:
        logger.info("Augmented samples included (8x multiplier)")
    if canonical_only:
        logger.info("Canonical forms only (reduced redundancy)")
    logger.info(f"State-actions file saved as: {out_actions}")
    
    # Stratified canonical split by move_number and perfect outcome
    logger.info("Generating stratified canonical split assignments...")
    # Assign each canonical_form to a bucket key
    # Use (move_number, winner_perfect) of the first occurrence
    canon_groups = {}
    for _, row in df_states.iterrows():
        cf = row['canonical_form']
        if cf not in canon_groups:
            canon_groups[cf] = (int(row['move_number']), int(row['winner_perfect']) if not pd.isna(row['winner_perfect']) else 0)
    # Group canonical forms by bucket
    bucket_to_canon = {}
    for cf, key in canon_groups.items():
        bucket_to_canon.setdefault(key, []).append(cf)
    # Split within each bucket
    rng = np.random.default_rng(seed)
    train_canonical, val_canonical, test_canonical = set(), set(), set()
    for key, cfs in bucket_to_canon.items():
        cfs = list(set(cfs))
        rng.shuffle(cfs)
        n = len(cfs)
        t1 = int(0.7 * n)
        t2 = int(0.85 * n)
        train_canonical.update(cfs[:t1])
        val_canonical.update(cfs[t1:t2])
        test_canonical.update(cfs[t2:])
    
    # Add split column to dataframes
    df_states['split'] = df_states['canonical_form'].apply(
        lambda x: 'train' if x in train_canonical else ('val' if x in val_canonical else 'test')
    )
    df_actions['split'] = df_actions['canonical_form'].apply(
        lambda x: 'train' if x in train_canonical else ('val' if x in val_canonical else 'test')
    )
    
    # Save with splits
    if export_parquet:
        df_states.to_parquet('data_raw/ttt_states_splits.parquet', index=False)
        df_actions.to_parquet('data_raw/ttt_state_actions_splits.parquet', index=False)
    else:
        df_states.to_csv('data_raw/ttt_states_splits.csv', index=False)
        df_actions.to_csv('data_raw/ttt_state_actions_splits.csv', index=False)

    # Persist canonical split membership for reproducibility
    try:
        split_sets = {
            'train_canonical': sorted(list(train_canonical)),
            'val_canonical': sorted(list(val_canonical)),
            'test_canonical': sorted(list(test_canonical)),
        }
        with open('data_raw/canonical_splits.json', 'w') as f:
            json.dump(split_sets, f, indent=2)
        logger.info('Canonical split sets saved to data_raw/canonical_splits.json')
    except Exception as e:
        logger.warning(f'Failed to save canonical split sets: {e}')
    
    # Optional NPZ export (numeric-only, with optional plane tensors)
    if export_npz:
        def to_npz(df: Any, out_path: str):
            # Numeric-only view
            num_df = df.select_dtypes(include=['number', 'bool']).copy()
            # Standardize to float32 for compactness
            for col in num_df.columns:
                if num_df[col].dtype == bool:
                    num_df[col] = num_df[col].astype('float32')
            X = num_df.to_numpy(dtype='float32', copy=False)
            cols = np.array(num_df.columns.to_list(), dtype=object)
            # Split index mapping if available
            split_idx = None
            if 'split' in df.columns:
                mapping = {'train': 0, 'val': 1, 'test': 2}
                split_idx = df['split'].map(mapping).fillna(-1).to_numpy(dtype=np.int16)
            # Optional 4x3x3 planes if present
            def stack_planes(prefix: str):
                keys = [f"{prefix}_{i}" for i in range(9)]
                if all(k in df.columns for k in keys):
                    arr = df[keys].to_numpy(dtype='float32')
                    return arr.reshape((-1, 3, 3))
                return None
            x_plane = stack_planes('x_plane')
            o_plane = stack_planes('o_plane')
            empty_plane = stack_planes('empty_plane')
            to_move_plane = stack_planes('to_move_plane')
            legal_plane = stack_planes('legal_plane')
            best_plane = stack_planes('best_plane')
            planes = None
            if x_plane is not None and o_plane is not None and empty_plane is not None and to_move_plane is not None:
                planes_list = [x_plane, o_plane, empty_plane, to_move_plane]
                if legal_plane is not None and best_plane is not None:
                    planes_list.extend([legal_plane, best_plane])
                planes = np.stack(planes_list, axis=1)  # (N,C,3,3)
            if planes is not None:
                np.savez_compressed(out_path, data=X, columns=cols, planes=planes, split_index=split_idx)
            else:
                np.savez_compressed(out_path, data=X, columns=cols, split_index=split_idx)

        to_npz(df_states, 'data_raw/ttt_states.npz')
        to_npz(df_actions, 'data_raw/ttt_state_actions.npz')
    logger.info("NPZ files saved: data_raw/ttt_states.npz, data_raw/ttt_state_actions.npz")

    # Save metadata for reproducibility
    try:
        import pyarrow as _pa  # type: ignore
        _pa_ver = getattr(_pa, '__version__', None)
    except Exception:
        _pa_ver = None
    meta = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'args': {
            'include_augmentation': include_augmentation,
            'canonical_only': canonical_only,
            'reachable_only': reachable_only,
            'valid_only': valid_only,
            'export_parquet': export_parquet,
            'export_npz': export_npz,
            'lambda_temp': lambda_temp,
            'q_temp': q_temp,
            'states_canonical_only': states_canonical_only,
            'epsilons': epsilons,
            'seed': seed,
        },
        'versions': {
            'numpy': getattr(np, '__version__', None),
            'pandas': getattr(pd, '__version__', None) if pd is not None else None,
            'pyarrow': _pa_ver,
        },
        'counts': {
            'states': int(len(df_states)),
            'actions': int(len(df_actions)),
            'train_canonical': int(len(train_canonical)),
            'val_canonical': int(len(val_canonical)),
            'test_canonical': int(len(test_canonical)),
        },
        'outputs': {
            'states_path': out_states,
            'actions_path': out_actions,
            'states_npz': 'data_raw/ttt_states.npz' if export_npz else None,
            'actions_npz': 'data_raw/ttt_state_actions.npz' if export_npz else None,
        },
    }
    try:
        with open('data_raw/metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info('Metadata saved to data_raw/metadata.json')
    except Exception as e:
        logger.warning(f'Failed to save metadata: {e}')
    
    logger.info(f"Train canonical forms: {len(train_canonical)}")
    logger.info(f"Val canonical forms: {len(val_canonical)}")
    logger.info(f"Test canonical forms: {len(test_canonical)}")
    logger.info("Split assignments saved (prevents leakage across symmetric variants)")
    
    logger.info("Dataset generation complete!")
    
    # Summary statistics
    logger.info("=== Dataset Summary ===")
    logger.info("State distribution by outcome (perfect play):")
    outcome_dist = df_states[df_states['reachable_from_start']]['winner_perfect'].value_counts()
    logger.info(f"\n{outcome_dist}")
    
    logger.info("State distribution by game phase:")
    phase_dist = df_states['game_phase'].value_counts()
    logger.info(f"\n{phase_dist}")
    
    logger.info("Action tactical features (reachable states):")
    if len(df_actions) > 0:
        tactical_cols = ['is_immediate_win', 'is_immediate_block', 'creates_fork', 
                        'blocks_opponent_fork', 'gives_opponent_immediate_win']
        for col in tactical_cols:
            if col in df_actions.columns:
                count = df_actions[col].sum()
                pct = 100 * count / len(df_actions)
                logger.info(f"  {col}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Tic-Tac-Toe datasets with exact labels and rich features")
    parser.add_argument('--no-augmentation', action='store_true', help='Disable symmetry augmentation for actions')
    parser.add_argument('--no-canonical-only', action='store_true', help='Include all symmetry variants (not just canonical)')
    parser.add_argument('--all-states', action='store_true', help='Include unreachable/invalid states (override reachable/valid filters)')
    parser.add_argument('--csv', action='store_true', help='Export CSV instead of Parquet')
    parser.add_argument('--no-npz', action='store_true', help='Disable NPZ export')
    parser.add_argument('--lambda-temp', type=float, default=0.5, help='Lambda temperature for soft DTT policy')
    parser.add_argument('--q-temp', type=float, default=1.0, help='Temperature for Q softmax policy')
    parser.add_argument('--epsilons', type=str, default='0.1', help='Comma-separated epsilon values, e.g. "0.05,0.1,0.2"')
    parser.add_argument('--states-canonical-only', action='store_true', help='Deduplicate states by canonical form')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting and trajectories')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='[%(levelname)s] %(message)s')

    include_augmentation = not args.no_augmentation
    canonical_only = not args.no_canonical_only
    reachable_only = not args.all_states
    valid_only = not args.all_states
    export_parquet = not args.csv
    export_npz = not args.no_npz
    eps_vals = [float(x) for x in args.epsilons.split(',') if x.strip()]

    generate_tic_tac_toe_dataset(
        include_augmentation=include_augmentation,
        canonical_only=canonical_only,
        reachable_only=reachable_only,
        valid_only=valid_only,
        export_parquet=export_parquet,
        export_npz=export_npz,
        lambda_temp=args.lambda_temp,
        q_temp=args.q_temp,
        states_canonical_only=args.states_canonical_only,
        epsilons=eps_vals,
        seed=args.seed,
    )

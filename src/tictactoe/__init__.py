"""tictactoe package.

Core algorithms, symmetry handling, datasets, and a simple CLI.

Convenience imports are exposed for common workflows.
"""

from .datasets import ExportArgs, run_export
from .orchestrator import extract_board_features, generate_state_action_dataset
from .solver import solve_all_reachable, solve_state

__all__ = [
    "solve_all_reachable",
    "solve_state",
    "extract_board_features",
    "generate_state_action_dataset",
    "run_export",
    "ExportArgs",
]

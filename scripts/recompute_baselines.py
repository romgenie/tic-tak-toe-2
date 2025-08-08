#!/usr/bin/env python3
"""
Recompute baseline counts for Tic-tac-toe using the packaged solver.

Outputs JSON to tests/data/baselines.json with keys:
  - reachable
  - nonterminal
  - terminal: {x, o, draw}

This is the single source of truth for baseline counts; do not hand-edit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from tictactoe.solver import solve_all_reachable
from tictactoe.game_basics import get_winner, is_draw


def main() -> int:
    solved: Dict[str, dict] = solve_all_reachable()
    reachable = len(solved)
    term = {"x": 0, "o": 0, "draw": 0}
    for key in solved.keys():
        b = [int(c) for c in key]
        w = get_winner(b)
        if w == 1:
            term["x"] += 1
        elif w == 2:
            term["o"] += 1
        elif is_draw(b):
            term["draw"] += 1
    nonterminal = reachable - sum(term.values())
    out = {"reachable": reachable, "nonterminal": nonterminal, "terminal": term}
    target = Path(__file__).resolve().parents[1] / "tests" / "data" / "baselines.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

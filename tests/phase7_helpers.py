"""Shared helpers for the Phase 7 test suites.

Not a conftest.py on purpose: the repo already has tests, and a second
conftest could clash with an existing one. pytest puts this folder on
sys.path (default "prepend" import mode), so test files can simply
`import phase7_helpers`.
"""
import os
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# Stockfish_train_eval.py reads params.yaml at import time
os.environ.setdefault("PARAMS_FILE", str(REPO / "params.yaml"))

BLOCK_END = "\n\n\n"

# A 19-ply opening (the Ruy Lopez main line), used by the drift scenario
OPENING_19 = ("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 "
              "7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4")


def evaluated_block(white_elo, black_elo, evals, result="1-0", movetext=None):
    """One game in the format Stockfish_train_eval.py writes (and prepare.py reads)."""
    evals_s = ",".join(str(e) for e in evals)
    movetext = movetext or f"1. e4 e5 {result}"
    return (f'[WhiteElo "{white_elo}"]\n[BlackElo "{black_elo}"]\n'
            f'[Result "{result}"]\n[Evals "{evals_s}"]\n\n{movetext}{BLOCK_END}')


def oscillating_evals(n, seed=0, scale=120):
    """Deterministic, realistic-looking White-perspective centipawn evals."""
    rng = random.Random(seed)
    cp, out = 20, []
    for _ in range(n):
        cp += rng.randint(-scale, scale)
        out.append(int(max(-3000, min(3000, cp))))
    return out


def random_legal_pgn(plies, seed=0, headers=None, result="*"):
    """A legal game of `plies` half-moves (fewer if it ends early), as PGN text."""
    import chess
    import chess.pgn

    rng = random.Random(seed)
    board = chess.Board()
    game = chess.pgn.Game()
    for k, v in (headers or {}).items():
        game.headers[k] = v
    node = game
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        mv = rng.choice(moves)
        board.push(mv)
        node = node.add_variation(mv)
    game.headers["Result"] = result
    return str(game)


def require(module_name):
    """Skip the calling test module if an optional dependency is missing."""
    return pytest.importorskip(module_name)

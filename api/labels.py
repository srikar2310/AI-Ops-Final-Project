"""Per-move quality labels for the results screen (eval graph, blunder list).

Uses the thresholds and the win-probability formula from features.py, so the
labels shown to the user are computed exactly like the training features.
features.py lives in the project root (run uvicorn from there, and COPY it
into the API Docker image).
"""
from typing import Dict, List

import numpy as np

from features import BLUNDER, INACCURACY, MISTAKE, cp_to_wp

LABELS = ("ok", "inaccuracy", "mistake", "blunder")


def label_for(loss):
    """Label for one move from the drop in the mover's win probability (0..1)."""
    if loss >= BLUNDER:
        return "blunder"
    if loss >= MISTAKE:
        return "mistake"
    if loss >= INACCURACY:
        return "inaccuracy"
    return "ok"


def annotate_moves(evals, moves_san, clip):
    """One dict per ply: side, san, eval, White win chance, loss and label.

    `loss` is computed like features.game_features: the mover's win-probability
    drop, floored at 0. Ply 0 is compared with an even position (0.5).
    """
    n = min(len(evals), len(moves_san))
    if n == 0:
        return []
    w = cp_to_wp(np.asarray(evals[:n]), clip)           # White win probability
    t = np.arange(n)
    prev = np.concatenate(([0.5], w[:-1]))
    loss = np.clip(np.where(t % 2 == 0, prev - w, w - prev), 0.0, 1.0)
    return [{
        "ply": i + 1,
        "side": "white" if i % 2 == 0 else "black",
        "san": moves_san[i],
        "eval_cp": int(evals[i]),
        "white_win_prob": round(float(w[i]), 4),
        "loss": round(float(loss[i]), 4),
        "label": label_for(float(loss[i])),
    } for i in range(n)]


def summarise(moves) -> Dict[str, Dict[str, int]]:
    """Counts of inaccuracies / mistakes / blunders per side."""
    out = {s: {k: 0 for k in LABELS[1:]} for s in ("white", "black")}
    for m in moves:
        if m["label"] != "ok":
            out[m["side"]][m["label"]] += 1
    return out

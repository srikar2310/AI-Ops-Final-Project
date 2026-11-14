"""Shared eval -> model-input conversion.

Used by prepare.py (training data) AND pyfunc_model.py (serving), so the two
can never disagree about how an eval sequence becomes network input.

Two kinds of input are produced for each (game, player) pair:
  * a sequence  : win probability per ply from the player's view (build_sample)
  * a feature vector of game-level statistics (game_features)
"""
import numpy as np

LICHESS_K = 0.00368208  # centipawns -> win probability (Lichess formula)

# Lichess-style thresholds on the drop in win probability caused by one move
INACCURACY, MISTAKE, BLUNDER, PERFECT = 0.10, 0.20, 0.30, 0.01

# Phases of the game, in plies (0-based): opening / middlegame / endgame
PHASES = ((0, 20), (20, 60), (60, 10 ** 9))

_LOSS_STATS = ["loss_mean", "loss_std", "loss_median",
               "inaccuracy_rate", "mistake_rate", "blunder_rate", "perfect_rate"]

FEATURE_NAMES = (
    [f"own_{n}" for n in _LOSS_STATS]            # the player's own move quality
    + [f"opp_{n}" for n in _LOSS_STATS]          # the opponent's move quality
    + ["own_loss_opening", "own_loss_middle", "own_loss_end"]
    + ["max_wp", "min_wp", "final_wp", "frac_winning", "frac_losing",
       "blown_win", "saved_loss"]
    + ["result_score", "result_known"]
    + ["game_length"]
)
N_FEATURES = len(FEATURE_NAMES)  # 27


def cp_to_wp(cp, clip):
    """White-perspective centipawns -> White win probability in (0, 1)."""
    cp = np.clip(np.asarray(cp, dtype=np.float32), -clip, clip)
    return 1.0 / (1.0 + np.exp(-LICHESS_K * cp))


def build_sample(cp, side, max_plies, clip):
    """Turn one game's evals into the stored sequence row for one player.

    cp:   White-perspective centipawn evals, one per ply
    side: 0 if the player is White, 1 if Black
    Returns (row, length): row is float32 [max_plies] holding the win
    probability from the player's point of view (padded with 0.5), and
    length is the number of real plies (after truncation).
    """
    cp = np.asarray(cp)[:max_plies]
    n = len(cp)
    wp = cp_to_wp(cp, clip)
    if side == 1:
        wp = 1.0 - wp
    row = np.full(max_plies, 0.5, dtype=np.float32)
    row[:n] = wp
    return row, n


def _loss_stats(loss):
    """Summary of per-move win-probability losses (7 numbers)."""
    if len(loss) == 0:
        return [0.0] * 7
    return [float(loss.mean()), float(loss.std()), float(np.median(loss)),
            float((loss >= INACCURACY).mean()), float((loss >= MISTAKE).mean()),
            float((loss >= BLUNDER).mean()), float((loss < PERFECT).mean())]


def _result_features(result, side):
    """(score from the player's view, known flag). Unknown result -> (0.5, 0)."""
    r = str(result).strip()
    if r == "1-0":
        return (1.0 if side == 0 else 0.0), 1.0
    if r == "0-1":
        return (0.0 if side == 0 else 1.0), 1.0
    if r in ("1/2-1/2", "1/2"):
        return 0.5, 1.0
    return 0.5, 0.0


def game_features(cp, side, result, clip):
    """Game-level feature vector (length N_FEATURES, float32) for one player.

    Uses the FULL eval sequence (not truncated to max_plies).
    cp:     White-perspective centipawn evals, one per ply
    side:   0 if the player is White, 1 if Black
    result: PGN result string ("1-0", "0-1", "1/2-1/2", or "*" if unknown)
    """
    cp = np.asarray(cp)
    n = len(cp)
    w = cp_to_wp(cp, clip)                              # White win probability
    t = np.arange(n)

    # Loss of the side that MOVED at each ply, as a drop in its win probability.
    prev = np.concatenate(([0.5], w[:-1]))
    loss = np.clip(np.where(t % 2 == 0, prev - w, w - prev), 0.0, 1.0)
    own_mask = (t % 2) == side
    own, opp = loss[own_mask], loss[~own_mask]

    own_stats = _loss_stats(own)
    phase_loss = []
    for lo, hi in PHASES:
        sel = own_mask & (t >= lo) & (t < hi)
        # an empty phase (short game) falls back to the player's overall mean
        phase_loss.append(float(loss[sel].mean()) if sel.any() else own_stats[0])

    wp = w if side == 0 else 1.0 - w                    # player's view
    final = float(wp[-1])
    conversion = [
        float(wp.max()), float(wp.min()), final,
        float((wp >= 0.7).mean()), float((wp <= 0.3).mean()),
        float(wp.max() >= 0.8 and final < 0.5),         # threw away a winning game
        float(wp.min() <= 0.2 and final > 0.5),         # rescued a lost game
    ]

    score, known = _result_features(result, side)
    feats = (own_stats + _loss_stats(opp) + phase_loss + conversion
             + [score, known] + [n / 100.0])
    return np.asarray(feats, dtype=np.float32)
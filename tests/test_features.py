"""Unit tests for the shared feature code (run with: pytest tests/)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import FEATURE_NAMES, N_FEATURES, build_sample, cp_to_wp, game_features  # noqa: E402

IX = {n: i for i, n in enumerate(FEATURE_NAMES)}
# White blunders at ply index 2 (+20 -> -600) and goes on to lose.
GAME = [20, 20, -600, -620, -650, -700, -900, -1000, -1200, -1500, -3000, -10000]


def test_cp_to_wp_is_symmetric_and_bounded():
    assert abs(cp_to_wp([0], 1500)[0] - 0.5) < 1e-6
    assert abs(cp_to_wp([300], 1500)[0] + cp_to_wp([-300], 1500)[0] - 1) < 1e-6
    assert 0 < cp_to_wp([10000], 1500)[0] < 1  # clipped, never saturates to 1


def test_build_sample_pads_truncates_and_flips_for_black():
    row_w, n = build_sample(GAME, 0, 20, 1500)
    row_b, _ = build_sample(GAME, 1, 20, 1500)
    assert n == len(GAME) and row_w.shape == (20,)
    assert np.allclose(row_w[:n] + row_b[:n], 1.0)       # Black's view is the mirror
    assert np.all(row_w[n:] == 0.5)                      # padding
    assert build_sample(GAME, 0, 5, 1500)[1] == 5        # truncation


def test_feature_vector_shape_and_finite():
    f = game_features(GAME, 0, "0-1", 1500)
    assert f.shape == (N_FEATURES,) and f.dtype == np.float32
    assert np.isfinite(f).all()


def test_blunder_is_attributed_to_the_right_player():
    white, black = game_features(GAME, 0, "0-1", 1500), game_features(GAME, 1, "0-1", 1500)
    assert white[IX["own_blunder_rate"]] > 0 and black[IX["own_blunder_rate"]] == 0
    assert white[IX["opp_blunder_rate"]] == 0 and black[IX["opp_blunder_rate"]] > 0


def test_result_is_from_the_players_point_of_view():
    w, b = game_features(GAME, 0, "0-1", 1500), game_features(GAME, 1, "0-1", 1500)
    assert w[IX["result_score"]] == 0 and b[IX["result_score"]] == 1
    unknown = game_features(GAME, 0, "*", 1500)
    assert unknown[IX["result_known"]] == 0 and unknown[IX["result_score"]] == 0.5


def test_tiny_games_do_not_produce_nans():
    for n in (1, 2, 5):
        for side in (0, 1):
            assert np.isfinite(game_features(np.arange(n) * 10, side, "1-0", 1500)).all()
"""Unit tests for features.py: the eval -> model-input conversion shared by
training (prepare.py) and serving (pyfunc_model.py).

Case IDs (UT-F-xx) match docs/test_cases.md.
"""
import numpy as np
import pytest

import phase7_helpers  # noqa: F401  (puts the repo root on sys.path)
from features import (BLUNDER, FEATURE_NAMES, INACCURACY, MISTAKE, N_FEATURES,
                      build_sample, cp_to_wp, game_features)

CLIP = 1500.0
IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def feat(cp, side=0, result="1-0"):
    return dict(zip(FEATURE_NAMES, game_features(cp, side, result, CLIP)))


# ------------------------------------------------------------------ cp_to_wp
def test_uf01_equal_position_is_half():
    assert cp_to_wp(0, CLIP) == pytest.approx(0.5)


def test_uf02_known_value_lichess_formula():
    # 1 / (1 + exp(-0.00368208 * 100))
    assert cp_to_wp(100, CLIP) == pytest.approx(0.59103, abs=1e-4)


def test_uf03_symmetric_and_monotonic():
    cp = np.arange(-1400, 1401, 50)
    wp = cp_to_wp(cp, CLIP)
    np.testing.assert_allclose(wp + cp_to_wp(-cp, CLIP), 1.0, atol=1e-6)
    assert np.all(np.diff(wp) > 0)
    assert np.all((wp > 0) & (wp < 1))


def test_uf04_clipping():
    assert cp_to_wp(10000, CLIP) == pytest.approx(cp_to_wp(1500, CLIP))
    assert cp_to_wp(-10000, CLIP) == pytest.approx(cp_to_wp(-1500, CLIP))


# -------------------------------------------------------------- build_sample
def test_uf05_padding_and_length():
    row, n = build_sample([0, 100, -100], side=0, max_plies=8, clip=CLIP)
    assert row.dtype == np.float32 and row.shape == (8,)
    assert n == 3
    np.testing.assert_allclose(row[3:], 0.5)
    assert row[1] == pytest.approx(cp_to_wp(100, CLIP))


def test_uf06_black_view_is_complement():
    cp = [30, -50, 200, -400]
    w, _ = build_sample(cp, 0, 6, CLIP)
    b, _ = build_sample(cp, 1, 6, CLIP)
    np.testing.assert_allclose(w[:4] + b[:4], 1.0, atol=1e-6)
    np.testing.assert_allclose(b[4:], 0.5)          # padding stays neutral


def test_uf07_truncation():
    row, n = build_sample(list(range(300)), 0, max_plies=200, clip=CLIP)
    assert n == 200 and row.shape == (200,)


# ------------------------------------------------------------- game_features
def test_uf08_feature_contract():
    assert N_FEATURES == 27 == len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    f = game_features(phase7_helpers.oscillating_evals(60), 0, "1-0", CLIP)
    assert f.shape == (27,) and f.dtype == np.float32
    assert np.all(np.isfinite(f))


def test_uf09_perfect_game_has_no_errors():
    f = feat([0] * 40)
    for side in ("own", "opp"):
        assert f[f"{side}_loss_mean"] == 0
        assert f[f"{side}_blunder_rate"] == 0
        assert f[f"{side}_perfect_rate"] == 1
    assert f["game_length"] == pytest.approx(0.40)


def test_uf10_blunder_is_attributed_to_the_right_player():
    # ply index 2 is a White move: White drops from equal to -1000 cp
    cp = [0, 0, -1000] + [-1000] * 17
    white, black = feat(cp, side=0), feat(cp, side=1)
    assert white["own_blunder_rate"] == pytest.approx(1 / 10)   # 10 White moves
    assert white["opp_blunder_rate"] == 0
    assert black["opp_blunder_rate"] == pytest.approx(1 / 10)
    assert black["own_blunder_rate"] == 0


def test_uf11_threshold_ordering():
    assert 0 < INACCURACY < MISTAKE < BLUNDER < 1
    # every blunder is also a mistake and an inaccuracy
    f = feat(phase7_helpers.oscillating_evals(80, seed=3, scale=400))
    assert f["own_inaccuracy_rate"] >= f["own_mistake_rate"] >= f["own_blunder_rate"]


def test_uf12_first_ply_gain_is_not_a_loss():
    f = feat([300] + [300] * 19)            # White's first move gains
    assert f["own_loss_mean"] == 0


@pytest.mark.parametrize("result,side,score,known", [
    ("1-0", 0, 1.0, 1.0), ("1-0", 1, 0.0, 1.0),
    ("0-1", 0, 0.0, 1.0), ("0-1", 1, 1.0, 1.0),
    ("1/2-1/2", 0, 0.5, 1.0), ("1/2", 1, 0.5, 1.0),
    ("*", 0, 0.5, 0.0), ("", 1, 0.5, 0.0), (None, 0, 0.5, 0.0),
])
def test_uf13_result_features(result, side, score, known):
    f = feat([0] * 20, side=side, result=result)
    assert f["result_score"] == score and f["result_known"] == known


def test_uf14_short_game_phase_fallback():
    f = feat(phase7_helpers.oscillating_evals(12, seed=1))
    # no middlegame / endgame plies -> both fall back to the overall mean
    assert f["own_loss_middle"] == pytest.approx(f["own_loss_mean"])
    assert f["own_loss_end"] == pytest.approx(f["own_loss_mean"])


def test_uf15_blown_win_and_saved_loss():
    cp = [0] * 10 + [1000] * 10 + [-1000] * 10   # White was winning, then lost
    white, black = feat(cp, 0, "0-1"), feat(cp, 1, "0-1")
    assert white["blown_win"] == 1 and white["saved_loss"] == 0
    assert black["saved_loss"] == 1 and black["blown_win"] == 0
    assert white["final_wp"] == pytest.approx(1 - black["final_wp"], abs=1e-6)


def test_uf16_features_use_full_sequence_not_truncated():
    cp = [0] * 250
    assert feat(cp)["game_length"] == pytest.approx(2.5)


def test_uf17_empty_sequence_is_rejected():
    # serving guards against this before calling (see test_pyfunc_unit.py)
    with pytest.raises(Exception):
        game_features([], 0, "1-0", CLIP)


def test_uf18_mate_scores_are_clipped_not_overflowing():
    f = game_features([10000, -10000] * 10, 0, "*", CLIP)
    assert np.all(np.isfinite(f))

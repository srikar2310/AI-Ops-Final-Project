"""Unit tests for api/drift.py (run from the repo root: pytest tests/test_drift.py)."""
import numpy as np
import pytest

from api.drift import FEATURE_DRIFT_MAX, WINDOW_SIZE, DriftMonitor
from features import FEATURE_NAMES, N_FEATURES, game_features

CLIP = 1500.0
RNG = np.random.default_rng(0)


def random_game(n=60, scale=60):
    return np.cumsum(RNG.normal(0, scale, n)).astype(int).tolist()


def make_monitor(**kw):
    games = [random_game() for _ in range(300)]
    feats = np.stack([game_features(np.asarray(g), i % 2, "1-0", CLIP) for i, g in enumerate(games)])
    stats = {"feature_names": FEATURE_NAMES, "feat_mean": feats.mean(0).tolist(),
             "feat_std": feats.std(0).tolist(), "rating_mean": 1800.0, "rating_std": 300.0}
    return DriftMonitor(stats, CLIP, **kw)


def test_no_report_before_min_samples():
    m = make_monitor(min_samples=30)
    for _ in range(5):
        m.observe(random_game(), "white", "1-0", 1800)
    assert WINDOW_SIZE._value.get() == 5


def test_similar_games_give_small_shift():
    m = make_monitor(min_samples=30)
    for i in range(150):
        m.observe(random_game(), "white" if i % 2 else "black", "1-0", 1800)
    assert FEATURE_DRIFT_MAX._value.get() < 1.0


def test_very_different_games_raise_the_alarm():
    m = make_monitor(min_samples=30)
    for _ in range(100):                      # wild games: huge swings, unknown result
        m.observe(random_game(scale=400), "white", "*", 1800)
    assert FEATURE_DRIFT_MAX._value.get() > 1.0


def test_mismatched_feature_names_are_rejected():
    with pytest.raises(ValueError):
        DriftMonitor({"feature_names": ["x"], "feat_mean": [0], "feat_std": [1],
                      "rating_mean": 0, "rating_std": 1}, CLIP)


def test_bad_input_does_not_raise():
    m = make_monitor()
    m.observe([], "white", "1-0", 1800)       # empty evals: logged, not raised
    assert N_FEATURES == len(FEATURE_NAMES)
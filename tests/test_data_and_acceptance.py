"""Checks on the REAL pipeline outputs (marker: data) and the serving-correctness
acceptance criteria of the registered model.

Model accuracy (MAE, per-band error) is deliberately NOT tested here: it is
reported as a description in docs/HLD.md, not as a pass/fail criterion.

Run after `dvc repro` and `python register_model.py`. Every test skips if the
file it needs is not there, so this suite is safe to run anywhere.

The thresholds below are the acceptance criteria in docs/test_plan.md. Keep
the two in sync.
Case IDs (DQ-xx, AC-xx) match docs/test_cases.md.
"""
import json

import numpy as np
import pytest

import phase7_helpers as h
from features import FEATURE_NAMES, N_FEATURES

pytestmark = pytest.mark.data

# ----------------------------------------------------- acceptance criteria
AC_MAX_PARITY_GAP = 1.0           # AC-1  serving MAE vs MAE logged at training

PROCESSED = h.REPO / "data" / "processed"
METRICS = h.REPO / "metrics"
REPORT = h.REPO / "reports" / "model_test_report.json"


def load_json(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(h.REPO)} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(split):
    path = PROCESSED / f"{split}.npz"
    if not path.exists():
        pytest.skip(f"{path.relative_to(h.REPO)} not found")
    return np.load(path)


@pytest.fixture(scope="module")
def params():
    import yaml
    with open(h.REPO / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------- data quality
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dq01_processed_arrays_are_consistent(split, params):
    z = load_npz(split)
    n, L = z["wp"].shape
    assert L == params["data"]["max_plies"]
    for key, shape in (("length", (n,)), ("side", (n,)), ("rating", (n,)),
                       ("feat", (n, N_FEATURES))):
        assert z[key].shape == shape, key
    assert n > 0 and n % 2 == 0                               # two samples per game
    assert set(np.unique(z["side"]).tolist()) <= {0, 1}
    assert np.all(z["side"][0::2] == 0) and np.all(z["side"][1::2] == 1)
    assert z["length"].min() >= params["data"]["min_plies"]
    assert z["length"].max() <= L
    assert np.all((z["wp"] >= 0) & (z["wp"] <= 1))
    assert np.all(np.isfinite(z["feat"]))
    assert 300 <= z["rating"].min() and z["rating"].max() <= 3300


def test_dq02_stats_match_feature_contract():
    stats = load_json(PROCESSED / "stats.json")
    assert stats["feature_names"] == FEATURE_NAMES
    assert len(stats["feat_mean"]) == len(stats["feat_std"]) == N_FEATURES
    assert 1000 < stats["rating_mean"] < 2500 and stats["rating_std"] > 50


def test_dq03_stats_come_from_train_only():
    stats = load_json(PROCESSED / "stats.json")
    z = load_npz("train")
    assert stats["rating_mean"] == pytest.approx(float(z["rating"].mean()), rel=1e-5)


def test_dq04_no_game_leaks_between_train_and_test():
    """Identical eval sequences in train and test would mean the same game is in both."""
    def keys(split):
        z = load_npz(split)
        wp = z["wp"][0::2]                                        # one row per game
        return {row.tobytes() for row, n in zip(wp, z["length"][0::2]) if n >= 30}
    overlap = keys("train") & keys("test")
    assert len(overlap) == 0, f"{len(overlap)} games of 30+ plies appear in both splits"


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dq05_evaluation_stage_had_no_failures(split):
    meta = load_json(METRICS / f"eval_{split}.json")
    assert meta["failed_games_last_run"] == 0
    assert meta["games"] > 0 and meta["positions_per_sec"] > 0


def test_dq06_prepare_stage_metrics():
    m = load_json(METRICS / "prepare.json")
    for split in ("train", "val", "test"):
        s = m[split]
        assert s["games_used"] > 0 and s["samples"] == 2 * s["games_used"]
        # most games should survive cleaning
        assert s["skipped_bad"] <= 0.01 * s["games_seen"]


# ------------------------------------------------------- acceptance criteria
@pytest.fixture(scope="module")
def report():
    return load_json(REPORT)


def test_ac01_serving_parity(report):
    """Training and serving compute the same thing (no train/serve skew)."""
    if report.get("logged_test_mae") is None:
        pytest.skip("no logged test MAE in the report")
    assert abs(report["retested"]["mae"] - report["logged_test_mae"]) <= AC_MAX_PARITY_GAP


def test_ac02_robustness_checks(report):
    """Unknown result handled; bad side and empty evals rejected; output in 0-4000."""
    failed = [name for name, ok in report["checks"].items() if not ok]
    assert not failed, f"failed checks: {failed}"

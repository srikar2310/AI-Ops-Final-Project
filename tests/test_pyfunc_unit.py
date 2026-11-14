"""Tests for pyfunc_model.RatingPyfunc, the code the MLflow model server runs.

UT-S-xx run the wrapper directly with a tiny model (no MLflow server).
IT-S-01 saves and reloads it through MLflow (marker: slow).
Case IDs match docs/test_cases.md.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import phase7_helpers as h

torch = pytest.importorskip("torch")
pytest.importorskip("mlflow")

from features import build_sample, game_features  # noqa: E402
from model import build_model  # noqa: E402
from pyfunc_model import RatingPyfunc, _side_to_int  # noqa: E402

CFG = {
    "model": {"channels": [8, 8], "kernel": 3, "dilations": [1, 2], "dropout": 0.0,
              "hidden": 8, "use_features": True, "feat_hidden": 8},
    "rating_mean": 1800.0, "rating_std": 300.0, "max_plies": 64,
    "data": {"cp_clip": 1500, "max_plies": 64, "min_plies": 10},
}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("pyfunc")
    torch.manual_seed(0)
    m = build_model(CFG["model"])
    torch.save(m.state_dict(), d / "best.pt")
    (d / "config.json").write_text(json.dumps(CFG))
    return {"weights": str(d / "best.pt"), "config": str(d / "config.json")}


@pytest.fixture(scope="module")
def wrapper(artifacts):
    w = RatingPyfunc()
    w.load_context(SimpleNamespace(artifacts=artifacts))
    return w


def records(n=3):
    out = []
    for g in range(n):
        ev = h.oscillating_evals(20 + 10 * g, seed=g)
        out.append({"evals": ev, "side": "white", "result": "1-0"})
        out.append({"evals": ev, "side": "black", "result": "1-0"})
    return out


@pytest.mark.parametrize("raw,expected", [
    ("white", 0), ("W", 0), (" White ", 0), ("0", 0), (0, 0),
    ("black", 1), ("b", 1), ("BLACK", 1), ("1", 1), (1, 1),
])
def test_us01_side_aliases(raw, expected):
    assert _side_to_int(raw) == expected


@pytest.mark.parametrize("bad", ["green", "", None, 2])
def test_us02_invalid_side_rejected(bad):
    with pytest.raises(ValueError):
        _side_to_int(bad)


def test_us03_predict_output_contract(wrapper):
    out = wrapper.predict(None, pd.DataFrame(records()))
    assert list(out.columns) == ["predicted_rating"] and len(out) == 6
    vals = out["predicted_rating"].to_numpy()
    assert np.all(np.isfinite(vals))
    np.testing.assert_allclose(vals, np.round(vals, 1))            # rounded to 0.1


def test_us04_result_column_is_optional(wrapper):
    df = pd.DataFrame(records()).drop(columns=["result"])
    out = wrapper.predict(None, df)
    unknown = wrapper.predict(None, pd.DataFrame(records()).assign(result="*"))
    pd.testing.assert_frame_equal(out, unknown)


def test_us05_rejects_empty_evals(wrapper):
    with pytest.raises(ValueError, match="at least one ply"):
        wrapper.predict(None, pd.DataFrame([{"evals": [], "side": "white", "result": "*"}]))


def test_us06_rejects_invalid_side(wrapper):
    with pytest.raises(ValueError):
        wrapper.predict(None, pd.DataFrame([{"evals": [1, 2, 3], "side": "green"}]))


def test_us07_matches_direct_model_computation(wrapper):
    """Serving = features.py + model forward + de-standardisation, nothing else."""
    recs = records(2)
    out = wrapper.predict(None, pd.DataFrame(recs))["predicted_rating"].to_numpy()
    rows, sides, lengths, feats = [], [], [], []
    for r in recs:
        s = 0 if r["side"] == "white" else 1
        row, n = build_sample(r["evals"], s, CFG["max_plies"], 1500)
        rows.append(row), sides.append(s), lengths.append(n)
        feats.append(game_features(r["evals"], s, r["result"], 1500))
    with torch.no_grad():
        z = wrapper.model(torch.tensor(np.stack(rows)), torch.tensor(sides),
                          torch.tensor(lengths), torch.tensor(np.stack(feats)))
    np.testing.assert_allclose(out, np.round(z.numpy() * 300.0 + 1800.0, 1), atol=0.05)


def test_us08_deterministic_and_order_preserving(wrapper):
    recs = records()
    a = wrapper.predict(None, pd.DataFrame(recs))["predicted_rating"].tolist()
    b = wrapper.predict(None, pd.DataFrame(recs[::-1]))["predicted_rating"].tolist()
    assert a == b[::-1]


def test_us09_long_games_are_accepted(wrapper):
    out = wrapper.predict(None, pd.DataFrame([{"evals": h.oscillating_evals(400),
                                               "side": "black", "result": "0-1"}]))
    assert np.isfinite(out["predicted_rating"].iloc[0])


@pytest.mark.slow
def test_is01_mlflow_save_and_load_roundtrip(artifacts, wrapper, tmp_path):
    """The same packaging train.py uses: save -> load -> predict gives identical output."""
    import mlflow.pyfunc
    path = tmp_path / "pyfunc_model"
    mlflow.pyfunc.save_model(
        path=str(path), python_model=RatingPyfunc(), artifacts=artifacts,
        code_paths=[str(h.REPO / f) for f in ("model.py", "features.py", "pyfunc_model.py")],
        pip_requirements=["torch", "numpy", "pandas", "mlflow"])
    loaded = mlflow.pyfunc.load_model(str(path))
    df = pd.DataFrame(records())
    pd.testing.assert_frame_equal(loaded.predict(df), wrapper.predict(None, df))

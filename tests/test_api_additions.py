"""Tests for /validate, /games/split, /config, per-move labels and the ops routes.

Run from the project root (so `api` and `features` import):
    python -m pytest tests/test_api_additions.py -v
Needs: pytest, httpx (FastAPI's TestClient). No Stockfish or MLflow required:
the engine pool and the model client are replaced by fakes.
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.engine import Analysis
from api.labels import annotate_moves, label_for, summarise
from api.main import create_app
from api.ops import build_pipeline
from api.settings import Settings

GAME = ("[White \"A\"]\n[Black \"B\"]\n[WhiteElo \"1500\"]\n[BlackElo \"1480\"]\n"
        "[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 "
        "6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0")          # 20 plies
SHORT = "[Result \"*\"]\n\n1. e4 e5 *"


class FakePool:
    engine_name = "FakeFish"

    def __init__(self, evals):
        self.evals = evals

    def ready(self):
        return True

    def analyse_game(self, moves, token):
        return Analysis(evals=self.evals[:len(moves)], seconds=0.1, positions=len(moves))


class FakeModel:
    def predict(self, records):
        return [1500.0] * len(records)

    def ready(self):
        return True

    def describe(self):
        return {"mode": "fake", "target": "none"}


def make_client(evals=None, **overrides):
    settings = Settings(engine_path="x", depth=11, hash_mb=64, mate_score=10000,
                        min_plies=10, **overrides)
    app = create_app(settings, FakePool(evals or [20] * 20), FakeModel())
    return TestClient(app)


# ---------------------------------------------------------------- /config
def test_config_returns_limits():
    with make_client() as c:
        body = c.get("/config").json()
    assert body["min_plies"] == 10 and body["engine_depth"] == 11
    assert body["max_pgn_bytes"] == 200_000


# -------------------------------------------------------------- /validate
def test_validate_ok():
    with make_client() as c:
        body = c.post("/validate", json={"pgn": GAME}).json()
    assert body["valid"] is True and body["plies"] == 20
    assert body["white_elo"] == 1500 and body["result"] == "1-0"


@pytest.mark.parametrize("text", ["", SHORT, "1. e4 e5 2. Ke4"])
def test_validate_bad_input_is_a_normal_answer(text):
    with make_client() as c:
        resp = c.post("/validate", json={"pgn": text})
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False and body["error"]["code"] == "bad_pgn"


# ----------------------------------------------------------- /games/split
def test_split_lists_games_and_flags_unusable_ones():
    with make_client() as c:
        body = c.post("/games/split", json={"text": GAME + "\n\n" + SHORT}).json()
    assert [g["usable"] for g in body["games"]] == [True, False]
    assert body["games"][1]["problem"]
    assert body["truncated"] is False
    # each item's PGN can go straight to /validate
    with make_client() as c:
        assert c.post("/validate", json={"pgn": body["games"][0]["pgn"]}).json()["valid"]


def test_split_respects_game_limit():
    with make_client(max_games_per_file=2) as c:
        body = c.post("/games/split", json={"text": (GAME + "\n\n") * 3}).json()
    assert len(body["games"]) == 2 and body["truncated"] is True


def test_split_empty_file_is_rejected():
    with make_client() as c:
        resp = c.post("/games/split", json={"text": "  "})
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "bad_pgn"


# ----------------------------------------------------------------- labels
def test_label_thresholds():
    assert label_for(0.05) == "ok"
    assert label_for(0.10) == "inaccuracy"
    assert label_for(0.20) == "mistake"
    assert label_for(0.30) == "blunder"


def test_annotate_marks_a_white_blunder():
    evals = [30] * 4 + [-600] * 16            # White's 5th ply throws the game away
    san = [f"m{i}" for i in range(20)]
    moves = annotate_moves(evals, san, 1500)
    assert moves[4]["label"] == "blunder" and moves[4]["side"] == "white"
    assert summarise(moves)["white"]["blunder"] == 1
    assert summarise(moves)["black"]["blunder"] == 0


def test_predict_response_has_moves():
    evals = [30] * 4 + [-600] * 16
    with make_client(evals) as c:
        body = c.post("/predict", json={"pgn": GAME}).json()
    assert len(body["moves"]) == 20 and body["moves"][0]["san"] == "e4"
    assert body["moves"][4]["label"] == "blunder"
    assert body["move_summary"]["white"]["blunder"] == 1


# -------------------------------------------------------------------- ops
def test_pipeline_reads_dvc_files(tmp_path):
    (tmp_path / "dvc.yaml").write_text("""
stages:
  evaluate:
    foreach:
      - train
    do:
      cmd: echo
      deps:
        - dataset_clean/${item}
      outs:
        - dataset_clean/${item}_evaluated
      metrics:
        - metrics/eval_${item}.json:
            cache: false
  prepare:
    cmd: echo
    deps:
      - dataset_clean/train_evaluated
    outs:
      - data/processed
""")
    (tmp_path / "dvc.lock").write_text("schema: '2.0'\nstages:\n  evaluate@train: {}\n")
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "eval_train.json").write_text(json.dumps(
        {"seconds": 10, "games": 5, "positions": 100, "games_per_sec": 0.5}))
    out = build_pipeline(tmp_path)
    by_name = {s["name"]: s for s in out["stages"]}
    assert ["evaluate@train", "prepare"] in out["edges"]
    assert by_name["evaluate@train"]["status"] == "has_run"
    assert by_name["prepare"]["status"] == "never_run"
    assert by_name["evaluate@train"]["summary"]["seconds"] == 10
    assert "evaluate_train --> prepare" in out["mermaid"]


def test_ops_service_and_links_respond():
    with make_client() as c:
        c.post("/predict", json={"pgn": GAME})
        svc = c.get("/ops/service").json()
        links = c.get("/ops/links").json()
    assert svc["engine_ready"] and svc["predictions"] >= 1
    assert "grafana" in links

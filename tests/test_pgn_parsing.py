"""PGN parsing tests for the data pipeline.

* prepare.parse_games / process_shard / prepare_split (evaluated shards -> arrays)
* Stockfish_train_eval.load_done / task_stream / write_meta (raw PGN -> tasks)
* Stockfish_train_eval._eval_game with a real engine (marker: engine)

Case IDs (UT-P-xx, IT-P-xx) match docs/test_cases.md.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import phase7_helpers as h
from features import N_FEATURES, build_sample, game_features
from prepare import parse_games, prepare_split, process_shard

MAX_PLIES, MIN_PLIES, CLIP = 200, 10, 1500.0


@pytest.fixture
def mixed_shard(tmp_path):
    """One shard with every kind of block prepare.py has to cope with."""
    good = h.oscillating_evals(40, seed=1)
    long = h.oscillating_evals(250, seed=2)
    text = (
        h.evaluated_block(1500, 1480, good, "1-0")                      # valid
        + '[WhiteElo "1500"]\n[BlackElo "1480"]\n[Result "1-0"]\n\n1. e4 1-0' + h.BLOCK_END  # no Evals
        + h.evaluated_block("?", 1480, good, "0-1")                     # bad Elo
        + h.evaluated_block(1500, 1480, ["12", "abc"], "0-1")           # non-integer eval
        + h.evaluated_block(1600, 1650, [10] * 5, "1/2-1/2")            # too short
        + h.evaluated_block(2100, 2050, long, "*")                      # longer than max_plies
        + '[WhiteElo "1700"]\n[BlackElo "1710"]\n[Evals "1,2,3'         # partial (crash) tail
    )
    path = tmp_path / "test_shard_001.pgn"
    path.write_text(text, encoding="utf-8")
    return path, good, long


# ------------------------------------------------------------- parse_games
def test_up01_parse_games_yields_every_complete_block(mixed_shard):
    path, good, _ = mixed_shard
    rows = list(parse_games(str(path)))
    assert len(rows) == 6                        # partial tail is ignored
    w, b, cp, result = rows[0]
    assert (w, b, result) == (1500, 1480, "1-0")
    assert cp.dtype == np.int32 and cp.tolist() == good
    assert rows[1] == (None, None, None, None)   # missing Evals
    assert rows[2] == (None, None, None, None)   # bad Elo
    assert rows[3] == (None, None, None, None)   # bad eval


def test_up02_missing_result_defaults_to_unknown(tmp_path):
    p = tmp_path / "s.pgn"
    p.write_text('[WhiteElo "1500"]\n[BlackElo "1400"]\n[Evals "1,2,3"]\n\n1. e4 *'
                 + h.BLOCK_END, encoding="utf-8")
    assert list(parse_games(str(p)))[0][3] == "*"


def test_up03_empty_shard(tmp_path):
    p = tmp_path / "empty.pgn"
    p.write_text("", encoding="utf-8")
    assert list(parse_games(str(p))) == []


# ------------------------------------------------------------ process_shard
def test_up04_process_shard_counts(mixed_shard):
    path, _, _ = mixed_shard
    _, counts = process_shard(str(path), MAX_PLIES, MIN_PLIES, CLIP)
    assert counts == {"games_seen": 6, "skipped_bad": 3, "skipped_short": 1, "truncated": 1}


def test_up05_process_shard_arrays(mixed_shard):
    path, good, long = mixed_shard
    a, _ = process_shard(str(path), MAX_PLIES, MIN_PLIES, CLIP)
    n = 4                                                   # 2 games x 2 players
    assert a["wp"].shape == (n, MAX_PLIES) and a["wp"].dtype == np.float32
    assert a["feat"].shape == (n, N_FEATURES) and a["feat"].dtype == np.float32
    assert a["length"].dtype == np.int16 and a["side"].dtype == np.int8
    assert a["side"].tolist() == [0, 1, 0, 1]
    assert a["rating"].tolist() == [1500, 1480, 2100, 2050]
    assert a["length"].tolist() == [40, 40, 200, 200]
    assert np.all(a["wp"][0, 40:] == 0.5)                    # padding
    assert np.all(np.isfinite(a["feat"]))


def test_up06_process_shard_matches_feature_functions(mixed_shard):
    """Training arrays are exactly what features.py produces (train/serve parity)."""
    path, good, long = mixed_shard
    a, _ = process_shard(str(path), MAX_PLIES, MIN_PLIES, CLIP)
    row, _ = build_sample(good, 1, MAX_PLIES, CLIP)
    np.testing.assert_array_equal(a["wp"][1], row)
    np.testing.assert_array_equal(a["feat"][1], game_features(good, 1, "1-0", CLIP))
    # features of the long game use all 250 plies, not the truncated 200
    np.testing.assert_array_equal(a["feat"][2], game_features(long, 0, "*", CLIP))


def test_up07_process_shard_with_no_usable_games(tmp_path):
    p = tmp_path / "s.pgn"
    p.write_text(h.evaluated_block(1500, 1500, [1, 2, 3]), encoding="utf-8")
    a, counts = process_shard(str(p), MAX_PLIES, MIN_PLIES, CLIP)
    assert a["wp"].shape == (0, MAX_PLIES) and a["feat"].shape == (0, N_FEATURES)
    assert counts["skipped_short"] == 1


# ------------------------------------------------------------ prepare_split
class FakeRay:
    """Runs Ray tasks inline, so prepare_split is testable without a cluster."""

    @staticmethod
    def remote(fn):
        class Task:
            @staticmethod
            def remote(*args):
                return fn(*args)
        return Task

    @staticmethod
    def get(futures):
        return list(futures)


def test_up08_prepare_split_concatenates_in_shard_order(tmp_path):
    for i, elo in ((2, 2000), (1, 1000), (3, 3000)):     # written out of order
        (tmp_path / f"s_{i:03d}.pgn").write_text(
            h.evaluated_block(elo, elo + 1, h.oscillating_evals(30, seed=i)), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored")
    arrays, counts = prepare_split(FakeRay, str(tmp_path), MAX_PLIES, MIN_PLIES, CLIP)
    assert arrays["rating"].tolist() == [1000, 1001, 2000, 2001, 3000, 3001]
    assert counts["shards"] == 3 and counts["games_used"] == 3 and counts["samples"] == 6


def test_up09_prepare_split_without_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_split(FakeRay, str(tmp_path), MAX_PLIES, MIN_PLIES, CLIP)


# ---------------------------------------------- Stockfish_train_eval parsing
@pytest.fixture(scope="module")
def sfe():
    return pytest.importorskip("Stockfish_train_eval")


def test_up10_load_done_drops_partial_tail(sfe, tmp_path):
    out = tmp_path / "shard.pgn"
    out.write_text(h.evaluated_block(1500, 1400, [1, 2], movetext="1. e4 e5 1-0")
                   + '[WhiteElo "1"]\n[Evals "1,', encoding="utf-8")
    done = sfe.load_done(str(out))
    assert done == {"1. e4 e5 1-0"}
    assert out.read_text(encoding="utf-8").endswith(h.BLOCK_END)   # tail removed on disk


def test_up11_load_done_missing_file(sfe, tmp_path):
    assert sfe.load_done(str(tmp_path / "nope.pgn")) == set()


def test_up12_task_stream_filters_and_resumes(sfe, tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    (inp / "s1.pgn").write_text(
        '[WhiteElo "1500"]\n[BlackElo "1400"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 1-0\n\n'
        '[WhiteElo "1600"]\n[BlackElo "1650"]\n[Result "0-1"]\n\n1. d4 d5 0-1\n\n'
        '[FEN "8/8/8/8/8/8/8/K6k w - - 0 1"]\n[Result "*"]\n\n1. Kb1 *\n\n'
        '[WhiteElo "1700"]\n[BlackElo "1700"]\n[Result "*"]\n\n*\n\n',
        encoding="utf-8")
    # the first game was already evaluated in an earlier run
    (out / "s1.pgn").write_text(
        h.evaluated_block(1500, 1400, [1, 2, 3], movetext="1. e4 e5 2. Nf3 1-0"),
        encoding="utf-8")
    tasks = list(sfe.task_stream(str(inp), str(out), limit=0))
    assert len(tasks) == 1                                   # FEN + empty + done skipped
    shard, uci, w, b, result, movetext = tasks[0]
    assert (shard, uci, w, b, result) == ("s1.pgn", ["d2d4", "d7d5"], "1600", "1650", "0-1")
    assert movetext == "1. d4 d5 0-1"


def test_up13_task_stream_limit(sfe, tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    (inp / "s.pgn").write_text("\n\n".join(
        h.random_legal_pgn(12, seed=i, result="1-0") for i in range(5)), encoding="utf-8")
    assert len(list(sfe.task_stream(str(inp), str(out), limit=2))) == 2


def test_up14_write_meta_accumulates(sfe, tmp_path):
    path = tmp_path / "m" / "eval.json"
    sfe.write_meta(str(path), "Stockfish X", games=10, plies=500, failed=1, seconds=5.0)
    meta = sfe.write_meta(str(path), "Stockfish X", games=10, plies=500, failed=0, seconds=5.0)
    assert meta["games"] == 20 and meta["positions"] == 1000
    assert meta["positions_per_sec"] == 100.0 and meta["failed_games_last_run"] == 0
    assert json.loads(path.read_text())["games"] == 20


# --------------------------------------------- real engine (integration)
@pytest.fixture(scope="module")
def engine(sfe):
    if not os.path.exists(sfe.ENGINE_PATH):
        pytest.skip(f"Stockfish not found at {sfe.ENGINE_PATH}")
    sfe.init_worker()
    yield sfe
    sfe._engine.quit()
    sfe._engine = None


@pytest.mark.engine
def test_ip01_eval_game_scores_checkmate(engine):
    # Fool's mate: Black mates on ply 4, so the last eval is -mate_score
    task = ("s.pgn", ["f2f3", "e7e5", "g2g4", "d8h4"], "1200", "1300", "0-1",
            "1. f3 e5 2. g4 Qh4# 0-1")
    res = engine._eval_game(task)
    assert res["plies"] == 4
    evals = [int(x) for x in res["block"].split('[Evals "')[1].split('"')[0].split(",")]
    assert evals[-1] == -engine.MATE_SCORE
    assert len(evals) == 4


@pytest.mark.engine
def test_ip02_eval_game_never_raises(engine):
    res = engine.eval_game(("s.pgn", ["e2e5"], "1", "1", "*", "illegal"))
    assert "error" in res


@pytest.mark.engine
def test_ip03_engine_output_roundtrips_through_prepare(engine, tmp_path):
    """evaluate stage output -> prepare stage input, with a real engine."""
    pgn = h.random_legal_pgn(24, seed=7, result="1/2-1/2")
    import io
    import chess.pgn
    game = chess.pgn.read_game(io.StringIO(pgn))
    moves = list(game.mainline_moves())
    movetext = f"{game.board().variation_san(moves)} 1/2-1/2"
    res = engine._eval_game(("s.pgn", [m.uci() for m in moves], "1500", "1550",
                             "1/2-1/2", movetext))
    shard = tmp_path / "s.pgn"
    shard.write_text(res["block"], encoding="utf-8")
    a, counts = process_shard(str(shard), MAX_PLIES, MIN_PLIES, CLIP)
    assert counts["games_seen"] == 1 and counts["skipped_bad"] == 0
    assert a["length"].tolist() == [len(moves)] * 2
    assert a["rating"].tolist() == [1500, 1550]

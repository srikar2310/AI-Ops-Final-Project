"""API tests against the running backend (marker: e2e).

These are black-box HTTP tests: they rely only on the interface in docs/LLD.md
(/health, /predict with {"pgn": ...}, /metrics), not on the backend's internals.

    docker compose up -d
    pytest -m "e2e and not drift" tests/test_api_live.py
    BACKEND_URL=http://otherhost:8000 pytest -m e2e ...      # another address

Case IDs (API-xx) match docs/test_cases.md.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import phase7_helpers as h

requests = pytest.importorskip("requests")
pytestmark = pytest.mark.e2e

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120
MAX_LATENCY_S = 5.0            # AC-8: one ~40-move game


@pytest.fixture(scope="module", autouse=True)
def backend_up():
    try:
        requests.get(f"{BACKEND}/health", timeout=5)
    except requests.RequestException:
        pytest.skip(f"backend not reachable at {BACKEND} (start it with docker compose up -d)")


@pytest.fixture(scope="module")
def sample_pgn():
    return json.loads((h.REPO / "sample_request.json").read_text())["pgn"]


def predict(pgn, **kw):
    return requests.post(f"{BACKEND}/predict", json={"pgn": pgn}, timeout=TIMEOUT, **kw)


def numbers_under_rating_keys(obj, found=None):
    """All numeric values whose key mentions rating or elo, anywhere in the JSON."""
    found = [] if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and \
                    re.search(r"rating|elo", str(k), re.I):
                found.append(v)
            else:
                numbers_under_rating_keys(v, found)
    elif isinstance(obj, list):
        for v in obj:
            numbers_under_rating_keys(v, found)
    return found


def metric_total(text, name):
    """Sum of every sample of a counter in Prometheus text format."""
    pat = re.compile(rf"^{name}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$", re.M)
    return sum(float(v) for v in pat.findall(text))


def metrics_text():
    return requests.get(f"{BACKEND}/metrics", timeout=10).text


# ------------------------------------------------------------------- health
def test_api01_health():
    r = requests.get(f"{BACKEND}/health", timeout=10)
    assert r.status_code == 200


def test_api02_openapi_documents_predict():
    spec = requests.get(f"{BACKEND}/openapi.json", timeout=10).json()
    assert "/predict" in spec["paths"] and "post" in spec["paths"]["/predict"]


# ------------------------------------------------------------- happy paths
def test_api03_predict_sample_game(sample_pgn):
    t0 = time.time()
    r = predict(sample_pgn)
    elapsed = time.time() - t0
    assert r.status_code == 200, r.text
    ratings = numbers_under_rating_keys(r.json())
    assert len(ratings) >= 2, f"no rating fields found in {r.json()}"
    assert all(100 <= x <= 3500 for x in ratings)
    assert elapsed < MAX_LATENCY_S, f"took {elapsed:.1f}s"


def test_api04_lichess_export_with_comments_and_clocks():
    pgn = ('[Event "Rated Blitz game"]\n[Site "https://lichess.org/abcdefgh"]\n'
           '[White "a"]\n[Black "b"]\n[Result "0-1"]\n[WhiteElo "1650"]\n[BlackElo "1702"]\n'
           '[TimeControl "300+0"]\n\n'
           '1. e4 { [%clk 0:05:00] } 1... c5 { [%clk 0:05:00] } 2. Nf3 { [%clk 0:04:58] } '
           '2... d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be2 e5?! 7. Nb3 Be7 8. O-O O-O '
           '9. Be3 Be6 10. f4 exf4 11. Rxf4 Nc6 12. Qd2 d5!? 0-1')
    assert predict(pgn).status_code == 200


def test_api05_long_game_with_unknown_result():
    r = predict(h.random_legal_pgn(160, seed=11, result="*"))
    assert r.status_code == 200, r.text


def test_api06_game_without_headers():
    assert predict("1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 1/2-1/2"
                   ).status_code == 200


def test_api07_concurrent_requests(sample_pgn):
    """More requests than engines in the pool: all must still succeed."""
    with ThreadPoolExecutor(4) as pool:
        codes = list(pool.map(lambda _: predict(sample_pgn).status_code, range(4)))
    assert codes == [200] * 4


# ------------------------------------------------------------ invalid input
@pytest.mark.parametrize("pgn", [
    "this is not chess",
    "1. e4 e5 2. Ke3 Ke6 3. Qh5 *",             # illegal king move from e1 to e3
    "",
    "   \n  ",
    "[White \"A\"]\n[Black \"B\"]\n[Result \"*\"]\n\n*",   # headers but no moves
], ids=["garbage", "illegal-move", "empty", "whitespace", "no-moves"])
def test_api08_bad_pgn_is_a_4xx_with_message(pgn):
    """Error shape (LLD 2.2): {"error": {"code": "bad_pgn", "message": ...}, "request_id": ...}"""
    r = predict(pgn)
    assert 400 <= r.status_code < 500, (r.status_code, r.text)
    body = r.json()
    assert body["error"]["code"] == "bad_pgn", body
    assert body["error"]["message"].strip(), "error response should explain the problem"
    assert body.get("request_id"), "error response should carry a request id"


@pytest.mark.parametrize("body", [{}, {"pgn": 123}, {"game": "1. e4"}],
                         ids=["missing", "not-a-string", "wrong-field"])
def test_api09_schema_violations_are_422(body):
    r = requests.post(f"{BACKEND}/predict", json=body, timeout=30)
    assert r.status_code == 422


def test_api10_non_json_body():
    r = requests.post(f"{BACKEND}/predict", data="1. e4 e5",
                      headers={"Content-Type": "text/plain"}, timeout=30)
    assert 400 <= r.status_code < 500


def test_api11_wrong_method_and_unknown_route():
    assert requests.get(f"{BACKEND}/predict", timeout=10).status_code == 405
    assert requests.get(f"{BACKEND}/no-such-route", timeout=10).status_code == 404


# ------------------------------------------------------------------ metrics
def test_api12_metrics_exposed():
    text = metrics_text()
    for name in ("api_requests_total", "api_request_latency_seconds_bucket",
                 "stockfish_engines_available", "feature_drift_max"):
        assert re.search(rf"^{name}", text, re.M), f"{name} missing from /metrics"


def test_api13_requests_and_errors_are_counted():
    before = metric_total(metrics_text(), "api_requests_total")
    predict("definitely not a pgn")
    after = metric_total(metrics_text(), "api_requests_total")
    assert after >= before + 1


def test_api14_engines_returned_to_pool(sample_pgn):
    predict(sample_pgn)
    text = metrics_text()
    m = re.search(r"^stockfish_engines_available(?:{[^}]*})?\s+([0-9.]+)", text, re.M)
    assert m and float(m.group(1)) >= 1, "no idle engines after a finished request"

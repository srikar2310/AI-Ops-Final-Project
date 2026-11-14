"""End-to-end tests of the whole docker compose stack (marker: e2e).

    docker compose up -d --build
    pytest -m "e2e and not drift"                       # everything but the drift scenario
    pytest -m drift                                     # drift scenario, run last

Override addresses with BACKEND_URL, MODEL_URL, MLFLOW_URL, PROMETHEUS_URL,
GRAFANA_URL, FRONTEND_URL.
Case IDs (E2E-xx) match docs/test_cases.md.
"""
import json
import math
import os
import re
import time

import pytest

import phase7_helpers as h

requests = pytest.importorskip("requests")
pytestmark = pytest.mark.e2e


def url(var, default):
    return os.environ.get(var, default).rstrip("/")


BACKEND = url("BACKEND_URL", "http://localhost:8000")
MODEL = url("MODEL_URL", "http://localhost:5001")
MLFLOW = url("MLFLOW_URL", "http://localhost:5000")
PROM = url("PROMETHEUS_URL", "http://localhost:9090")
GRAFANA = url("GRAFANA_URL", "http://localhost:3000")
FRONTEND = url("FRONTEND_URL", "http://localhost:8501")

ALERTS = {"ApiDown", "HighErrorRate", "NoStockfishEngines", "SlowPredictions", "InputDrift"}


def get(u, **kw):
    try:
        return requests.get(u, timeout=kw.pop("timeout", 10), **kw)
    except requests.RequestException as exc:
        pytest.fail(f"{u} unreachable: {exc}")


@pytest.fixture(scope="module", autouse=True)
def stack_up():
    try:
        requests.get(f"{BACKEND}/health", timeout=5)
    except requests.RequestException:
        pytest.skip("stack not running (docker compose up -d)")


def gauge(name):
    text = get(f"{BACKEND}/metrics").text
    m = re.search(rf"^{name}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)", text, re.M)
    assert m, f"{name} not in /metrics"
    return float(m.group(1))


def prom_query(q):
    r = get(f"{PROM}/api/v1/query", params={"query": q}).json()
    assert r["status"] == "success"
    return r["data"]["result"]


# ----------------------------------------------------------- service health
@pytest.mark.parametrize("name,u,check", [
    ("backend", f"{BACKEND}/health", None),
    ("model-server", f"{MODEL}/ping", None),
    ("mlflow", f"{MLFLOW}/health", None),
    ("prometheus", f"{PROM}/-/ready", None),
    ("grafana", f"{GRAFANA}/api/health", "ok"),
    ("frontend", f"{FRONTEND}/_stcore/health", "ok"),
])
def test_e2e01_every_service_is_healthy(name, u, check):
    r = get(u)
    assert r.status_code == 200, f"{name}: {r.status_code}"
    if check:
        assert check in r.text


# ----------------------------------------------------------- model server
def test_e2e02_model_server_invocations():
    ev = h.oscillating_evals(60, seed=5)
    body = {"dataframe_records": [{"evals": ev, "side": "white", "result": "1-0"},
                                  {"evals": ev, "side": "black", "result": "1-0"}]}
    r = requests.post(f"{MODEL}/invocations", json=body, timeout=60)
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert all(math.isfinite(p["predicted_rating"]) and 0 < p["predicted_rating"] < 4000
               for p in preds)


def test_e2e03_model_server_rejects_bad_side():
    body = {"dataframe_records": [{"evals": [1, 2, 3], "side": "green", "result": "*"}]}
    r = requests.post(f"{MODEL}/invocations", json=body, timeout=60)
    assert r.status_code >= 400


def test_e2e04_champion_alias_is_registered():
    r = get(f"{MLFLOW}/api/2.0/mlflow/registered-models/alias",
            params={"name": "chess-rating", "alias": "champion"})
    assert r.status_code == 200, r.text
    assert r.json()["model_version"]["version"]


# ----------------------------------------------------------- full user flow
def test_e2e05_frontend_to_backend_flow():
    """What the UI does: send the example game to the backend's REST API."""
    pgn = json.loads((h.REPO / "sample_request.json").read_text())["pgn"]
    r = requests.post(f"{BACKEND}/predict", json={"pgn": pgn}, timeout=120)
    assert r.status_code == 200, r.text
    page = get(FRONTEND)
    assert page.status_code == 200


# ------------------------------------------------------------- monitoring
def test_e2e06_prometheus_scrapes_backend():
    targets = get(f"{PROM}/api/v1/targets").json()["data"]["activeTargets"]
    assert targets, "no scrape targets"
    down = [t["scrapeUrl"] for t in targets if t["health"] != "up"]
    assert not down, f"targets down: {down}"


def test_e2e07_alert_rules_loaded():
    groups = get(f"{PROM}/api/v1/rules").json()["data"]["groups"]
    names = {rule["name"] for g in groups for rule in g["rules"]}
    assert ALERTS <= names, f"missing alert rules: {ALERTS - names}"


def test_e2e08_no_alerts_firing_on_a_healthy_stack():
    firing = [a["labels"]["alertname"] for a in
              get(f"{PROM}/api/v1/alerts").json()["data"]["alerts"]
              if a["state"] == "firing" and a["labels"]["alertname"] != "InputDrift"]
    assert not firing, f"firing: {firing}"


@pytest.mark.slow
def test_e2e09_request_reaches_prometheus():
    """A request made now shows up in Prometheus after the next scrape."""
    def total():
        res = prom_query("sum(api_requests_total)")
        return float(res[0]["value"][1]) if res else 0.0
    before = total()
    for _ in range(3):
        get(f"{BACKEND}/health")
    deadline = time.time() + 60
    while time.time() < deadline:
        if total() >= before + 3:
            return
        time.sleep(5)
    pytest.fail("Prometheus did not see the new requests within 60 s")


def test_e2e10_grafana_dashboard_provisioned():
    r = get(f"{GRAFANA}/api/search", params={"type": "dash-db"}, auth=("admin", "admin"))
    assert r.status_code == 200 and len(r.json()) >= 1
    ds = get(f"{GRAFANA}/api/datasources", auth=("admin", "admin")).json()
    assert any(d["type"] == "prometheus" for d in ds)


# ----------------------------------------------------- drift (run separately)
def real_test_games(limit):
    folder = h.REPO / "dataset_clean" / "test"
    shards = sorted(folder.glob("*.pgn"))
    if not shards:
        pytest.skip("dataset_clean/test not found; needed for the recovery half")
    import chess.pgn
    games = []
    with open(shards[0], encoding="utf-8", errors="ignore") as f:
        while len(games) < limit:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            if len(list(g.mainline_moves())) >= 20:
                games.append(str(g))
    return games



@pytest.mark.drift
@pytest.mark.slow
def test_e2e11_drift_detected_and_recovers():
    """20 copies of a 19-ply opening push drift to the cap; real games bring it back."""
    pgn = f'[Result "*"]\n\n{h.OPENING_19} *'
    for _ in range(20):
        assert requests.post(f"{BACKEND}/predict", json={"pgn": pgn},
                             timeout=120).status_code == 200
    peak = gauge("feature_drift_max")
    assert peak > 1.0, f"drift {peak:.2f} did not cross the InputDrift threshold"

    for g in real_test_games(40):
        requests.post(f"{BACKEND}/predict", json={"pgn": g}, timeout=120)
    after = gauge("feature_drift_max")
    assert after < peak, f"drift did not fall after real games ({peak:.2f} -> {after:.2f})"

# Test Plan: Chess Rating Estimator

## 1. Objectives

1. Show that every component behaves as specified in [LLD.md](LLD.md): feature engineering, PGN parsing, model inputs, the serving wrapper, the REST API and the monitoring stack.
2. Show that training and serving compute exactly the same thing (no train/serve skew).
3. Show that the system as a whole meets the acceptance criteria in section 6. Model accuracy is not a test objective: it is reported as a description in [HLD.md](HLD.md), not as pass or fail.
4. Show that the application fails safely on bad input and that faults are visible in monitoring.

## 2. Scope

**In scope:** `features.py`, `prepare.py`, `Stockfish_train_eval.py`, `model.py`, `pyfunc_model.py`, the outputs of the DVC pipeline, the registered champion model, the FastAPI backend, the MLflow model server, Prometheus alert rules, Grafana provisioning, the Streamlit UI (manual), and the drift check.

**Out of scope:** `Parse.py` and `clean_data.py` are one-off scripts with no functions to call; their output is covered indirectly by the data-quality checks (DQ). `train.py`'s training loop is covered by the MLflow runs and the serving-parity check rather than unit tests. Model accuracy targets (MAE, error by rating band) are out of scope for testing. Load testing beyond a few concurrent users is out of scope.

## 3. Test levels and approach

| Level | Suite (file) | What it checks | Needs | Marker |
|---|---|---|---|---|
| Unit | `tests/test_features_unit.py` (UT-F) | Win-probability conversion, padding, the 27 features, error attribution, result handling, edge cases | Python only | none |
| Unit | `tests/test_pgn_parsing.py` (UT-P) | Parsing evaluated shards, skip and truncation counts, Ray stage with a fake Ray, resume logic and task filtering of the evaluation stage | Python only | none |
| Integration | `tests/test_pgn_parsing.py` (IT-P) | Real Stockfish evaluation, mate scoring, crash safety, evaluate-stage output read back by the prepare stage | Stockfish binary | `engine` |
| Unit | `tests/test_model_unit.py` (UT-M) | `make_inputs` channels, padding invariance, batch independence, architecture validation, feature buffers | PyTorch | none |
| Unit | `tests/test_pyfunc_unit.py` (UT-S) | Serving wrapper I/O contract, input validation, parity with direct computation | PyTorch, MLflow | none |
| Integration | `tests/test_pyfunc_unit.py` (IT-S) | MLflow save/load round trip gives identical predictions | MLflow | `slow` |
| Data quality | `tests/test_data_and_acceptance.py` (DQ) | Real `data/processed`, `stats.json`, `metrics/*.json`: shapes, ranges, no leakage, no failures | pipeline outputs | `data` |
| Acceptance | `tests/test_data_and_acceptance.py` (AC) | Serving parity and robustness checks recorded in `reports/model_test_report.json` | `register_model.py` output | `data` |
| API | `tests/test_api_live.py` (API) | Black-box HTTP tests of `/health`, `/predict`, `/metrics`: happy paths, bad input, concurrency, latency, metrics counting | running stack | `e2e` |
| End to end | `tests/test_e2e_stack.py` (E2E) | All six services healthy, model server, registry alias, UI-to-backend flow, Prometheus targets and rules, Grafana provisioning, drift scenario | running stack | `e2e` (+ `drift`) |
| Manual UI | [test_cases.md](test_cases.md) (UI) | Foolproof UI: input modes, errors, results, responsiveness, service-down message | browser | manual |

Design principles:

* **Black-box API tests.** The API and E2E suites use only the interface documented in the LLD, so they test the contract the UI depends on, not the implementation.
* **Every suite skips cleanly** when its dependency (Stockfish, data, a running stack) is missing, so `pytest` is always safe to run.
* **Real data, real engine where it matters.** IT-P runs the real Stockfish binary; DQ and AC run on the real pipeline outputs.

## 4. Environment

| Item | Value |
|---|---|
| OS | Linux laptop (Ubuntu), Docker Engine with compose v2 |
| Python | 3.12 (compose images); tests also pass on 3.13 |
| Stockfish | `/usr/games/stockfish` (training data: Stockfish 16; backend image: 15.1, a known limitation) |
| Test tools | pytest 8+, requests; `conftest.py` at the repo root puts the root on `sys.path` and registers the markers |
| Stack | `docker compose up -d --build` (backend 8000, frontend 8501, model-server 5001, mlflow 5000, prometheus 9090, grafana 3000) |

## 5. How to run

```bash
pip install -r requirements.txt            # includes pytest, requests
mkdir -p reports

# 1. offline: unit, integration, data quality, acceptance (no stack needed)
pytest -m "not e2e" --junitxml=reports/junit_offline.xml

# 2. online: start the stack, then API + E2E (drift scenario excluded)
docker compose up -d --build
pytest -m "e2e and not drift" --junitxml=reports/junit_e2e.xml

# 3. drift scenario last, because it fills the drift window with odd games
pytest -m drift --junitxml=reports/junit_drift.xml

# 4. counts for the test report
python scripts/test_report.py reports/junit_*.xml > reports/test_results.md
```

A run passes when pytest ends with `N passed` and no `failed` or `error`. Skips are reported separately and are not counted as passes. The manual UI cases (UI-xx) are run in a browser and recorded in the report by hand.

## 6. Acceptance criteria

The release is accepted when every criterion below is met. The criteria are about the system working correctly; model accuracy is deliberately not an acceptance criterion.

| ID | Criterion | Threshold | Verified by |
|---|---|---|---|
| AC-1 | Train/serve parity: test MAE re-measured through the serving wrapper equals the MAE logged at training | gap ≤ 1.0 point | AC-01, UT-P-06, UT-S-07 |
| AC-2 | Model robustness: unknown result handled; bad side and empty evals rejected; output in 0 to 4000 | all pass | AC-02, UT-S-01 to UT-S-09 |
| AC-3 | `/predict` latency for a typical (about 40-move) game | < 5 s | API-03 |
| AC-4 | Invalid input never causes a server error | 100% of invalid inputs give a 4xx with a message | API-08 to API-11 |
| AC-5 | Stack health after `docker compose up` | all 6 services healthy, all scrape targets up, 5 alert rules loaded | E2E-01, E2E-06, E2E-07 |
| AC-6 | Monitoring | requests, errors and engine state visible in `/metrics` and Prometheus | API-12 to API-14, E2E-09, E2E-10 |
| AC-7 | Drift detection | a shifted input stream raises `feature_drift_max` above 1 (InputDrift threshold), and it falls again on normal games | E2E-11 |
| AC-8 | Pipeline and data integrity | all data-quality checks pass | DQ-01 to DQ-06 |
| AC-9 | Automated tests | 100% of unit and integration tests pass; every other failure is listed in the report with a cause | all |

## 7. Entry and exit criteria

**Entry:** the code is committed; `dvc repro` and `register_model.py` have run; the compose stack builds.

**Exit:** all AC met, or every unmet criterion recorded in the test report with its cause and a decision (fix now, or accept as a known limitation).

## 8. Defect handling

A failing case is re-run once to rule out flakiness (e.g. a slow first scrape). A confirmed failure is logged in the report's defect table with the case ID, symptom, cause and status.

## 9. Risks to testing

| Risk | Mitigation |
|---|---|
| The Stockfish version differs between machines, so evals differ slightly | Unit tests never assert exact engine scores, only mate scores and structure |
| E2E tests depend on the stack's state (drift window, counters) | Assertions use differences (before vs after); drift scenario runs last and alone |
| Prometheus scrape interval delays metrics | E2E-09 polls for up to 60 s |

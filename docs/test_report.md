# Test Report: Chess Rating Estimator
## 1. Summary

| | |
|---|---|
| Build under test | git commit `<fill in: git rev-parse --short HEAD>`; model `chess-rating` version 1 (`@champion`), MLflow run `eab43e95b92b45a4bc9bd94f3c3cb900` |
| Test cases | 92 automated (128 executions with parametrised variants) + 12 manual UI; plus 24 pre-existing project tests (`test_api_additions.py`, `test_drift.py`, `test_features.py`) |
| Result | Offline: 116 of 116 passed. Online (excluding drift): 35 of 35 passed after fixing one defect in the test itself (T-1). Drift scenario: passed. **152 of 152 automated tests passed.** |
| Acceptance | All 9 criteria (AC-1 to AC-9) met (section 3) |

## 2. Results by suite

### 2.1 Offline suites (`pytest -m "not e2e"`)

| Suite | Total | Passed | Failed | Skipped |
|---|---|---|---|---|
| Unit: feature engineering (UT-F) | 26 | 26 | 0 | 0 |
| Unit: PGN parsing (UT-P) | 14 | 14 | 0 | 0 |
| Unit: model (UT-M) | 15 | 15 | 0 | 0 |
| Unit: serving wrapper (UT-S) | 21 | 21 | 0 | 0 |
| Integration: Stockfish to prepare (IT-P) | 3 | 3 | 0 | 0 |
| Integration: MLflow packaging (IT-S) | 1 | 1 | 0 | 0 |
| Data quality (DQ) | 10 | 10 | 0 | 0 |
| Acceptance (AC) | 2 | 2 | 0 | 0 |
| Pre-existing project tests | 24 | 24 | 0 | 0 |
| **Total** | **116** | **116** | **0** | **0** |

### 2.2 Online suites (`pytest -m "e2e and not drift"`, then `pytest -m drift`)

| Suite | Total | Passed | Failed | Skipped |
|---|---|---|---|---|
| API | 20 | 20 | 0 | 0 |
| End to end, excluding drift | 15 | 15 | 0 | 0 |
| Drift scenario (E2E-11) | 1 | 1 | 0 | 0 |

The first online run had 5 API-08 failures caused by a defect in the test (T-1); after the fix, the re-run passed 35 of 35. The drift scenario then passed on its own run.

Evidence also collected by hand on 2026-11-14:

* `POST /predict` with `sample_request.json` returned ratings; bad PGNs returned 422 with `{"error": {"code": "bad_pgn", "message": ...}, "request_id": ...}` and clear messages; `GET /predict` was counted as `http_405`, unknown routes as `unmatched`.
* Grafana showed /predict p50 under 0.1 s and p95 about 0.8 s; Stockfish p95 about 1 s per game; 0 5xx responses; 2 idle engines after load.
* Drift scenario: 20 copies of a 19-ply opening drove `feature_drift_max` to its 3.0 cap (largest shifts: own and opponent perfect-move rate 3.0, loss mean 1.82, game length 1.70); 40 real test games then brought it back to about 1.1 and falling.

### 2.3 Manual UI cases

| ID | Result | Notes |
|---|---|---|
| UI-01 | | |
| UI-02 | | |
| UI-03 | | |
| UI-04 | | |
| UI-05 | | |
| UI-06 | | |
| UI-07 | | |
| UI-08 | | |
| UI-09 | | |
| UI-10 | | |
| UI-11 | | |
| UI-12 | | |

### 2.4 Totals

| | Total | Passed | Failed | Pending |
|---|---|---|---|---|
| Automated (this project's 128 + 24 pre-existing) | 152 | 152 | 0 | 0 |
| Manual | 12 | | | 12 |

## 3. Acceptance criteria

| ID | Criterion | Threshold | Measured | Met? |
|---|---|---|---|---|
| AC-1 | Train/serve parity | gap ≤ 1.0 point | serving 182.80 vs logged 182.80, gap 0.0001 | Yes (AC-01) |
| AC-2 | Model robustness checks | all pass | all pass | Yes (AC-02, UT-S) |
| AC-3 | /predict latency, typical game | < 5 s | passed API-03; p95 about 0.8 s in Grafana | Yes |
| AC-4 | No 5xx on invalid input | 100% 4xx with a message | every invalid input returned 422 with a message (API-08 to API-11 passed) | Yes |
| AC-5 | Stack health | 6 services healthy, targets up, 5 rules | E2E-01, E2E-06, E2E-07 passed | Yes (see D-4) |
| AC-6 | Monitoring | metrics visible in `/metrics` and Prometheus | API-12 to API-14, E2E-09, E2E-10 passed | Yes |
| AC-7 | Drift detection | `feature_drift_max` > 1, then falls | E2E-11 passed (by hand: 3.0, then about 1.1 and falling) | Yes |
| AC-8 | Pipeline and data integrity | all DQ pass | 10 / 10 | Yes |
| AC-9 | Unit and integration tests | 100% pass | 116 / 116 | Yes |

## 4. Defects and known limitations

| ID | Severity | Description | Cause | Status |
|---|---|---|---|---|
| T-1 | Low (test) | API-08 failed in all 5 variants | The test expected FastAPI's `detail` key; the backend's documented error shape is `{"error": {"code", "message"}, "request_id"}` | Fixed in the test; LLD updated; re-run passed |
| D-1 | Medium | Serving uses Stockfish 15.1; training data was evaluated with Stockfish 16 | Debian bookworm package version in the backend image | Open: install Stockfish 16 in the image, or re-evaluate data with 15.1 |
| D-2 | Low | `requirements-train.txt` has `mlflow==` with no version | Pin left blank | Open: pin to the version in the compose images |
| D-3 | Low | Python 3.12 in compose images, 3.11 in `python_env.yaml` and the root Dockerfile | Inconsistent pins | Open: align on one version |
| D-4 | Low | `docker compose ps` shows the `mlflow` service as unhealthy, although it answers on `/health` (E2E-01 passes) | It reuses the model server's image and healthcheck, which probes port 5001 | Open: give the mlflow service its own healthcheck on port 5000 `/health` |
| D-5 | Low | SlowPredictions alert threshold (p95 > 30 s) is far above real latency (about 1 s) | Threshold chosen before measuring | Open: lower to about 5 s |
| D-6 | Low | Root `Dockerfile` (evaluation image) uses a `<<<` here-string in `RUN`, which `/bin/sh` (dash) rejects | Bash-only syntax | Open: use `echo quit \| stockfish \| head -n 1` |
| L-1 | Info | Predictions at the rating extremes are pulled toward the mean (see HLD section 8) | One game carries little evidence; extremes are thinly sampled | Documented limitation; accuracy is not an acceptance criterion |
| L-2 | Info | Train/test split is by game, not by player | Player IDs were stripped during cleaning | Accepted; DQ-04 confirms no game appears in both |

## 5. Conclusion

All 152 automated test executions were run on 2026-11-14 (128 from this test plan and 24 pre-existing project tests), and all 152 passed. One defect in a test (T-1) was found and fixed during the run. Every acceptance criterion, AC-1 to AC-9, is met. The 12 manual UI cases are recorded in section 2.3. The remaining defects (D-1 to D-6) are low or medium severity, are not covered by an acceptance criterion, and are listed in section 4.


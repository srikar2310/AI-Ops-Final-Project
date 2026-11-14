# Chess Rating Estimator - Backend API

PGN in -> local Stockfish (same settings as training) -> rating model -> JSON out.
Frontend and backend are separate: the UI only talks to this API over a configurable REST URL.

## Run (from the repo root)

```bash
pip install -r api/requirements.txt -r api/requirements-dev.txt
pip install -r api/requirements-local.txt            # only for local-model mode

# A) dev: model loaded in-process from models/best.pt + models/config.json
CHESS_SERVING_BACKEND=local uvicorn api.main:app --port 8000

# B) rubric mode: separate MLflow model server, API calls it over REST
mlflow models serve -m <MODEL_URI> -p 5001 --env-manager local
CHESS_SERVING_BACKEND=rest CHESS_SERVING_URL=http://localhost:5001 uvicorn api.main:app --port 8000
```
Docs UI: http://localhost:8000/docs   Metrics: http://localhost:8000/metrics

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/analyze` | `{pgn}` -> per-move eval table, blunder/mistake/inaccuracy labels, per-player summary |
| POST | `/predict` | `{pgn, side: white\|black\|both = both, include_analysis = false}` -> estimated rating(s) |
| GET | `/health` | liveness |
| GET | `/ready` | 200 only when Stockfish AND the model are usable, else 503 |
| GET | `/model-info` | model backend, run id / commit / URI, engine depth, preprocessing settings |
| GET | `/metrics` | Prometheus |

Schemas: see `api/schemas.py` (OpenAPI is generated at `/openapi.json`).

Errors always look like `{"error": {"code", "message", "request_id"}}`:

| HTTP | code | meaning |
|---|---|---|
| 422 | `invalid_pgn` / `unsupported_game` / `game_too_short` / `game_too_long` / `invalid_request` | fix the input |
| 503 | `engine_busy` / `model_unavailable` | retry shortly |
| 504 | `engine_timeout` | game took longer than `CHESS_GAME_TIMEOUT_S` |
| 502 | `engine_failure` / `model_error` | dependency failed |
| 500 | `internal_error` | bug; quote the request id |

## Train/serve parity (the important part)

* Eval depth, hash, mate score, min plies and cp clip are read from `params.yaml`, never from env vars.
* `engine_service.py` re-implements `Stockfish_train_eval._eval_game` line for line
  (white-perspective cp after each ply, mate / stalemate rules, Threads=1).
* Features and the model input come from `features.py` / `pyfunc_model.py` - imported, not copied.
* `tests/api/test_features_parity.py` and `test_engine_parity.py` enforce all of this.

## Configuration (env vars, prefix `CHESS_`)

`STOCKFISH_PATH`, `ENGINE_POOL_SIZE` (2), `ENGINE_ACQUIRE_TIMEOUT_S` (30), `MOVE_TIMEOUT_S` (10),
`GAME_TIMEOUT_S` (120), `MAX_GAME_PLIES` (300), `SERVING_BACKEND` (rest|local), `SERVING_URL`,
`ARTIFACTS_DIR`, `MLFLOW_MODEL_URI`, `MLFLOW_RUN_ID`, `GIT_COMMIT`, `LOG_FEATURES` (true),
`FEATURE_LOG_PATH`, `LOG_LEVEL`, `LOG_JSON`, `CORS_ORIGINS`.

## Drift hook

With `CHESS_LOG_FEATURES=true`, every prediction appends one JSON line (request id, color, prediction,
all 27 named features) to `data/serving/features.jsonl`. Phase 6 compares that file against
`data/processed/stats.json` (train feat_mean / feat_std).

## Tests

```bash
pytest                      # everything; the Stockfish parity test skips if the binary is missing
pytest -m "not stockfish"   # no engine needed
```

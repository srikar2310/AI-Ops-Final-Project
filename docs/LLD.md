# Low-Level Design: Chess Rating Estimator

## 1. Services and configuration

| Service | Base URL (inside compose) | Configured by |
|---|---|---|
| frontend | `http://frontend:8501` | `API_URL=http://backend:8000` (only dependency on the rest of the system) |
| backend | `http://backend:8000` | environment variables below + `params.yaml` |
| model-server | `http://model-server:5001` | registry alias `models:/chess-rating@champion` |
| mlflow | `http://mlflow:5000` | `sqlite:////mlflow-data/mlflow.db`, artifacts `/mlflow-data/mlruns` |
| prometheus | `http://prometheus:9090` | `monitoring/prometheus.yml`, `monitoring/alerts.yml` |
| grafana | `http://grafana:3000` | `monitoring/grafana/provisioning`, `monitoring/grafana/dashboards` |

**Backend settings** (`api/settings.py`). Engine settings come from `params.yaml` so the API analyses a game exactly like the training data: `eval.engine_path` (override `STOCKFISH_PATH`), `eval.depth`, `eval.hash_mb`, `eval.mate_score`, `data.min_plies`, `data.cp_clip`. Everything else is an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_POOL_SIZE` | 2 | Stockfish engines in the pool (one game each at a time) |
| `ENGINE_ACQUIRE_TIMEOUT_S` | 5 | wait for a free engine before answering 503 |
| `ENGINE_POSITION_TIMEOUT_S` | 30 | one search; longer means the engine is hung |
| `ANALYSIS_TIMEOUT_S` | 90 | whole-game budget; longer answers 504 |
| `API_MAX_PLIES` | 400 | longest accepted game |
| `API_MAX_PGN_BYTES` | 200000 | largest accepted PGN for one game |
| `API_MAX_UPLOAD_BYTES` | 5000000 | largest uploaded file for `/games/split` |
| `API_MAX_GAMES_PER_FILE` | 100 | games listed from one file |
| `MODEL_MODE` | `rest` | `rest` = MLflow model server over HTTP; `local` = load the model in-process |
| `MLFLOW_URL` | `http://127.0.0.1:5001/invocations` | model server endpoint (REST mode) |
| `MODEL_URI` | `models:/chess-rating@champion` | model to load (local mode) |
| `MODEL_TIMEOUT_S` | 30 | model-server request timeout |
| `DRIFT_STATS_PATH` | `data/processed/stats.json` | training statistics for the drift monitor (drift is off if unreadable) |
| `PROJECT_ROOT` | `.` | folder holding `dvc.yaml`, `dvc.lock`, `metrics/` for `/ops/pipeline` |
| `MLFLOW_TRACKING_URL` | `http://127.0.0.1:5000` | MLflow tracker read by `/ops/runs` |
| `MLFLOW_UI_URL`, `GRAFANA_URL`, `PROMETHEUS_URL`, `RAY_URL` | localhost addresses | links returned by `/ops/links` (opened by the browser) |
| `LOG_LEVEL`, `LOG_FILE` | `INFO`, none | logging (file is rotated at 5 MB, 3 backups) |
| `CORS_ORIGINS` | `*` | allowed browser origins, comma-separated |

The backend runs with **one uvicorn worker**: the engine pool and the Prometheus metrics are per process, so it scales by adding containers, not workers.

## 2. Backend REST API (FastAPI, port 8000)

All bodies are JSON. Every response carries an `X-Request-ID` header (taken from the request's `X-Request-ID` if sent, otherwise generated); the same id appears in error bodies and in every log line of that request. Interactive documentation: `/docs` (`/` redirects there).

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Analyse one game and estimate the players' ratings |
| POST | `/validate` | Check a PGN without running Stockfish (instant feedback in the UI) |
| POST | `/games/split` | List the games in an uploaded PGN file |
| GET | `/config` | Limits the UI shows and checks against |
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness: engine pool and model service usable |
| GET | `/metrics` | Prometheus metrics |
| GET | `/ops/pipeline`, `/ops/runs`, `/ops/service`, `/ops/links` | Read-only data for the Pipeline Console |

### 2.1 Error format (all endpoints)

```json
{"error": {"code": "bad_pgn",
           "message": "The game is too short: it has 0 half-moves and the model needs at least 10."},
 "request_id": "3f9c2a1b7e04"}
```

`code` is machine-readable (and is the `error_type` label of `api_errors_total`); `message` is written for an end user and the UI shows it as is.

| Status | `code` | When |
|---|---|---|
| 422 | `bad_pgn` | The PGN is unusable: empty; larger than `API_MAX_PGN_BYTES`; unreadable; no game found; more than one game; a variant other than standard; a custom start position (`FEN`); an invalid or illegal move; fewer than `data.min_plies` (10) or more than `API_MAX_PLIES` (400) half-moves. Also an empty, oversized or game-less file on `/games/split`. |
| 422 | `invalid_request` | The body does not match the schema (not JSON, missing `pgn`, wrong type, unknown `player`). The message names the field. |
| 404 / 405 | `http_404` / `http_405` | Unknown route / wrong HTTP method. |
| 502 | `model_unavailable` | The model service is unreachable, returned an error, or returned something unusable. |
| 503 | `engine_busy` | All engines stayed busy for `ENGINE_ACQUIRE_TIMEOUT_S`. Sends `Retry-After: 5`. |
| 503 | `engine_failure` | Stockfish crashed or hung, and a retry on a fresh engine also failed. |
| 503 | `ops_unavailable` | A Pipeline Console source (`dvc.yaml`, MLflow tracker) cannot be read. |
| 504 | `analysis_timeout` | The whole game took longer than `ANALYSIS_TIMEOUT_S`. |
| 500 | `internal_error` | Any unexpected exception. The traceback is logged, never returned. |

### 2.2 `POST /predict`

**Request**

```json
{"pgn": "[White \"A\"]\n[Black \"B\"]\n[WhiteElo \"1500\"]\n[BlackElo \"1480\"]\n[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 ... 1-0",
 "player": "both"}
```

| Field | Type | Required | Rules |
|---|---|---|---|
| `pgn` | string | yes | Exactly one standard-chess game from the normal start position, 10 to 400 half-moves, at most 200,000 bytes. Headers optional. |
| `player` | `"white"`, `"black"` or `"both"` | no (default `both`) | Which player(s) to rate. |

**Processing**

1. `pgn_utils.parse_pgn` validates the text (rules in 2.1). If the `Result` tag is missing or `*`, the result is inferred from the final position when it is checkmate, stalemate or insufficient material (`result_inferred: true`).
2. `EnginePool.analyse_game` borrows an engine and evaluates the position after every ply at `eval.depth`, White's view; checkmate scores ±`mate_score`, stalemate and insufficient material score 0, exactly as `Stockfish_train_eval.py`. On an engine crash or hang the engine is replaced and the game is analysed once more.
3. One record per requested side, `{"evals": [...], "side": "white"|"black", "result": "<result>"}`, is sent to the model service (section 3).
4. Each prediction is fed to the drift monitor; metrics are updated.
5. `labels.annotate_moves` labels every move from the mover's win-probability drop (inaccuracy ≥ 0.10, mistake ≥ 0.20, blunder ≥ 0.30, same thresholds as the training features).

**Response 200** (`PredictResponse`)

| Field | Type | Meaning |
|---|---|---|
| `request_id` | string | Same as the `X-Request-ID` header. |
| `players[]` | list | One per requested side: `side`, `name` (PGN tag, default "White"/"Black"), `predicted_rating` (float, 0.1), `actual_rating` (int or null, from `WhiteElo`/`BlackElo`). |
| `game` | object | `plies`, `result`, `result_known`, `result_inferred`. |
| `analysis` | object | `engine` (name reported by Stockfish), `depth`, `positions_searched`, `seconds`, `evals_white_cp[]` (one per ply). |
| `model` | object | `mode` (`rest`/`local`), `target` (URL or model URI), `seconds`. |
| `moves[]` | list | Per ply: `ply` (1-based), `side`, `san`, `eval_cp`, `white_win_prob` (0 to 1), `loss` (0 to 1), `label` (`ok`, `inaccuracy`, `mistake`, `blunder`). |
| `move_summary` | object | `{"white": {"inaccuracy": n, "mistake": n, "blunder": n}, "black": {...}}` |
| `warnings[]` | list of strings | Plain-language notes: unknown result, inferred result, or a short game (under 20 half-moves). |

### 2.3 `POST /validate`

Request `{"pgn": "..."}`. Applies the same checks as `/predict` without running Stockfish, so the UI can give instant feedback. A bad game is a normal answer, not an HTTP error:

| Field | Meaning |
|---|---|
| `valid` | true or false |
| `error` | `{code, message}` when `valid` is false |
| `plies`, `result`, `result_known`, `result_inferred`, `white`, `black`, `white_elo`, `black_elo` | Summary when valid |
| `warnings[]` | Same warnings as `/predict` |

### 2.4 `POST /games/split`

Request `{"text": "<contents of a PGN file>"}` (at most 5 MB). Response:

| Field | Meaning |
|---|---|
| `games[]` | Up to 100 items: `index` (1-based), `white`, `black`, `white_elo`, `black_elo`, `result`, `plies`, `usable` (passes the `/predict` checks), `problem` (message if not), `pgn` (this game alone, ready for `/predict`) |
| `truncated` | true if the file held more games than the limit |
| `limit` | the limit (`API_MAX_GAMES_PER_FILE`) |

### 2.5 `GET /config`

`{api_version, min_plies, max_plies, max_pgn_bytes, max_upload_bytes, max_games_per_file, short_game_plies, engine_depth, analysis_timeout_s}`. The UI reads its limits from here and hard-codes none.

### 2.6 `GET /health` and `GET /ready`

| Endpoint | 200 body | Other |
|---|---|---|
| `/health` | `{"status": "ok"}` (the process is up) | used by the Docker healthcheck |
| `/ready` | `{"status": "ready", "engine": true, "model": true}` | 503 with `"status": "not_ready"` and the failing part `false`; the UI sidebar shows this |

### 2.7 `GET /metrics`

Prometheus text format. See section 6.

### 2.8 Pipeline Console endpoints (`api/ops.py`)

Read-only, no authentication (keep them inside your own network). A source that cannot be read gives 503 `ops_unavailable`.

| Method | Path | Response |
|---|---|---|
| GET | `/ops/pipeline` | `stages[]`: `name` (foreach stages expanded as DVC names them, e.g. `evaluate@train`), `cmd`, `deps[]`, `outs[]`, `status` (`has_run` if in `dvc.lock`, else `never_run`), `last_run` (time of its newest metrics file), `summary` (seconds, games, positions/s, games/s, failed games, val/test MAE where present), `metrics` (the stage's metrics JSON). Also `edges[]` (`[producer, consumer]`), `mermaid` (the DAG as Mermaid text), `lock_found`. |
| GET | `/ops/runs?limit=20&experiment=<name>` | `runs[]` (newest first, `limit` 1 to 100): `run_id`, `experiment`, `status`, `start_time`, `end_time`, `git_commit`, selected `metrics` and `params`; `counts` of runs per status (FINISHED, FAILED, ...). |
| GET | `/ops/service` | Counters of this API process: `uptime_seconds`, `engine_ready`, `model_ready`, `requests_total`, `requests_by_status_class`, `errors_by_type`, `requests_in_progress`, `predictions`, `predict_avg_seconds`, `stockfish{positions, failures, restarts, engines_available}`. |
| GET | `/ops/links` | `{ray, mlflow, grafana, prometheus}` browser addresses. |

## 3. Model server API (MLflow pyfunc, port 5001)

### 3.1 `GET /ping` and `GET /health`

200 when the model is loaded.

### 3.2 `POST /invocations`

**Request** (`Content-Type: application/json`), one record per player to rate:

```json
{"dataframe_records": [
  {"evals": [20, 15, 35, -10, 40], "side": "white", "result": "1-0"},
  {"evals": [20, 15, 35, -10, 40], "side": "black", "result": "1-0"}
]}
```

| Field | Type | Required | Rules |
|---|---|---|---|
| `evals` | list of int | yes | White-perspective centipawns after every ply; at least 1 element. Clipped to ±`cp_clip` (1500); only the first `max_plies` (200) feed the CNN, all plies feed the features. |
| `side` | string | yes | `white`/`w`/`0` or `black`/`b`/`1` (case-insensitive). Anything else raises `ValueError`. |
| `result` | string | no | `1-0`, `0-1`, `1/2-1/2` or `1/2`; anything else, or missing, means unknown (`result_known = 0`). |

**Response 200**

```json
{"predictions": [{"predicted_rating": 1734.2}, {"predicted_rating": 1690.8}]}
```

`predicted_rating` is a float rounded to 0.1, in the same order as the input records.

**Errors:** an empty `evals` list or an invalid `side` returns a 4xx/5xx from MLflow with the `ValueError` message; the backend must map that to its own error.

## 4. Module design (Python)

### 4.1 `features.py` (shared by training and serving)

| Function | Signature | Contract |
|---|---|---|
| `cp_to_wp` | `(cp, clip) -> ndarray[float32]` | White win probability `1/(1+exp(-0.00368208*clip(cp,±clip)))`, in (0, 1). |
| `build_sample` | `(cp, side, max_plies, clip) -> (row[max_plies] float32, n int)` | Player's-view win probability, truncated to `max_plies`, padded with 0.5; `n` = real plies kept. |
| `game_features` | `(cp, side, result, clip) -> ndarray[27] float32` | Order given by `FEATURE_NAMES`. Uses the full sequence. Requires at least 1 ply. |
| `FEATURE_NAMES`, `N_FEATURES` | constants | 27 names; model contract. |

Thresholds: inaccuracy 0.10, mistake 0.20, blunder 0.30 drop in win probability; "perfect" below 0.01. Phases: plies 0 to 19, 20 to 59, 60+ (an empty phase falls back to the player's overall mean loss). Ply `t` is White's move when `t` is even.

### 4.2 `prepare.py`

| Function | Contract |
|---|---|
| `parse_games(path)` | Splits a shard on the `"\n\n\n"` terminator, drops the trailing partial block, yields `(white_elo, black_elo, cp ndarray[int32], result)`; yields `(None, None, None, None)` for a block with missing `Evals`, missing or non-integer Elo, or a non-integer eval. |
| `process_shard(path, max_plies, min_plies, clip)` | Returns `(arrays, counts)`. Arrays: `wp [N,L] float32`, `length [N] int16`, `side [N] int8`, `rating [N] float32`, `feat [N,27] float32`, two samples per game (White then Black). Counts: `games_seen`, `skipped_bad`, `skipped_short` (< `min_plies`), `truncated` (> `max_plies`). |
| `prepare_split(ray, in_dir, ...)` | One Ray task per shard; results concatenated in sorted-shard order (deterministic). Raises `FileNotFoundError` with no shards. |

Outputs: `data/processed/{train,val,test}.npz`, `stats.json` (`max_plies`, `rating_mean`, `rating_std`, `feature_names`, `feat_mean[27]`, `feat_std[27]`, train split only), `metrics/prepare.json`.

### 4.3 `model.py`

| Item | Contract |
|---|---|
| `make_inputs(wp [B,L], side [B], length [B]) -> (x [B,6,L], mask [B,L])` | Channels: centred wp, delta×5, own loss×5, opponent loss×5, own-move flag, mask. All channels are 0 on padding. |
| `RatingCNN(...)` | `forward(wp, side, length, feats) -> [B]` standardised rating. Raises `ValueError` if `use_features` and `feats is None`, if `len(channels) != len(dilations)`, or if `kernel` is even. Buffers `feat_mean`, `feat_std`, `feat_mask` are saved with the weights; `set_feature_stats` replaces std below 1e-6 by 1. |
| `build_model(cfg)` | Builds from the `train` section of `params.yaml`. |

### 4.4 `pyfunc_model.py`

`RatingPyfunc.load_context` reads artifacts `config` (JSON: `model`, `rating_mean`, `rating_std`, `max_plies`, `feature_names`, `data`) and `weights` (state dict). `predict` implements section 3.2 and returns `DataFrame[predicted_rating]` = `z * rating_std + rating_mean`, rounded to 0.1.

### 4.5 `Stockfish_train_eval.py`

Input: PGN shards. Output: same-named shards where each game is

```
[WhiteElo "1500"]
[BlackElo "1480"]
[Result "1-0"]
[Evals "20,15,30,..."]

1. e4 e5 ... 1-0


```

(terminated by `"\n\n\n"`). Checkmate scores ±`mate_score` for the side that delivered mate; stalemate and insufficient material score 0. Games with a `FEN` header or no moves are skipped. Failed games are logged and retried on the next run. Metrics JSON: `stockfish`, `depth`, `hash_mb`, `cache_plies`, `games`, `positions`, `failed_games_last_run`, `seconds`, `positions_per_sec`, `games_per_sec`, `git_commit` (accumulated across resumed runs).

### 4.6 `register_model.py`

Selects by `best_val_mae` among FINISHED runs with `train_samples >= 100000`; re-scores the test set through the pyfunc model; passes if serving MAE is within 1.0 of logged `test_mae`, robustness checks pass (unknown result gives finite output; invalid side and empty evals are rejected; output in 0 to 4000) and MAE beats the mean baseline. Writes `reports/model_test_report.json` (`run_id`, `model_uri`, `validation_mae`, `logged_test_mae`, `retested{samples, mae, rmse, within_100, within_200, baseline_mae}`, `bands[{band, samples, mae, bias}]`, `checks`, `parity_ok`, `passed`). Exit code 2 on failure. Sets tags `val_mae`, `serving_test_mae`, `source_run_id`, `git_commit`, `test_samples` on the new model version and moves alias `champion`.

## 5. Data contracts on disk

| Path | Format | Producer | Consumer |
|---|---|---|---|
| `dataset_clean/{split}/*.pgn` | PGN, headers `WhiteElo`, `BlackElo`, `Result` | `clean_data.py` | evaluate stage |
| `dataset_clean/{split}_evaluated/*.pgn` | section 4.5 | evaluate stage | prepare, `register_model.py` |
| `data/processed/*.npz`, `stats.json` | section 4.2 | prepare | train, backend drift check |
| `models/best.pt`, `models/config.json` | torch state dict, JSON | train | pyfunc artifacts |
| `metrics/*.json` | JSON | each stage | DVC, Pipeline Console |
| `reports/model_test_report.json` | section 4.6 | `register_model.py` | test report |

## 6. Prometheus metrics

Defined in `api/metrics.py` and `api/drift.py`, scraped from `GET /metrics`.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `api_requests_total` | counter | `method`, `endpoint`, `status` | HTTP requests. `endpoint` is the route template (`/predict`), or `unmatched` for unknown routes, so labels stay bounded. |
| `api_request_latency_seconds` | histogram | `endpoint` | Request latency (buckets 10 ms to 120 s). |
| `api_requests_in_progress` | gauge | | Requests being handled now. |
| `api_errors_total` | counter | `endpoint`, `error_type` | Requests ending in an error; `error_type` is the error `code` (section 2.1). |
| `pgn_plies` | histogram | | Half-moves per submitted game. |
| `predicted_rating` | histogram | `side` | Distribution of predicted ratings (400 to 3200). |
| `model_request_seconds` | histogram | | Time spent in the model service. |
| `stockfish_analysis_seconds` | histogram | | Stockfish time for one whole game. |
| `stockfish_position_seconds` | histogram | | Stockfish time per position. |
| `stockfish_positions_total` | counter | | Positions searched. |
| `stockfish_engine_failures_total` | counter | | Engine crashes or hangs. |
| `stockfish_engine_restarts_total` | counter | | Engines restarted after a failure. |
| `stockfish_engines_available` | gauge | | Idle engines in the pool. |
| `feature_drift_shift` | gauge | `feature` | Window mean minus training mean, in training standard deviations, for each of the 27 features. |
| `feature_drift_max` | gauge | | Largest absolute `feature_drift_shift`. |
| `predicted_rating_shift` | gauge | | Mean predicted rating of the window minus the training mean rating, in training standard deviations. |
| `drift_window_size` | gauge | | Player-samples in the drift window (2 per game when both players are rated; maximum 200). |

The drift gauges are published once the window holds at least 30 samples.

Alert rules (`monitoring/alerts.yml`; visible in Prometheus, no Alertmanager):

| Alert | Expression | For |
|---|---|---|
| ApiDown | `up{job="chess-api"} == 0` | 1 m |
| HighErrorRate | 5xx share of `api_requests_total` over 5 m `> 0.05` | 5 m |
| NoStockfishEngines | `stockfish_engines_available == 0` | 1 m |
| SlowPredictions | p95 of `api_request_latency_seconds{endpoint="/predict"}` `> 30` s | 5 m |
| InputDrift | `feature_drift_max > 1` | 15 m |

## 7. Error handling and logging

* **Input errors** never produce a 500: `pgn_utils.parse_pgn` turns every problem into a `BadPGNError` (422) with a message for a non-technical user; schema problems become 422 `invalid_request`.
* **Engine errors** (`api/engine.py`): a crashed or hung engine is discarded and replaced lazily on next use, and the game is analysed once more on a fresh engine (2 attempts). Pool exhaustion gives 503 with `Retry-After`; an over-long game gives 504 and the engine goes back to the pool unharmed.
* **Model service errors** (`api/model_client.py`): connection failures, non-200 answers, malformed bodies and a wrong number of predictions all become 502 `model_unavailable`.
* **Monitoring never breaks a prediction**: the drift monitor catches and logs its own errors.
* **Last line of defence**: middleware turns any unexpected exception into 500 `internal_error`, logs the traceback, and still records the request in the metrics.
* **Logging** (`api/logging_setup.py`): one line per event, format `%(asctime)s %(levelname)s [request_id] logger: message`, to stdout and optionally a rotating file (`LOG_FILE`). Every request is logged with method, path, status and duration (health, ready and metrics probes are not, to keep the log readable); `/predict` also logs the parsed game, the predictions and the time split between Stockfish and the model. 4xx are logged as WARNING, 5xx as ERROR. Pipeline scripts log with `%(asctime)s %(levelname)s %(message)s`.


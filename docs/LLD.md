# Low-Level Design: Chess Rating Estimator

> Mandatory document 3 of 5: endpoint definitions and I/O specifications.
> Version 1.0 (final draft), 2026-11-14.
> **[to confirm]** marks details that depend on `api/` and `frontend/` code not yet reviewed. Fill them in from `api/schemas.py`, `api/main.py`, `api/ops.py` and `api/metrics.py`, or from the live OpenAPI page at http://localhost:8000/docs (`curl -s localhost:8000/openapi.json | python -m json.tool` gives the exact schemas).

## 1. Services and configuration

| Service | Base URL (inside compose) | Configured by |
|---|---|---|
| frontend | `http://frontend:8501` | `API_URL=http://backend:8000` |
| backend | `http://backend:8000` | `MODEL_MODE=rest`, `MLFLOW_URL=http://model-server:5001/invocations`, `MLFLOW_TRACKING_URL=http://mlflow:5000`, `PROJECT_ROOT=/app`, `ENGINE_POOL_SIZE=2`, `DRIFT_STATS_PATH=/app/data/processed/stats.json`, `MLFLOW_UI_URL`, `GRAFANA_URL`, `PROMETHEUS_URL` (browser-facing links) |
| model-server | `http://model-server:5001` | registry alias `models:/chess-rating@champion` |
| mlflow | `http://mlflow:5000` | `sqlite:////mlflow-data/mlflow.db`, artifacts `/mlflow-data/mlruns` |
| prometheus | `http://prometheus:9090` | `monitoring/prometheus.yml`, `monitoring/alerts.yml` |
| grafana | `http://grafana:3000` | `monitoring/grafana/provisioning`, `monitoring/grafana/dashboards` |

The frontend's only dependency on the rest of the system is `API_URL`.

## 2. Backend REST API (FastAPI, port 8000)

All bodies are JSON (`Content-Type: application/json`). Interactive docs at `/docs`.

### 2.1 `GET /health`

Liveness and readiness. Used by the compose healthcheck and the frontend's sidebar status.

| | |
|---|---|
| Request | none |
| 200 | Service is up. Body **[to confirm]**, e.g. `{"status": "ok", ...}` with engine and model-server status. |
| 5xx / no answer | Service down; the frontend shows the red *service* message. |

### 2.2 `POST /predict`

Estimate both players' ratings from one PGN game.

**Request**

```json
{"pgn": "[White \"A\"]\n[Black \"B\"]\n[WhiteElo \"1500\"]\n[BlackElo \"1480\"]\n[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 ... 1-0"}
```

| Field | Type | Required | Rules |
|---|---|---|---|
| `pgn` | string | yes | One standard-chess game in PGN. Headers optional; `Result` used if present (`1-0`, `0-1`, `1/2-1/2`, otherwise unknown). Games with a `FEN` start position are not supported. Size and minimum-length limits **[to confirm]**. |

**Processing**

1. Parse with `python-chess`; reject if no legal mainline.
2. Borrow a Stockfish engine from the pool; evaluate the position after every ply at the configured depth, White's view, mates mapped to ±10000 cp (same rules as `Stockfish_train_eval.py`).
3. Build one record per player, `{"evals": [...], "side": "white"|"black", "result": "<Result>"}`, and `POST` both to the model server's `/invocations`.
4. Update the drift window with the 27 features of each record; update Prometheus metrics.
5. Return ratings, per-move evals and move labels (inaccuracy / mistake / blunder; `api/labels.py`).

**Response 200** **[to confirm field names against `api/schemas.py`]**

| Field (expected) | Type | Meaning |
|---|---|---|
| white / black predicted rating | number | Estimated Elo, one decimal. |
| white / black rating from the PGN | integer or null | `WhiteElo` / `BlackElo` headers, if present. |
| per-move evaluations | list of numbers | Centipawns (or win probability) after each ply, for the graph. |
| per-move labels | list | Move classification for the coloured dots. |
| plies, result, engine info, timings | various | Metadata. |

**Errors**

| Status | When | Counted as |
|---|---|---|
| 422 | Body is not JSON or `pgn` is missing / not a string (FastAPI validation). | 4xx |
| 422 | PGN is empty, cannot be parsed, contains an illegal move, or is too short (fewer than 10 half-moves). | `bad_pgn` error |
| 405 | Wrong HTTP method (e.g. `GET /predict`). | `http_405` error |
| 404 | Unknown route. | endpoint label `unmatched` |
| 503 **[to confirm]** | No Stockfish engine available, engine crashed, or model server unreachable. The engine is restarted. | 5xx; drives HighErrorRate / NoStockfishEngines |

Application errors (such as `bad_pgn`) use this body:

```json
{"error": {"code": "bad_pgn",
           "message": "The game is too short: it has 0 half-moves and the model needs at least 10."},
 "request_id": "..."}
```

`code` is machine-readable and is the label on the error metric; `message` is written for the end user and is shown by the UI as is. Examples of `message`: "The PGN is empty. Paste the full text of one game."; "The game contains an invalid or illegal move: illegal san: 'Ke3' in ...". Request-schema violations caught by FastAPI itself (missing `pgn`, wrong type) keep FastAPI's `{"detail": [...]}` body **[to confirm]**.

### 2.3 `GET /metrics`

Prometheus text exposition format (`text/plain; version=0.0.4`). See section 6.

### 2.4 Pipeline Console endpoints **[to confirm]**

`api/ops.py` serves the data for the Pipeline Console page: DVC stages from `dvc.yaml` / `dvc.lock`, stage metrics from `metrics/*.json`, and MLflow runs from `MLFLOW_TRACKING_URL`, plus the browser-facing links `MLFLOW_UI_URL`, `GRAFANA_URL`, `PROMETHEUS_URL`. List each route here with its response fields.

| Method | Path | Response |
|---|---|---|
| GET | `/ops/...` **[to confirm]** | |

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

Confirmed names:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `api_requests_total` | counter | `status`, endpoint **[to confirm]** | Requests by endpoint and status code. |
| `api_request_latency_seconds` | histogram | `endpoint` | Request latency. |
| `stockfish_engines_available` | gauge | | Idle engines in the pool. |
| `feature_drift_max` | gauge | | Largest per-feature shift of the rolling window from the training mean, in training standard deviations, capped at 3.0. |

Also on the dashboard, names **[to confirm from `api/metrics.py`]**: errors by type (`bad_pgn`, `http_405`), predictions total, Stockfish seconds per game and positions per second, engine failures and restarts, predicted rating by side, model-service latency, requests in progress, submitted game length, per-feature drift, predicted-rating shift, drift window fill.

Alert rules (`monitoring/alerts.yml`):

| Alert | Condition |
|---|---|
| ApiDown | backend scrape target down |
| HighErrorRate | 5xx share of requests above 0.05 |
| NoStockfishEngines | `stockfish_engines_available` at 0 |
| SlowPredictions | /predict p95 latency above 30 s |
| InputDrift | `feature_drift_max` above 1 |

## 7. Error handling and logging

* Bad user input never produces a 500: PGN errors become a 4xx with a message the UI shows in red.
* Engine errors: the engine is restarted and returned to the pool; the request fails with a 5xx and is counted.
* Model-server errors and timeouts become a 5xx with a clear message.
* Logging: `api/logging_setup.py` **[to confirm format, e.g. JSON lines with request id, endpoint, status, latency]**. Pipeline scripts log with `%(asctime)s %(levelname)s %(message)s`.

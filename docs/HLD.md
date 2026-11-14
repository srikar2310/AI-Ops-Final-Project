# High-Level Design: Chess Rating Estimator

> Mandatory document 2 of 5: design choices and rationale.
> Version 1.0 (final draft), 2026-11-14. Items marked **[to confirm]** depend on code not yet reviewed.
> Architecture diagram and block descriptions: see [architecture.md](architecture.md).

## 1. Problem statement and goals

**Problem.** Given one chess game, estimate the playing strength (Elo) of each player from the quality of their moves alone.

**Who uses it.** Chess players and coaches who want a quick strength estimate for a game (for example an over-the-board game, or an unrated opponent), without any chess-engine knowledge.

**Model metrics** (measured on the held-out test set of 20,000 games / 40,000 player-samples):

| Metric | Why |
|---|---|
| Mean absolute error (MAE) in rating points | Primary metric; directly interpretable ("off by N points on average"). |
| RMSE | Penalises large misses. |
| Share of predictions within 100 and 200 points | User-facing notion of "close enough". |
| MAE and bias per 200-point rating band | Detects regression toward the mean at the extremes. |
| Improvement over the predict-the-mean baseline | Proves the model learned something. |

These metrics describe the model; they are tracked in MLflow and reported, but they are not release acceptance criteria. The project's acceptance criteria are about the system working correctly (serving parity, robustness, latency, error handling, monitoring, drift detection, data integrity) and are listed in [test_plan.md](test_plan.md#6-acceptance-criteria).

## 2. Data

| Item | Choice | Rationale |
|---|---|---|
| Source | Lichess open database, January 2017 standard rated games | Public, large, real ratings on one consistent rating system. |
| Sample | About 100k train / 20k val / 20k test games | Brief asks for about 100k games; enough for a ~96k-parameter model. |
| Filtering | Rated games only, no bullet (base + 40 × increment < 180 s), more than 10 moves | Bullet and very short games say little about strength. |
| Rating balance | Games are accepted into 100-point bins (500 to 2500) up to a normal-shaped target (mean 1800, sd 220) | The raw dump is heavily concentrated around 1500; balancing gives the model enough examples at both ends. 2500 cap because 2017 Lichess has few higher-rated players. |
| Evaluations | All games re-evaluated by local Stockfish rather than using Lichess's embedded `[%eval]` | Only a minority of games carry evals and they come from various engine versions and depths. One engine, one depth for every game gives consistent inputs, and the same engine runs at serving time. |
| Split | Random split at the game level during sampling | Each game, and therefore both its samples, is in exactly one split. Split by player was not done; see limitations. |

## 3. Pipeline design

The pipeline is a DVC DAG of three stages, `evaluate` (per split), `prepare`, `train`, plus a registration step. `dvc repro` rebuilds only what changed; `dvc.lock` records the hash of every input and output, and `params.yaml` holds every tunable value, so DVC re-runs a stage when one of its parameters changes.

| Decision | Rationale |
|---|---|
| Stockfish evaluation with a `multiprocessing` pool, one single-threaded engine per worker | Engine search is CPU-bound and does not parallelise well inside one game; many independent single-thread engines give near-linear scaling. Resumable output means a crash costs minutes, not hours. |
| Opening cache (first 16 plies) | Openings repeat a lot; caching cuts engine time without changing results beyond move 16. |
| Feature preparation on Ray (one task per shard) | Meets the "Ray / Spark / K8s" requirement; shards are independent, so tasks parallelise cleanly, and the Ray dashboard shows progress. `process_shard` is a plain function, so it is unit-testable without Ray. |
| Throughput logged per stage (`metrics/eval_*.json`, `metrics/prepare.json`) | Required for the "speed and throughput" criterion, and shown on the Pipeline Console. |

## 4. Model design

**Inputs per player-sample**

1. Sequence: White's win probability after every ply, converted from centipawns with the Lichess formula `1 / (1 + exp(-0.00368208 * cp))`, clipped at ±1500 cp, seen from the player's side, padded to 200 plies. `model.make_inputs` expands it to 6 channels (centred win probability, change since previous ply, own losses, opponent losses, own-move flag, padding mask).
2. 27 game-level features (`features.FEATURE_NAMES`): own and opponent loss statistics (mean, std, median, inaccuracy / mistake / blunder / perfect-move rates at 0.10 / 0.20 / 0.30 / 0.01 win-probability drop), own loss by game phase (opening plies 0 to 19, middlegame 20 to 59, endgame 60+), conversion features (max, min, final win probability, time spent winning or losing, blown wins, saved losses), result, and game length.

**Network.** Four dilated 1D conv blocks (32, 64, 64, 128 channels; kernel 5; dilations 1, 2, 4, 8) with masked average and max pooling, concatenated with a 64-unit MLP over the normalised features, then a 64-unit head. About 96k parameters. Target is the standardised rating; Huber loss; AdamW; ReduceLROnPlateau; early stopping on validation MAE (patience 6).

| Decision | Rationale |
|---|---|
| Win probability rather than raw centipawns | A 300 cp drop matters a lot in an equal position and hardly at all when already +1000. Win probability matches how strength actually shows. |
| CNN over the sequence plus hand-made features | The features carry the classic signal (ACPL-style accuracy, blunder rates); the CNN picks up patterns over time (when mistakes happen, how advantages are converted). The hybrid beat the CNN-only runs on validation MAE (185.8 vs 187.2). |
| Dilated convolutions instead of a GRU | Receptive field of about 60 plies with few parameters, trains fast on GPU and is fully parallel at inference. |
| Normalisation statistics stored as model buffers | Serving cannot use different statistics from training. |
| Two samples per game | Each player gets an independent estimate; the model is told which side it rates. |

**Model selection and promotion.** Ranked by validation MAE only; the test set is used once per run for reporting. `register_model.py` re-tests the chosen model through the serving wrapper and promotes it by moving the `champion` alias only if all checks pass and it beats the current champion.

**Current champion** (registry version 1, run `eab43e95`, git commit `f389325`):

| Metric | Value |
|---|---|
| Validation MAE | 185.8 |
| Test MAE | 182.8 |
| Predict-the-mean baseline MAE | 232.6 (model is 21.4% better) |
| Test RMSE | 230.4 |
| Within 100 points | 34.0% |
| Within 200 points | 62.1% |
| Serving-parity MAE (re-measured through the pyfunc wrapper) | 182.8 |

## 5. Serving design

| Decision | Rationale |
|---|---|
| Model served by `mlflow models serve` from the registry alias `models:/chess-rating@champion` | Meets the "MLflow for APIfication" criterion; promotion and rollback are an alias move, no code change. |
| Separate FastAPI backend in front of the model server | The model server only knows numbers. PGN parsing, Stockfish, validation, error handling and metrics belong in an application layer the UI can call. |
| Stockfish pool inside the backend (2 engines) | Starting an engine per request costs about a second; a pool reuses warm engines and bounds CPU use. |
| Streamlit frontend talking only to `API_URL` | Required loose coupling; the UI can be redeployed or replaced without touching inference. |
| docker compose with six services | One command brings up the whole stack identically on any machine; each service has a healthcheck and dependencies wait for health. |

**Rollback.** Point the `champion` alias at the previous version in the MLflow UI (or `MlflowClient().set_registered_model_alias`) and restart `model-server`.

## 6. Monitoring design

* **Instrumentation.** The backend exports Prometheus metrics: request count by endpoint and status, request latency histogram, errors by type, Stockfish time per game and throughput, idle engines, predicted-rating distribution by side, model-service latency, submitted game length, and drift gauges. Exact metric names are listed in [LLD.md](LLD.md#6-prometheus-metrics).
* **Dashboard.** A provisioned Grafana dashboard shows all of the above in near real time.
* **Alerts.** ApiDown, HighErrorRate (5xx ratio above 5%), NoStockfishEngines, SlowPredictions (p95 above 30 s), InputDrift (`feature_drift_max` above 1).
* **Drift.** The backend keeps a rolling window of the features of recently submitted games (100 games) and compares the window mean of each of the 27 features with the training mean, in units of training standard deviations (`stats.json`). The largest shift is exported as `feature_drift_max`, capped at 3.0. Sustained drift above 1 fires InputDrift, and the retraining pipeline (`scripts/retrain.sh`) **[to confirm trigger mechanism]** re-runs `dvc repro` and `register_model.py`.

## 7. Non-functional requirements

| Area | Target / approach |
|---|---|
| Latency | A typical 40-move game is analysed in about 1 s at p95 on a laptop (measured: /predict p95 about 0.8 s; Stockfish p95 about 1 s per game). |
| Reliability | Bad input returns a clear 4xx, never a 500; an engine crash is caught, the engine restarted, and the request fails cleanly. |
| Reproducibility | Every run maps to a git commit plus an MLflow run ID; data maps to a `dvc.lock` hash. |
| Environment parity | Docker images for every service; MLproject plus `python_env.yaml` for training. |
| Security | No user data is stored; Grafana admin password set by environment; services bind only on localhost ports in development. No secrets in the repo. |
| Scalability | Backend is stateless apart from the drift window; engine pool size and backend replicas can be raised. |

## 8. Known limitations and risks

These are known and accepted for this release; each is also listed in the test report.

1. **Engine version mismatch.** Training data was evaluated with Stockfish 16; the backend image installs Stockfish 15.1 from Debian bookworm. Evaluations differ slightly, adding noise at serving time.
2. **Regression toward the mean.** As with most rating regressors, very low and very high ratings are pulled toward 1800: on the test set, players below 1200 are over-estimated by about 400 points on average and players at 2200 or above under-estimated by about 400 (`reports/model_test_report.json`).
3. **Split by game, not by player.** The same player can appear in train and test. With 100k games from a month of Lichess, overlap is small, but it is not ruled out.
4. **One game is little evidence.** Even a perfect model cannot be precise from 30 moves; the UI should present the estimate as a range.
5. **Time control and clock data are ignored.** The model only sees move quality.
6. **Environment drift.** `requirements-train.txt` has an unpinned `mlflow==`; Python is 3.12 in the compose images but 3.11 in `python_env.yaml` and the root Dockerfile.
7. **Ops gaps.** The `mlflow` service inherits the model server's healthcheck and reports unhealthy; the SlowPredictions threshold (30 s) is far above real latency (about 1 s).

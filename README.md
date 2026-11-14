# Chess Rating Estimator

Estimate each player's Elo rating from a single chess game. The game is replayed with Stockfish, and a hybrid 1D CNN plus feature MLP predicts the ratings from the quality of the moves. Test MAE is 182.8 rating points (predict-the-mean baseline: 232.6).

## Quick start

```bash
docker compose up -d --build
docker compose ps          # wait until services are healthy
```

| What | Where |
|---|---|
| App (UI) | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0"}'
```

## Rebuilding the pipeline

```bash
pip install -r requirements-train.txt
dvc repro                   # evaluate -> prepare -> train
python register_model.py    # test the best run and promote it to chess-rating@champion
```

Or with MLflow Projects: `mlflow run . -e train`.

## Tests

```bash
python -m pytest -m "not e2e"                 # unit, integration, data checks: no stack needed
docker compose up -d
python -m pytest -m "e2e and not drift"       # API + end-to-end against the running stack
python -m pytest -m drift                     # drift scenario (1-3 min), run last
```

See [docs/test_plan.md](docs/test_plan.md).

## Documentation

| Document | File |
|---|---|
| Architecture diagram and blocks | [docs/architecture.md](docs/architecture.md) |
| High-level design | [docs/HLD.md](docs/HLD.md) |
| Low-level design (API and I/O) | [docs/LLD.md](docs/LLD.md) |
| Test plan, cases and report | [docs/test_plan.md](docs/test_plan.md), [docs/test_cases.md](docs/test_cases.md), [docs/test_report.md](docs/test_report.md) |
| User manual | [docs/USER_MANUAL.md](docs/USER_MANUAL.md) |

## Repository structure

Only code and configuration are in Git. The datasets, the trained model and the MLflow runs are not: they are versioned with DVC (the small `*.dvc` and `dvc.lock` files record their hashes) and kept in a DVC store outside this repository. To get the data, rebuild it as described in [Rebuilding the dataset from scratch](#rebuilding-the-dataset-from-scratch).

```
.
├── Parse.py                     sample the Lichess dump into rating-balanced train/val/test shards
├── clean_data.py                keep only Elo, result and moves (strip clocks and Lichess evals)
├── Stockfish_train_eval.py      DVC stage `evaluate`: Stockfish eval of every position (multiprocessing, resumable)
├── prepare.py                   DVC stage `prepare`: evals -> arrays + 27 features (Ray, one task per shard)
├── features.py                  feature engineering shared by training AND serving
├── model.py                     hybrid 1D CNN + feature MLP (PyTorch)
├── train.py                     DVC stage `train`: training with MLflow tracking
├── pyfunc_model.py              MLflow pyfunc wrapper used by the model server
├── register_model.py            pick the best run by validation, test it through the wrapper, set alias `champion`
│
├── dvc.yaml, dvc.lock           pipeline DAG (evaluate -> prepare -> train) and pinned hashes
├── params.yaml                  every tunable setting (engine depth, plies, model hyper-parameters)
├── MLproject, python_env.yaml   MLflow Project entry points (prepare, train) and environment
├── requirements.txt             runtime + test dependencies
├── requirements-train.txt       training dependencies (Ray, PyTorch, MLflow)
├── Dockerfile                   image for running the Stockfish evaluation stage
├── docker-compose.yml           the full stack: model-server, mlflow, backend, frontend, prometheus, grafana
├── conftest.py                  pytest setup: repo root on the import path, test markers
├── statement.txt                problem statement
│
├── api/                         FastAPI backend
│   ├── main.py                  app factory, endpoints, error handlers, request middleware
│   ├── schemas.py               request/response models (OpenAPI at /docs)
│   ├── pgn_utils.py             PGN parsing and validation (all user-facing input errors)
│   ├── engine.py                Stockfish engine pool with crash/hang recovery
│   ├── model_client.py          calls the MLflow model server (or loads the model locally)
│   ├── labels.py                per-move inaccuracy / mistake / blunder labels
│   ├── drift.py                 input-drift monitor (rolling window vs training statistics)
│   ├── ops.py                   /ops/* endpoints for the Pipeline Console (DVC, MLflow, live stats)
│   ├── metrics.py               Prometheus metrics
│   ├── errors.py                error types -> HTTP status + code
│   ├── logging_setup.py         logging with a per-request id
│   └── settings.py              settings from params.yaml + environment variables
│
├── frontend/                    Streamlit UI (talks to the backend only through API_URL)
│   ├── app.py                   main page: paste / upload / play a game -> ratings, eval graph, replay
│   ├── api_client.py            the only module that calls the backend
│   ├── pages/
│   │   ├── 1_Pipeline_Console.py   DVC DAG, MLflow runs, live service stats, tool links
│   │   └── 2_Help.py               in-app user manual
│   ├── components/board/        click-to-move chessboard (HTML/JS component)
│   ├── components/replay/       game replay viewer (HTML/JS component)
│   └── requirements.txt
│
├── docker/                      Dockerfile.backend, Dockerfile.frontend, Dockerfile.model + their requirements
├── monitoring/
│   ├── prometheus.yml           scrape config
│   ├── alerts.yml               ApiDown, HighErrorRate, NoStockfishEngines, SlowPredictions, InputDrift
│   └── grafana/                 provisioned datasource and the chess-rating dashboard
├── scripts/
│   ├── retrain.sh               retraining pipeline: dvc repro -> register -> export
│   ├── export_model.py          champion model -> serving/model for the model-server image
│   └── test_report.py           pytest JUnit XML -> pass/fail tables for the test report
│
├── tests/                       pytest suites (see docs/test_plan.md)
│   ├── test_features_unit.py, test_features.py      feature engineering
│   ├── test_pgn_parsing.py                          PGN parsing, evaluation stage, real Stockfish
│   ├── test_model_unit.py, test_pyfunc_unit.py      model inputs, network, serving wrapper
│   ├── test_data_and_acceptance.py                  data quality + serving-parity acceptance
│   ├── test_api_additions.py, test_drift.py         API internals and drift monitor
│   ├── test_api_live.py, test_e2e_stack.py          black-box API and full-stack end-to-end tests
│   └── phase7_helpers.py                            shared test helpers
│
└── docs/                        architecture, HLD, LLD, test plan / cases / report, user manual
```

Pipeline outputs that are small enough are committed for reference: `metrics/*.json` (per-stage throughput, failures, training metrics) and `reports/model_test_report.json` (the registration test).

## Rebuilding the dataset from scratch

Anyone can rebuild the data from the public Lichess database. Everything runs from the repository root.

**1. Requirements.** Python 3.11 or newer, `pip install -r requirements-train.txt` (this includes `python-chess`, DVC, Ray, PyTorch and MLflow), `zstd` to decompress the dump, and Stockfish. The training data was evaluated with **Stockfish 16**; install it (e.g. `sudo apt install stockfish` on Ubuntu 24.04, or a binary from stockfishchess.org) and set `eval.engine_path` in `params.yaml` if it is not at `/usr/games/stockfish`.

**2. Download the games.** The project uses the Lichess standard rated games of January 2017:

```bash
wget https://database.lichess.org/standard/lichess_db_standard_rated_2017-01.pgn.zst
zstd -d lichess_db_standard_rated_2017-01.pgn.zst -o lichess_db.pgn
```

`Parse.py` reads a file named **`lichess_db.pgn` in the current folder** (change `input_path` at the top of the script to use another path). The decompressed file is several GB.

**3. Sample and split: `python Parse.py`.** Streams the dump and keeps rated games that are not bullet (base + 40 × increment of at least 180 s) and have more than 10 moves. Games are accepted into 100-point rating bins from 500 to 2500, filled up to a normal-shaped target (mean 1800, sd 220), and assigned to train, val and test at random in the ratio 5:1:1, targeting 100,000 / 20,000 / 20,000 games. It stops once every split reaches 95% of its target, and prints progress every 10,000 games scanned. Output: `dataset/{train,val,test}/*_shard_NNN.pgn` (10,000 games per train shard, 5,000 per val/test shard).

No random seed is set, so a rerun selects a different (but statistically equivalent) set of games and will not reproduce the exact dataset behind the published results; add `random.seed(...)` at the top of `Parse.py` if you need repeatable runs.

**4. Clean: `python clean_data.py`.** Reads `dataset/{train,val,test}/` and writes `dataset_clean/{train,val,test}/` shards that keep only the `WhiteElo`, `BlackElo` and `Result` headers and the moves; all comments (clock times, Lichess's own engine evals) are removed, so every game is evaluated by the same local engine in the next step.

**5. Run the pipeline.**

```bash
rm -rf dataset_clean/*_evaluated          # the evaluate stage resumes into existing output; start clean for new data
dvc add dataset_clean/train dataset_clean/val dataset_clean/test   # record the new data version
dvc repro                                  # evaluate (per split) -> prepare -> train
python register_model.py                   # test the best run and promote it to chess-rating@champion
```

`dvc repro` runs `Stockfish_train_eval.py` for each split (depth 11, one single-threaded engine per worker; workers default to CPU count minus 4), then `prepare.py` on Ray, then `train.py`. The evaluation is the slow part: on a 32-thread machine with 28 workers it ran at about 720 positions per second, which took about 2.8 hours for the 100,000 training games and about 35 minutes each for validation and test (`metrics/eval_*.json`). It can be stopped and restarted without losing work. The prepare stage takes seconds, and training a few minutes on a GPU.

## AI declaration

I used an AI assistant (Claude, by Anthropic) while building this project.

**My own work.** The idea for the project and its overall design are mine: the problem framing (estimating rating from one game's move quality), the choice of data (Lichess games re-evaluated with a local Stockfish, sampled to balance the rating distribution), the model approach (a 1D CNN over the evaluation sequence combined with game-level features), the structure of the repository and the pipeline (DVC stages, Ray, MLflow tracking and registry, FastAPI backend, Streamlit frontend, docker compose, Prometheus and Grafana, drift detection and retraining), and the decisions along the way, including the acceptance criteria. I reviewed the AI-assisted output, ran the code, the pipeline and the tests myself, and I am responsible for everything in this submission.

**Where AI helped.** AI was used to help write the code files that implement this design. In particular, it:

* helped write the source code for the data pipeline, model and training, serving, backend, frontend, Docker and monitoring setup, working from my design;
* reviewed the Docker, docker compose, MLflow and monitoring setup, pointed out problems (for example absolute MLflow artifact paths, Python and Stockfish version mismatches, healthcheck and alert-threshold issues) and suggested how to test them;
* wrote the Phase 7 test suites (`tests/test_features_unit.py`, `test_pgn_parsing.py`, `test_model_unit.py`, `test_pyfunc_unit.py`, `test_data_and_acceptance.py`, `test_api_live.py`, `test_e2e_stack.py`, `phase7_helpers.py`), the root `conftest.py` and `scripts/test_report.py`;
* drafted the documents in `docs/` (architecture, HLD, LLD, test plan, test cases, test report, user manual) and this README, from the project's code and records; the test results in the report come from my own runs;
* added docstrings to the code without changing its behaviour.

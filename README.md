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
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample_request.json
```

## Rebuilding the pipeline

```bash
pip install -r requirements-train.txt
dvc pull                    # data
dvc repro                   # evaluate -> prepare -> train
python register_model.py    # test the best run and promote it to chess-rating@champion
```

Or with MLflow Projects: `mlflow run . -e train`.

## Tests

```bash
pytest -m "not e2e"                       # unit + integration, no stack needed
docker compose up -d && pytest -m e2e     # end-to-end against the running stack
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

## Layout

```
api/         FastAPI backend (PGN parsing, Stockfish pool, drift, metrics)
frontend/    Streamlit UI (analysis, Pipeline Console, Help)
docker/      Dockerfiles for backend, frontend, model server
monitoring/  Prometheus config + alerts, Grafana provisioning + dashboard
scripts/     retrain.sh
tests/       pytest suites
features.py prepare.py model.py train.py pyfunc_model.py register_model.py Stockfish_train_eval.py
dvc.yaml params.yaml MLproject python_env.yaml docker-compose.yml
```

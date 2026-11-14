# Test Cases: Chess Rating Estimator

## Unit: feature engineering (`tests/test_features_unit.py`)

| ID | Description | Expected result |
|---|---|---|
| UT-F-01 | `cp_to_wp(0)` | 0.5 |
| UT-F-02 | `cp_to_wp(100)` against the Lichess formula | 0.5910 ± 0.0001 |
| UT-F-03 | Symmetry and monotonicity over −1400..1400 cp | `wp(x) + wp(−x) = 1`; strictly increasing; in (0, 1) |
| UT-F-04 | Clipping at ±1500 cp | ±10000 cp gives the same value as ±1500 |
| UT-F-05 | `build_sample` padding and length | length 3; padded with 0.5; float32 |
| UT-F-06 | Black's view | complement of White's on real plies; padding stays 0.5 |
| UT-F-07 | Truncation of a 300-ply game | 200 values, length 200 |
| UT-F-08 | Feature vector contract | 27 unique names; shape (27,), float32, finite |
| UT-F-09 | Perfect game (all evals 0) | no losses, blunder rate 0, perfect rate 1, length 0.40 |
| UT-F-10 | Blunder attributed to the correct player | White's own blunder rate 1/10, Black's opponent blunder rate 1/10, others 0 |
| UT-F-11 | Threshold ordering | inaccuracy ≥ mistake ≥ blunder rates |
| UT-F-12 | Gain on the first move | not counted as a loss |
| UT-F-13 | Result features (9 variants: 1-0, 0-1, draws, unknown, empty, None, both sides) | correct score and known flag |
| UT-F-14 | Short game phase fallback | middlegame and endgame loss equal overall mean |
| UT-F-15 | Blown win / saved loss | White blown_win = 1, Black saved_loss = 1 |
| UT-F-16 | Features use the full sequence | 250-ply game gives length 2.5 |
| UT-F-17 | Empty evals | raises (callers must guard) |
| UT-F-18 | Mate scores (±10000) | finite features |

## Unit: PGN parsing (`tests/test_pgn_parsing.py`)

| ID | Description | Expected result |
|---|---|---|
| UT-P-01 | `parse_games` on a shard with valid, no-Evals, bad Elo, bad eval, short, long blocks and a partial tail | 6 rows; invalid blocks give `None`s; partial tail ignored |
| UT-P-02 | Missing `Result` header | result `*` |
| UT-P-03 | Empty shard | no rows |
| UT-P-04 | `process_shard` counts | seen 6, bad 3, short 1, truncated 1 |
| UT-P-05 | `process_shard` arrays | 4 samples; dtypes and shapes per LLD; sides 0,1,0,1; padding 0.5 |
| UT-P-06 | Arrays equal `features.py` output (train/serve parity) | exact equality; long game's features use all 250 plies |
| UT-P-07 | Shard with no usable games | empty arrays of the right shape |
| UT-P-08 | `prepare_split` with shards written out of order | concatenated in sorted shard order; counts aggregated |
| UT-P-09 | `prepare_split` on an empty folder | `FileNotFoundError` |
| UT-P-10 | `load_done` with a crash-truncated output file | partial game removed on disk; finished games returned |
| UT-P-11 | `load_done` with no file | empty set |
| UT-P-12 | `task_stream` filtering and resume | FEN game, empty game and already-done game skipped; 1 task with correct UCI moves and headers |
| UT-P-13 | `task_stream` limit | stops after 2 |
| UT-P-14 | `write_meta` across resumed runs | games and positions accumulate; throughput recomputed |

## Integration: engine (`tests/test_pgn_parsing.py`, marker `engine`)

| ID | Description | Expected result |
|---|---|---|
| IT-P-01 | Evaluate Fool's mate with real Stockfish | 4 evals; last is −10000 |
| IT-P-02 | `eval_game` with an illegal move | returns an error record, does not raise |
| IT-P-03 | Engine output read by `process_shard` | 1 game, 0 bad, 2 samples with correct lengths and ratings |

## Unit: model (`tests/test_model_unit.py`)

| ID | Description | Expected result |
|---|---|---|
| UT-M-01 | `make_inputs` shapes | x [B, 6, L], mask [B, L] |
| UT-M-02 | Padding | all 6 channels 0 after `length` |
| UT-M-03 | Own-move channel | alternates by side |
| UT-M-04 | Loss channels | own drop on own ply, opponent drop on opponent ply, scaled ×5, never negative |
| UT-M-05 | Centred channel | 0.75 → 0.5, 0.25 → −0.5 |
| UT-M-06 | Forward pass | shape [B], finite |
| UT-M-07 | Changing padded values | output unchanged |
| UT-M-08 | Eval mode, one sample vs batch | identical outputs |
| UT-M-09 | Invalid architecture (2 variants) | `ValueError` |
| UT-M-10 | Features missing when enabled | `ValueError` |
| UT-M-11 | CNN-only model | works without features |
| UT-M-12 | Feature stats | std 0 replaced by 1; buffers saved and reloaded with weights |
| UT-M-13 | Feature mask | masked feature has no effect |
| UT-M-14 | `build_model` from `params.yaml` | blocks match config |

## Unit and integration: serving wrapper (`tests/test_pyfunc_unit.py`)

| ID | Description | Expected result |
|---|---|---|
| UT-S-01 | Side aliases (10 variants) | white/W/0 → 0, black/b/1 → 1 |
| UT-S-02 | Invalid side (4 variants) | `ValueError` |
| UT-S-03 | Output contract | one column `predicted_rating`, one row per record, finite, rounded to 0.1 |
| UT-S-04 | Missing `result` column | same as result `*` |
| UT-S-05 | Empty evals | `ValueError` "at least one ply" |
| UT-S-06 | Invalid side in a request | `ValueError` |
| UT-S-07 | Wrapper vs direct `features.py` + model + de-standardisation | equal within 0.05 |
| UT-S-08 | Determinism and order | reversed input gives reversed output |
| UT-S-09 | 400-ply game | finite prediction |
| IT-S-01 | MLflow `save_model` → `load_model` → predict | identical to the in-memory wrapper |

## Data quality (`tests/test_data_and_acceptance.py`, marker `data`)

| ID | Description | Expected result |
|---|---|---|
| DQ-01 | Each split's `.npz` (3 variants) | consistent shapes; 2 samples per game, White then Black; lengths within min/max plies; wp in [0, 1]; finite features; ratings 300 to 3300 |
| DQ-02 | `stats.json` | feature names equal `FEATURE_NAMES`; 27 means and stds |
| DQ-03 | Normalisation from train only | `rating_mean` equals train mean |
| DQ-04 | Leakage | no game of 30+ plies in both train and test |
| DQ-05 | Evaluate stage metrics (3 variants) | 0 failed games; positive throughput |
| DQ-06 | Prepare stage metrics | 2 samples per used game; ≤ 1% bad games |

## Acceptance (`tests/test_data_and_acceptance.py`, marker `data`)

| ID | Description | Expected result |
|---|---|---|
| AC-01 | Serving parity: MAE re-measured through the pyfunc wrapper vs MAE logged at training | gap ≤ 1.0 |
| AC-02 | Robustness checks recorded by `register_model.py` | all pass |

## API (`tests/test_api_live.py`, marker `e2e`)

| ID | Description | Expected result |
|---|---|---|
| API-01 | `GET /health` | 200 |
| API-02 | OpenAPI document | `POST /predict` documented |
| API-03 | Predict `sample_request.json` | 200; ratings 100 to 3500; under 5 s |
| API-04 | Real Lichess export with clocks, comments, `?!` | 200 |
| API-05 | 160-ply legal game, unknown result | 200 |
| API-06 | Movetext without headers | 200 |
| API-07 | 4 concurrent requests (pool of 2 engines) | all 200 |
| API-08 | Bad PGN (5 variants: garbage, illegal move, empty, whitespace, no moves) | 4xx with a `detail` message |
| API-09 | Schema violations (3 variants) | 422 |
| API-10 | Non-JSON body | 4xx |
| API-11 | `GET /predict`, unknown route | 405, 404 |
| API-12 | `/metrics` | request counter, latency histogram, engine gauge, drift gauge present |
| API-13 | Request counting | counter rises after a bad request |
| API-14 | Engine pool after a request | at least 1 idle engine |

## End to end (`tests/test_e2e_stack.py`, marker `e2e`)

| ID | Description | Expected result |
|---|---|---|
| E2E-01 | Health of backend, model-server, mlflow, prometheus, grafana, frontend (6 variants) | 200 each |
| E2E-02 | Model server `POST /invocations` | 2 finite predictions |
| E2E-03 | Model server, bad side | ≥ 400 |
| E2E-04 | Registry alias `chess-rating@champion` | exists |
| E2E-05 | UI flow: example game to backend; UI page loads | 200, 200 |
| E2E-06 | Prometheus targets | all up |
| E2E-07 | Alert rules | ApiDown, HighErrorRate, NoStockfishEngines, SlowPredictions, InputDrift loaded |
| E2E-08 | Healthy stack | no alert firing (except InputDrift) |
| E2E-09 | Request visible in Prometheus | within 60 s |
| E2E-10 | Grafana | at least 1 dashboard; Prometheus datasource |
| E2E-11 | Drift scenario (marker `drift`): 20 × 19-ply opening, then 40 real test games | drift rises above 1; then falls |

## Manual UI cases (run in a browser at http://localhost:8501)

| ID | Steps | Expected result |
|---|---|---|
| UI-01 | Open the app | Main page loads with no errors; sidebar shows the service as up |
| UI-02 | Click *Insert example game*, then *Analyse game* | Ratings for both players shown next to the file's ratings; graph and replay shown |
| UI-03 | Paste a game exported from Lichess with clocks and comments | Green message; analysis works |
| UI-04 | Paste random text | Red message explaining what to fix; no crash, no stack trace |
| UI-05 | Paste a game with an illegal move | Red message naming the problem |
| UI-06 | Upload a `.pgn` with several games, pick the second | That game is analysed |
| UI-07 | Upload an empty file, and a non-PGN file | Clear red message |
| UI-08 | Enter a game on the board, use *Undo*, set the result, analyse | Works; undo removes the last move |
| UI-09 | Check the graph | Blue, orange and red dots on inaccuracies, mistakes and blunders |
| UI-10 | `docker compose stop backend`, reload the page | Red *service* message in the sidebar; no crash. Restart with `docker compose start backend` |
| UI-11 | Narrow the window to phone width (or use the browser's device mode) | Layout stays usable, no horizontal scrolling |
| UI-12 | Open Pipeline Console | DVC stages, MLflow runs and success/failure counts shown; Grafana and MLflow links open |

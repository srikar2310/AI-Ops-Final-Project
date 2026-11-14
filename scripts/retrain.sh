#!/usr/bin/env bash
# Retrain pipeline: rebuild with DVC, test + register the best model, redeploy it.
# Run it when the InputDrift alert fires (or on a schedule).
set -euo pipefail
cd "$(dirname "$0")/.."

dvc repro                          # prepare -> train (evaluate stages rerun only if their inputs changed)
python register_model.py           # picks the best run by VALIDATION MAE, tests it, sets alias 'champion'
python scripts/export_model.py     # champion -> serving/model
dvc add serving/model
git add serving/model.dvc dvc.lock metrics/ serving/model_info.json
echo "Review with 'git status', commit, then: dvc push && docker compose up -d --build model-server"
"""Pick the best training run, TEST it end to end, and register it as the champion.

Steps
  1. Find finished runs in the given MLflow experiments and rank them by a
     VALIDATION metric (never by test, so the test set stays untouched for selection).
  2. Load the best run's servable model (the pyfunc wrapper) exactly as the API
     would, and run it on EVERY game of the evaluated test set.
  3. Check it:
       * parity   - the MAE measured through the serving wrapper matches the test
                    MAE logged during training (catches any train/serve mismatch)
       * robustness - an unknown result still gives a finite prediction; a bad
                    side or empty evals is rejected
  4. Write a test report (overall error, error per rating band) and log it to the run.
  5. Register the model in the MLflow Model Registry and point the alias
     `champion` at it, unless the current champion is already at least as good.

Usage
    python register_model.py                      # test and register
    python register_model.py --dry-run            # test only
    python register_model.py --experiments chess-rating-hybrid --force

Then serve it:
    mlflow models serve -m "models:/chess-rating@champion" --env-manager local -p 5001
"""
import argparse
import json
import logging
import os
import sys

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from prepare import parse_games

log = logging.getLogger("register")

BANDS = [(0, 1200), (1200, 1400), (1400, 1600), (1600, 1800),
         (1800, 2000), (2000, 2200), (2200, 4000)]


# ------------------------------------------------------------------ selection
def find_candidates(experiments, metric, min_train_samples):
    df = mlflow.search_runs(experiment_names=experiments, output_format="pandas")
    if df.empty:
        raise SystemExit(f"No runs found in experiments {experiments}.")
    df = df[df["status"] == "FINISHED"]
    col = f"metrics.{metric}"
    if col not in df.columns:
        raise SystemExit(f"No run has the metric '{metric}'.")
    df = df[df[col].notna()]
    if "params.train_samples" in df.columns:  # drop smoke tests (--max-samples)
        samples = pd.to_numeric(df["params.train_samples"], errors="coerce").fillna(0)
        df = df[samples >= min_train_samples]
    if df.empty:
        raise SystemExit("No finished full-size runs with that metric.")
    return df.sort_values(col).reset_index(drop=True)


def show_table(df, metric):
    wanted = ["run_id", "params.train.channels", "params.train.use_features",
              "params.train.exclude_features", f"metrics.{metric}", "metrics.test_mae"]
    cols = [c for c in wanted if c in df.columns]
    table = df[cols].copy()
    table["run_id"] = table["run_id"].str[:8]
    print("\nCandidate runs (best first, ranked by validation):")
    print(table.rename(columns=lambda c: c.replace("params.train.", "").replace("metrics.", ""))
          .to_string(index=False))


def load_candidate(run_id, experiment_id):
    """Load the servable model of a run; returns (uri, model)."""
    uri = f"runs:/{run_id}/pyfunc_model"
    try:
        return uri, mlflow.pyfunc.load_model(uri)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load %s (%s); looking for a logged model instead", uri, exc)
    try:  # MLflow 3 stores models as separate "logged models"
        for m in mlflow.search_logged_models(experiment_ids=[experiment_id],
                                             output_format="list"):
            if getattr(m, "source_run_id", None) == run_id:
                uri = f"models:/{m.model_id}"
                return uri, mlflow.pyfunc.load_model(uri)
    except Exception as exc:  # noqa: BLE001
        log.warning("Logged-model lookup failed: %s", exc)
    raise SystemExit(f"Could not load the pyfunc model of run {run_id}.")


# -------------------------------------------------------------------- testing
def collect_test_records(test_dir, min_plies):
    """Two records per game (one per player), as the API would send them."""
    records, actual = [], []
    shards = sorted(f for f in os.listdir(test_dir) if f.endswith(".pgn"))
    if not shards:
        raise SystemExit(f"No .pgn shards in {test_dir}.")
    for shard in shards:
        for w_elo, b_elo, cp, result in parse_games(os.path.join(test_dir, shard)):
            if cp is None or len(cp) < min_plies:
                continue
            evals = cp.tolist()
            records.append({"evals": evals, "side": "white", "result": result})
            actual.append(w_elo)
            records.append({"evals": evals, "side": "black", "result": result})
            actual.append(b_elo)
    return records, np.asarray(actual, dtype=float)


def predict_all(model, records, chunk=2000):
    out = []
    for i in range(0, len(records), chunk):
        res = model.predict(pd.DataFrame(records[i:i + chunk]))
        out.append(res["predicted_rating"].to_numpy(dtype=float))
    return np.concatenate(out)


def score(pred, actual, train_mean):
    err = pred - actual
    return {
        "samples": int(len(actual)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "within_100": float((np.abs(err) <= 100).mean()),
        "within_200": float((np.abs(err) <= 200).mean()),
        "baseline_mae": float(np.abs(actual - train_mean).mean()),
    }


def per_band(pred, actual):
    rows = []
    for lo, hi in BANDS:
        sel = (actual >= lo) & (actual < hi)
        if sel.sum() == 0:
            continue
        err = pred[sel] - actual[sel]
        rows.append({"band": f"{lo}-{hi}", "samples": int(sel.sum()),
                     "mae": float(np.abs(err).mean()), "bias": float(err.mean())})
    return rows


def robustness_checks(model, sample):
    """(name, passed) pairs: the model must behave sensibly on awkward input."""
    checks = []

    def run(record):
        return model.predict(pd.DataFrame([record]))["predicted_rating"].iloc[0]

    unknown = dict(sample, result="*")
    try:
        checks.append(("unknown result gives a finite prediction",
                       bool(np.isfinite(run(unknown)))))
    except Exception:  # noqa: BLE001
        checks.append(("unknown result gives a finite prediction", False))

    for name, bad in (("invalid side is rejected", dict(sample, side="green")),
                      ("empty evals are rejected", dict(sample, evals=[]))):
        try:
            run(bad)
            checks.append((name, False))
        except Exception:  # noqa: BLE001
            checks.append((name, True))

    try:
        pred = run(sample)
        checks.append(("prediction is in a sane range (0-4000)", bool(0 <= pred <= 4000)))
    except Exception:  # noqa: BLE001
        checks.append(("prediction is in a sane range (0-4000)", False))
    return checks


# ---------------------------------------------------------------- registration
def current_champion_val(client, name, alias):
    try:
        mv = client.get_model_version_by_alias(name, alias)
        return mv, float(mv.tags.get("val_mae", "inf"))
    except Exception:  # noqa: BLE001 - no such alias yet
        return None, None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["chess-rating-hybrid"],
                    help="only runs made with the CURRENT model code can be loaded")
    ap.add_argument("--metric", default="best_val_mae", help="validation metric (lower = better)")
    ap.add_argument("--name", default="chess-rating")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--test-dir", default="dataset_clean/test_evaluated")
    ap.add_argument("--stats", default="data/processed/stats.json")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--report", default="reports/model_test_report.json")
    ap.add_argument("--min-train-samples", type=int, default=100_000)
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="allowed gap (rating points) between logged and re-measured test MAE")
    ap.add_argument("--force", action="store_true", help="replace the champion even if not better")
    ap.add_argument("--dry-run", action="store_true", help="test only, do not register")
    args = ap.parse_args()

    log.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    df = find_candidates(args.experiments, args.metric, args.min_train_samples)
    show_table(df, args.metric)
    best = df.iloc[0]
    run_id, val_mae = best["run_id"], float(best[f"metrics.{args.metric}"])
    logged_test_mae = float(best["metrics.test_mae"]) if "metrics.test_mae" in df.columns else None
    log.info("Selected run %s (validation MAE %.2f)", run_id, val_mae)

    uri, model = load_candidate(run_id, best["experiment_id"])
    log.info("Loaded %s", uri)

    with open(args.params, "r", encoding="utf-8") as f:
        min_plies = int(yaml.safe_load(f)["data"]["min_plies"])
    with open(args.stats, "r", encoding="utf-8") as f:
        train_mean = float(json.load(f)["rating_mean"])

    records, actual = collect_test_records(args.test_dir, min_plies)
    log.info("Testing on %d samples (%d games) through the serving wrapper",
             len(records), len(records) // 2)
    pred = predict_all(model, records)
    overall = score(pred, actual, train_mean)
    bands = per_band(pred, actual)

    checks = robustness_checks(model, records[0])
    parity_ok = (logged_test_mae is None
                 or abs(overall["mae"] - logged_test_mae) <= args.tolerance)

    print("\n=== Test report ===")
    print(f"MAE {overall['mae']:.1f} (baseline {overall['baseline_mae']:.1f}) | "
          f"RMSE {overall['rmse']:.1f} | within 100: {overall['within_100']:.1%} | "
          f"within 200: {overall['within_200']:.1%}")
    if logged_test_mae is not None:
        print(f"Parity with training: logged test MAE {logged_test_mae:.2f} vs "
              f"re-measured {overall['mae']:.2f} -> {'PASS' if parity_ok else 'FAIL'}")
    print("\nError by true rating band (bias = mean prediction - truth):")
    print(pd.DataFrame(bands).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nRobustness checks:")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    all_ok = parity_ok and all(ok for _, ok in checks) and overall["mae"] < overall["baseline_mae"]
    if overall["mae"] >= overall["baseline_mae"]:
        print("FAIL  model does not beat the predict-the-mean baseline")

    report = {"run_id": run_id, "model_uri": uri, "validation_mae": val_mae,
              "logged_test_mae": logged_test_mae, "retested": overall, "bands": bands,
              "checks": {n: ok for n, ok in checks}, "parity_ok": parity_ok, "passed": all_ok}
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(args.report, "test_report")
        mlflow.log_metrics({"serving_test_mae": overall["mae"]})
    log.info("Report written to %s and logged to run %s", args.report, run_id)

    if not all_ok:
        log.error("Model FAILED testing; not registering.")
        sys.exit(2)
    if args.dry_run:
        log.info("Dry run: tests passed, nothing registered.")
        return

    client = MlflowClient()
    champion, champ_val = current_champion_val(client, args.name, args.alias)
    if champion is not None and champ_val <= val_mae and not args.force:
        log.info("Current champion (version %s, validation MAE %.2f) is at least as good; "
                 "not replacing it. Use --force to override.", champion.version, champ_val)
        return

    mv = mlflow.register_model(uri, args.name)
    tags = {"val_mae": val_mae, "serving_test_mae": round(overall["mae"], 3),
            "source_run_id": run_id, "git_commit": best.get("tags.git_commit", "unknown"),
            "test_samples": overall["samples"]}
    for key, value in tags.items():
        client.set_model_version_tag(args.name, mv.version, key, str(value))
    client.set_registered_model_alias(args.name, args.alias, mv.version)
    log.info("Registered %s version %s and set alias '%s'", args.name, mv.version, args.alias)
    print(f'\nServe it with:\n  mlflow models serve -m "models:/{args.name}@{args.alias}" '
          f"--env-manager local -p 5001")


if __name__ == "__main__":
    main()
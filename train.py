"""Train the 1D CNN rating regressor and track the run with MLflow.

Reads data/processed/{train,val,test}.npz (from prepare.py) and the `train`
section of params.yaml. Every run is tagged with the Git commit and the DVC
hash of the prepared data, so a run ID maps to exactly one code + data version.

    python train.py
    python train.py --max-samples 5000 --epochs 2     # quick smoke test
"""
import argparse
import copy
import json
import logging
import os
import random
import subprocess
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml

from features import FEATURE_NAMES
from model import build_model
from pyfunc_model import RatingPyfunc

log = logging.getLogger("train")


# ----------------------------------------------------------------- utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sh(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def dvc_data_hash(stage="prepare", lock="dvc.lock"):
    """md5 of the prepared-data output recorded in dvc.lock."""
    try:
        with open(lock, "r", encoding="utf-8") as f:
            outs = yaml.safe_load(f)["stages"][stage]["outs"]
        return ",".join(o["md5"] for o in outs)
    except Exception:
        return "unknown"


def load_split(path, device, max_samples=0):
    z = np.load(path)
    n = len(z["rating"]) if not max_samples else min(max_samples, len(z["rating"]))
    return {
        "wp": torch.from_numpy(z["wp"][:n]).float().to(device),
        "side": torch.from_numpy(z["side"][:n]).long().to(device),
        "length": torch.from_numpy(z["length"][:n]).long().to(device),
        "rating": torch.from_numpy(z["rating"][:n]).float().to(device),
        "feat": torch.from_numpy(z["feat"][:n]).float().to(device),
    }


def predict_z(model, data, batch_size=2048):
    """Standardised predictions for a whole split."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data["rating"]), batch_size):
            s = slice(i, i + batch_size)
            out.append(model(data["wp"][s], data["side"][s], data["length"][s],
                             data["feat"][s]))
    return torch.cat(out)


def score(pred, true):
    """Metrics in rating points."""
    err = pred - true
    return {
        "mae": err.abs().mean().item(),
        "rmse": err.pow(2).mean().sqrt().item(),
        "within_100": (err.abs() <= 100).float().mean().item(),
        "within_200": (err.abs() <= 200).float().mean().item(),
    }


# ---------------------------------------------------------------------- main
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--metrics", default="metrics/train.json")
    ap.add_argument("--experiment", default="chess-rating-cnn")
    ap.add_argument("--max-samples", type=int, default=0, help="smoke test on a subset")
    ap.add_argument("--epochs", type=int, default=0, help="override params.yaml")
    args = ap.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        allp = yaml.safe_load(f)
    cfg = allp["train"]
    epochs = args.epochs or int(cfg["epochs"])

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    with open(os.path.join(args.data_dir, "stats.json"), "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean, std = stats["rating_mean"], stats["rating_std"]

    train = load_split(os.path.join(args.data_dir, "train.npz"), device, args.max_samples)
    val = load_split(os.path.join(args.data_dir, "val.npz"), device, args.max_samples)
    test = load_split(os.path.join(args.data_dir, "test.npz"), device, args.max_samples)
    log.info("Samples: train %d | val %d | test %d",
             len(train["rating"]), len(val["rating"]), len(test["rating"]))

    model = build_model(cfg).to(device)
    exclude = set(cfg.get("exclude_features") or [])
    unknown = exclude - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown names in exclude_features: {sorted(unknown)}")
    model.set_feature_stats(
        stats["feat_mean"], stats["feat_std"],
        [0.0 if name in exclude else 1.0 for name in FEATURE_NAMES])
    if exclude:
        log.info("Features switched off: %s", sorted(exclude))
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.HuberLoss()

    y_train_z = (train["rating"] - mean) / std
    y_val_z = (val["rating"] - mean) / std
    baseline_val = score(torch.full_like(val["rating"], mean), val["rating"])
    baseline_test = score(torch.full_like(test["rating"], mean), test["rating"])
    log.info("Baseline (predict train mean) val MAE: %.1f", baseline_val["mae"])

    if "MLFLOW_RUN_ID" not in os.environ:  # under `mlflow run` the experiment is preset
        mlflow.set_experiment(args.experiment)
    with mlflow.start_run(log_system_metrics=True) as run:  # CPU/GPU/memory over time
        mlflow.set_tags({
            "git_commit": sh(["git", "rev-parse", "HEAD"]),
            "git_dirty": str(bool(sh(["git", "status", "--porcelain"]))),
            "git_tag": sh(["git", "describe", "--tags", "--abbrev=0"]),
            "data_dvc_md5": dvc_data_hash(),
            "device": str(device),
        })
        try:  # record how the evaluation data was produced
            with open("metrics/eval_train.json", "r", encoding="utf-8") as f:
                ev = json.load(f)
            mlflow.set_tags({"stockfish": ev["stockfish"], "eval_depth": str(ev["depth"])})
        except Exception:
            pass
        mlflow.log_params({f"train.{k}": v for k, v in cfg.items()})
        mlflow.log_params({f"data.{k}": v for k, v in allp.get("data", {}).items()})
        mlflow.log_params({"model_parameters": n_params, "train_samples": len(train["rating"])})
        mlflow.log_metrics({"baseline_val_mae": baseline_val["mae"],
                            "baseline_test_mae": baseline_test["mae"]})

        best_mae, best_epoch, best_state, bad = float("inf"), 0, None, 0
        bs = int(cfg["batch_size"])
        n = len(train["rating"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            perm = torch.randperm(n, device=device)
            total = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                pred = model(train["wp"][idx], train["side"][idx], train["length"][idx],
                             train["feat"][idx])
                loss = loss_fn(pred, y_train_z[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)

            z_val = predict_z(model, val)
            val_loss = loss_fn(z_val, y_val_z).item()
            val_scores = score(z_val * std + mean, val["rating"])
            sched.step(val_scores["mae"])

            mlflow.log_metrics({
                "train_loss": total / n,
                "val_loss": val_loss,
                "val_mae": val_scores["mae"],
                "val_rmse": val_scores["rmse"],
                "val_within_100": val_scores["within_100"],
                "val_within_200": val_scores["within_200"],
                "lr": opt.param_groups[0]["lr"],
            }, step=epoch)
            log.info("epoch %2d | train loss %.4f | val MAE %.1f (baseline %.1f) | %.1fs",
                     epoch, total / n, val_scores["mae"], baseline_val["mae"],
                     time.time() - t0)

            if val_scores["mae"] < best_mae:
                best_mae, best_epoch, bad = val_scores["mae"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= int(cfg["patience"]):
                    log.info("Early stopping at epoch %d (best %d)", epoch, best_epoch)
                    break

        # ---- final evaluation: best weights, test set used exactly once
        model.load_state_dict(best_state)
        z_test = predict_z(model, test)
        test_scores = score(z_test * std + mean, test["rating"])
        log.info("TEST MAE %.1f (baseline %.1f) | within 100: %.1f%% | within 200: %.1f%%",
                 test_scores["mae"], baseline_test["mae"],
                 100 * test_scores["within_100"], 100 * test_scores["within_200"])
        mlflow.log_metrics({
            "best_val_mae": best_mae,
            "best_epoch": best_epoch,
            "test_mae": test_scores["mae"],
            "test_rmse": test_scores["rmse"],
            "test_within_100": test_scores["within_100"],
            "test_within_200": test_scores["within_200"],
        })

        # ---- save model + everything the API will need to use it
        os.makedirs(args.model_dir, exist_ok=True)
        weights = os.path.join(args.model_dir, "best.pt")
        config = os.path.join(args.model_dir, "config.json")
        torch.save(best_state, weights)
        with open(config, "w", encoding="utf-8") as f:
            json.dump({"model": cfg, "rating_mean": mean, "rating_std": std,
                       "max_plies": stats["max_plies"],
                       "feature_names": FEATURE_NAMES,
                       "data": allp.get("data", {})}, f, indent=2)
        mlflow.log_artifact(weights, "model")
        mlflow.log_artifact(config, "model")
        mlflow.log_artifact(args.params, "config")
        mlflow.log_artifact("model.py", "code")
        try:  # servable with: mlflow models serve -m runs:/<run_id>/pyfunc_model
            mlflow.pyfunc.log_model(
                artifact_path="pyfunc_model",
                python_model=RatingPyfunc(),
                artifacts={"weights": weights, "config": config},
                code_paths=["model.py", "features.py", "pyfunc_model.py"],
                pip_requirements=["torch>=2.1", "numpy", "pandas", "mlflow>=2.12"],
            )
        except Exception:  # keep the trained model even if packaging fails
            log.exception("Could not log the pyfunc model")

        os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
        with open(args.metrics, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run.info.run_id,
                "best_epoch": best_epoch,
                "val_mae": best_mae,
                "test_mae": test_scores["mae"],
                "test_rmse": test_scores["rmse"],
                "test_within_100": test_scores["within_100"],
                "test_within_200": test_scores["within_200"],
                "baseline_test_mae": baseline_test["mae"],
                "model_parameters": n_params,
            }, f, indent=2)
        log.info("MLflow run ID: %s", run.info.run_id)


if __name__ == "__main__":
    main()
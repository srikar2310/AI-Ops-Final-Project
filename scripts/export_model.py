"""Export the registered champion model to a self-contained folder for Docker.

    python scripts/export_model.py              # -> serving/model
    dvc add serving/model                       # version it with DVC

Why: the MLflow registry stores absolute host paths, which do not exist inside a
container. The exported folder (MLmodel, weights, code/) is portable.
Uses MLFLOW_TRACKING_URI if set, else sqlite:///mlflow.db.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="chess-rating")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--out", default="serving/model")
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mv = MlflowClient().get_model_version_by_alias(args.name, args.alias)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{args.name}@{args.alias}",
                                        dst_path=str(out))
    if not (out / "MLmodel").exists():          # some MLflow versions nest the model one level down
        nested = [p.parent for p in out.rglob("MLmodel")]
        if not nested:
            raise SystemExit(f"No MLmodel file found under {out}")
        for item in nested[0].iterdir():
            shutil.move(str(item), out / item.name)

    info = {"name": args.name, "alias": args.alias, "version": mv.version,
            "source_run_id": mv.tags.get("source_run_id"), "git_commit": mv.tags.get("git_commit"),
            "val_mae": mv.tags.get("val_mae"), "serving_test_mae": mv.tags.get("serving_test_mae")}
    (out.parent / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"Exported {args.name} v{mv.version} -> {out}")

    env = out / "python_env.yaml"
    if env.exists():
        py = yaml.safe_load(env.read_text()).get("python")
        print(f"\nThis model was logged with Python {py}.")
        print(f"Build with:  PYTHON_VERSION={'.'.join(str(py).split('.')[:2])} docker compose up --build")


if __name__ == "__main__":
    main()
"""Read-only endpoints that feed the pipeline console.

The browser can't read dvc.yaml, dvc.lock, metrics/*.json or the MLflow
tracker itself, so the API does it and returns plain JSON.

    GET /ops/pipeline   DVC stages, their dependency edges, last-run info and metrics
    GET /ops/runs       recent MLflow runs (status + key metrics) and status counts
    GET /ops/service    live request / error / Stockfish counters of THIS process
    GET /ops/links      addresses of the Ray, MLflow, Grafana and Prometheus UIs

Nothing here changes state. There is no authentication, so do not expose these
routes outside your own network.
"""
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import yaml
from fastapi import APIRouter, Query, Request
from prometheus_client import REGISTRY

from .errors import OpsUnavailable

log = logging.getLogger("api.ops")
router = APIRouter(prefix="/ops", tags=["ops"])

_STARTED = time.time()
_SUMMARY_KEYS = ("seconds", "games", "positions", "positions_per_sec",
                 "games_per_sec", "val_mae", "test_mae", "failed_games_last_run")
_RUN_METRICS = ("best_val_mae", "test_mae", "serving_test_mae",
                "baseline_test_mae", "test_within_100", "best_epoch")
_RUN_PARAMS = ("train.channels", "train.use_features", "train.lr", "train_samples")


# ------------------------------------------------------------------ DVC files
def _read_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    except (OSError, yaml.YAMLError) as exc:
        raise OpsUnavailable(f"Could not read {path.name}: {exc}") from exc


def _subst(obj, item):
    """Replace ${item} in every string of a (nested) stage definition."""
    if isinstance(obj, str):
        return obj.replace("${item}", str(item))
    if isinstance(obj, list):
        return [_subst(o, item) for o in obj]
    if isinstance(obj, dict):
        return {_subst(k, item): _subst(v, item) for k, v in obj.items()}
    return obj


def _expand(doc):
    """dvc.yaml stages with `foreach` expanded the way DVC names them (evaluate@train)."""
    stages = {}
    for name, st in (doc.get("stages") or {}).items():
        if isinstance(st, dict) and "foreach" in st and isinstance(st["foreach"], list):
            for item in st["foreach"]:
                stages[f"{name}@{item}"] = _subst(st.get("do", {}), item)
        else:
            stages[name] = st or {}
    return stages


def _paths(entries):
    """DVC lists entries as 'path' or {'path': {options}}."""
    out = []
    for e in entries or []:
        if isinstance(e, str):
            out.append(e.rstrip("/"))
        elif isinstance(e, dict):
            out.extend(k.rstrip("/") for k in e)
    return out


def _inside(root, rel):
    """Resolve `rel` under `root`; None if it would escape the project folder."""
    p = (root / rel).resolve()
    return p if root == p or root in p.parents else None


def _summarise(metrics):
    merged = {}
    for data in metrics.values():
        if not isinstance(data, dict):
            continue
        splits = [v for v in data.values() if isinstance(v, dict) and "seconds" in v]
        if splits:  # prepare.json holds one block per split
            secs = sum(v.get("seconds", 0) for v in splits)
            games = sum(v.get("games_seen", 0) for v in splits)
            merged["seconds"] = round(secs, 1)
            merged["games"] = games
            merged["games_per_sec"] = round(games / secs, 2) if secs else 0
        for key in _SUMMARY_KEYS:
            if key in data and key not in merged:
                merged[key] = data[key]
    return merged


def build_pipeline(root):
    root = Path(root).resolve()
    doc = _read_yaml(root / "dvc.yaml")
    if doc is None:
        raise OpsUnavailable("dvc.yaml was not found in the project folder.")
    lock = _read_yaml(root / "dvc.lock") or {}
    locked = set((lock.get("stages") or {}).keys())

    stages, producers = [], {}
    for name, st in _expand(doc).items():
        outs = _paths(st.get("outs")) + _paths(st.get("metrics"))
        info = {"name": name, "cmd": " ".join(str(st.get("cmd", "")).split()),
                "deps": _paths(st.get("deps")), "outs": outs,
                "status": "has_run" if name in locked else "never_run",
                "last_run": None, "summary": {}, "metrics": {}}
        newest = 0.0
        for rel in _paths(st.get("metrics")):
            path = _inside(root, rel)
            if path is None or not path.is_file():
                continue
            try:
                info["metrics"][rel] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            newest = max(newest, path.stat().st_mtime)
        if newest:
            info["last_run"] = datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()
        info["summary"] = _summarise(info["metrics"])
        for out in outs:
            producers[out] = name
        stages.append(info)

    edges = []
    for st in stages:
        for dep in st["deps"]:
            for out, producer in producers.items():
                if producer != st["name"] and (dep == out or dep.startswith(out + "/")):
                    edge = [producer, st["name"]]
                    if edge not in edges:
                        edges.append(edge)

    def node(name):
        return re.sub(r"\W", "_", name)

    lines = ["graph LR"] + [f'  {node(s["name"])}["{s["name"]}"]' for s in stages]
    lines += [f"  {node(a)} --> {node(b)}" for a, b in edges]
    return {"stages": stages, "edges": edges, "mermaid": "\n".join(lines),
            "lock_found": bool(lock)}


@router.get("/pipeline")
def pipeline(request: Request):
    return build_pipeline(request.app.state.settings.project_root)


# --------------------------------------------------------------------- MLflow
def _mlflow(settings, path, body):
    url = settings.mlflow_tracking_url.rstrip("/") + path
    try:
        resp = requests.post(url, json=body, timeout=settings.ops_timeout_s)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.error("MLflow tracker request failed (%s): %r", url, exc)
        raise OpsUnavailable(
            "The experiment tracker (MLflow) is not reachable. Start it with "
            "'mlflow server' or 'mlflow ui' and check MLFLOW_TRACKING_URL.") from exc


def _kv(items, wanted):
    d = {x["key"]: x["value"] for x in items or []}
    return {k: d[k] for k in wanted if k in d}


@router.get("/runs")
def runs(request: Request, limit: int = Query(20, ge=1, le=100),
         experiment: Optional[str] = None):
    s = request.app.state.settings
    exps = _mlflow(s, "/api/2.0/mlflow/experiments/search", {"max_results": 100})
    names = {e["experiment_id"]: e["name"] for e in exps.get("experiments", [])
             if experiment is None or e["name"] == experiment}
    if not names:
        return {"runs": [], "counts": {}}
    found = _mlflow(s, "/api/2.0/mlflow/runs/search",
                    {"experiment_ids": list(names), "max_results": limit,
                     "order_by": ["attributes.start_time DESC"]})
    out, counts = [], {}
    for r in found.get("runs", []):
        info, data = r["info"], r.get("data", {})
        counts[info["status"]] = counts.get(info["status"], 0) + 1
        tags = {t["key"]: t["value"] for t in data.get("tags", [])}
        out.append({
            "run_id": info["run_id"], "experiment": names.get(info["experiment_id"]),
            "status": info["status"], "start_time": info.get("start_time"),
            "end_time": info.get("end_time"), "git_commit": tags.get("git_commit"),
            "metrics": _kv(data.get("metrics"), _RUN_METRICS),
            "params": _kv(data.get("params"), _RUN_PARAMS)})
    return {"runs": out, "counts": counts}


# ----------------------------------------------------------- live service stats
def _samples(name):
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == name:
                yield sample


def _total(name, **labels):
    return sum(s.value for s in _samples(name)
               if all(s.labels.get(k) == v for k, v in labels.items()))


@router.get("/service")
def service(request: Request):
    """Counters of this API process (they restart from 0 when it restarts)."""
    by_class, errors = {}, {}
    for s in _samples("api_requests_total"):
        cls = str(s.labels.get("status", "?"))[:1] + "xx"
        by_class[cls] = by_class.get(cls, 0) + int(s.value)
    for s in _samples("api_errors_total"):
        t = s.labels.get("error_type", "?")
        errors[t] = errors.get(t, 0) + int(s.value)
    n = _total("api_request_latency_seconds_count", endpoint="/predict")
    secs = _total("api_request_latency_seconds_sum", endpoint="/predict")
    pool, model = request.app.state.pool, request.app.state.model
    return {
        "uptime_seconds": round(time.time() - _STARTED),
        "engine_ready": bool(pool and pool.ready()),
        "model_ready": bool(model and model.ready()),
        "requests_total": sum(by_class.values()),
        "requests_by_status_class": by_class,
        "errors_by_type": errors,
        "requests_in_progress": int(_total("api_requests_in_progress")),
        "predictions": int(n),
        "predict_avg_seconds": round(secs / n, 2) if n else None,
        "stockfish": {"positions": int(_total("stockfish_positions_total")),
                      "failures": int(_total("stockfish_engine_failures_total")),
                      "restarts": int(_total("stockfish_engine_restarts_total")),
                      "engines_available": int(_total("stockfish_engines_available"))},
    }


@router.get("/links")
def links(request: Request):
    s = request.app.state.settings
    return {"ray": s.ray_url, "mlflow": s.mlflow_ui_url or s.mlflow_tracking_url,
            "grafana": s.grafana_url, "prometheus": s.prometheus_url}

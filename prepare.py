"""Prepare stage (Ray): evaluated PGN shards -> compact NumPy arrays.

Reads dataset_clean/{split}_evaluated/*.pgn (written by Stockfish_train_eval.py)
and writes data/processed/{split}.npz plus data/processed/stats.json.

Each PGN shard is processed by one Ray task, so shards are converted in
parallel. Results are collected in shard order, so the output is deterministic.
Watch the pipeline live in the Ray dashboard (http://127.0.0.1:8265 by default).

Every game becomes TWO samples (one per player). For each sample we store the
win probability after every ply from THAT player's point of view. The target
is that player's rating.

Arrays in each .npz (N samples, L = max_plies):
    wp      float32 [N, L]  win probability for the player (0..1), padded with 0.5
    length  int16   [N]     number of real plies (<= L)
    side    int8    [N]     0 = player is White, 1 = player is Black
    rating  float32 [N]     the player's Elo (the regression target)
    feat    float32 [N, F]  game-level features (see features.FEATURE_NAMES), incl. the result

To keep the dashboard open after the job, start a persistent cluster first:
    ray start --head
    python prepare.py --address auto
"""
import argparse
import json
import logging
import os
import re
import time

import numpy as np
import yaml

from features import FEATURE_NAMES, N_FEATURES, build_sample, game_features

log = logging.getLogger("prepare")

BLOCK_END = "\n\n\n"  # game terminator written by Stockfish_train_eval.py
TAG_RE = re.compile(r'\[(\w+) "([^"]*)"\]')
COUNT_KEYS = ("games_seen", "skipped_bad", "skipped_short", "truncated")


def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["data"]


def parse_games(path):
    """Yield (white_elo, black_elo, evals, result) for every complete game in a shard."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    parts = data.split(BLOCK_END)
    parts.pop()  # trailing piece is empty (or a partial game); ignore it
    for block in parts:
        tags = dict(TAG_RE.findall(block))
        evals = tags.get("Evals")
        if not evals:
            yield None, None, None, None
            continue
        try:
            w, b = int(tags["WhiteElo"]), int(tags["BlackElo"])
            cp = np.array(evals.split(","), dtype=np.int32)
        except (KeyError, ValueError):
            yield None, None, None, None
            continue
        yield w, b, cp, tags.get("Result", "*")


def process_shard(path, max_plies, min_plies, clip):
    """Convert ONE shard. Runs as a Ray task (and is a plain function, so it
    can also be unit-tested without Ray). Returns (arrays, counts)."""
    rows, lengths, sides, ratings, feats = [], [], [], [], []
    counts = dict.fromkeys(COUNT_KEYS, 0)

    for w_elo, b_elo, cp, result in parse_games(path):
        counts["games_seen"] += 1
        if cp is None:
            counts["skipped_bad"] += 1  # missing/invalid rating or evals
            continue
        if len(cp) < min_plies:
            counts["skipped_short"] += 1
            continue
        if len(cp) > max_plies:
            counts["truncated"] += 1

        for side, elo in ((0, w_elo), (1, b_elo)):
            row, n = build_sample(cp, side, max_plies, clip)
            rows.append(row)
            lengths.append(n)
            sides.append(side)
            ratings.append(elo)
            feats.append(game_features(cp, side, result, clip))

    arrays = {
        "wp": np.stack(rows) if rows else np.zeros((0, max_plies), dtype=np.float32),
        "length": np.array(lengths, dtype=np.int16),
        "side": np.array(sides, dtype=np.int8),
        "rating": np.array(ratings, dtype=np.float32),
        "feat": (np.stack(feats) if feats
                 else np.zeros((0, N_FEATURES), dtype=np.float32)),
    }
    return arrays, counts


def prepare_split(ray, in_dir, max_plies, min_plies, clip):
    shards = sorted(f for f in os.listdir(in_dir) if f.endswith(".pgn"))
    if not shards:
        raise FileNotFoundError(f"No .pgn shards found in {in_dir}")

    task = ray.remote(process_shard)
    futures = [task.remote(os.path.join(in_dir, s), max_plies, min_plies, clip)
               for s in shards]
    results = ray.get(futures)  # same order as `shards`

    arrays = {k: np.concatenate([r[0][k] for r in results])
              for k in ("wp", "length", "side", "rating", "feat")}
    counts = {k: sum(r[1][k] for r in results) for k in COUNT_KEYS}
    counts["shards"] = len(shards)
    counts["games_used"] = (counts["games_seen"] - counts["skipped_bad"]
                            - counts["skipped_short"])
    counts["samples"] = int(len(arrays["rating"]))
    return arrays, counts


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset_clean")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--metrics", default="metrics/prepare.json")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4),
                    help="Ray CPUs to use (ignored with --address)")
    ap.add_argument("--address", default=None,
                    help="connect to a running Ray cluster, e.g. 'auto'")
    args = ap.parse_args()

    import ray  # imported here so process_shard stays testable without Ray

    p = load_params(args.params)
    max_plies, min_plies, clip = int(p["max_plies"]), int(p["min_plies"]), float(p["cp_clip"])
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)

    if args.address:
        ctx = ray.init(address=args.address, logging_level=logging.WARNING)
    else:
        ctx = ray.init(num_cpus=args.workers, include_dashboard=True,
                       logging_level=logging.WARNING)
    log.info("Ray started | CPUs: %s | dashboard: %s",
             ray.cluster_resources().get("CPU"),
             getattr(ctx, "dashboard_url", None) or "n/a")

    metrics = {"max_plies": max_plies, "min_plies": min_plies, "cp_clip": clip,
               "ray_cpus": ray.cluster_resources().get("CPU")}
    stats = {"max_plies": max_plies}

    try:
        for split in args.splits:
            t0 = time.time()
            in_dir = os.path.join(args.root, f"{split}_evaluated")
            arrays, counts = prepare_split(ray, in_dir, max_plies, min_plies, clip)
            np.savez(os.path.join(args.out, f"{split}.npz"), **arrays)

            seconds = time.time() - t0
            counts["seconds"] = round(seconds, 1)
            counts["games_per_sec"] = round(counts["games_seen"] / seconds, 1) if seconds else 0
            counts["truncated_pct"] = round(
                100 * counts["truncated"] / max(1, counts["games_used"]), 2)
            counts["rating_min"] = float(arrays["rating"].min())
            counts["rating_max"] = float(arrays["rating"].max())
            counts["rating_mean"] = round(float(arrays["rating"].mean()), 2)
            counts["rating_std"] = round(float(arrays["rating"].std()), 2)
            metrics[split] = counts
            log.info("[%s] %s", split, counts)

            if split == "train":
                # Normalisation stats come from TRAIN ONLY (no leakage into val/test)
                stats["rating_mean"] = float(arrays["rating"].mean())
                stats["rating_std"] = float(arrays["rating"].std())
                stats["feature_names"] = FEATURE_NAMES
                stats["feat_mean"] = arrays["feat"].mean(axis=0).tolist()
                stats["feat_std"] = arrays["feat"].std(axis=0).tolist()
    finally:
        ray.shutdown()

    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("Wrote arrays to %s and metrics to %s", args.out, args.metrics)


if __name__ == "__main__":
    main()
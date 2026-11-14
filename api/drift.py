"""Input-drift monitor: are the games users submit like the games we trained on?

For every prediction the API computes the same game-level features the model
uses (features.game_features) and keeps the last WINDOW of them. Each feature's
shift is measured in TRAINING standard deviations:

    shift = (mean of the window - training mean) / training std

Published to Prometheus (see monitoring/):
    feature_drift_shift{feature}   signed shift per feature
    feature_drift_max              largest absolute shift
    predicted_rating_shift         same idea for the predicted rating
    drift_window_size              how many games the window holds

Nothing is reported until MIN_SAMPLES games are in the window. Monitoring must
never break a prediction, so observe() swallows and logs its own errors.
Training statistics come from data/processed/stats.json (written by prepare.py,
train split only).
"""
import json
import logging
import os
import threading
from collections import deque

import numpy as np
from prometheus_client import Gauge

from features import FEATURE_NAMES, game_features

log = logging.getLogger("api.drift")

FEATURE_SHIFT = Gauge("feature_drift_shift",
                      "Window mean minus training mean, in training std", ["feature"])
FEATURE_DRIFT_MAX = Gauge("feature_drift_max", "Largest absolute feature shift")
RATING_SHIFT = Gauge("predicted_rating_shift",
                     "Mean predicted rating minus training mean rating, in training std")
WINDOW_SIZE = Gauge("drift_window_size", "Games currently in the drift window")


class DriftMonitor:
    def __init__(self, stats, clip, window=200, min_samples=30):
        if list(stats["feature_names"]) != list(FEATURE_NAMES):
            raise ValueError("stats.json was made with different features than features.py; "
                             "re-run the prepare stage")
        self.mean = np.asarray(stats["feat_mean"], dtype=float)
        std = np.asarray(stats["feat_std"], dtype=float)
        # (near-)constant column, e.g. result_known: float32 rounding leaves a std of ~1e-5,
        # which would turn tiny differences into huge shifts. Treat it as constant.
        self.std = np.where(std < 1e-4, 1.0, std)
        self.rating_mean = float(stats["rating_mean"])
        self.rating_std = float(stats["rating_std"]) or 1.0
        self.clip, self.min_samples = clip, min_samples
        self._feats = deque(maxlen=window)
        self._ratings = deque(maxlen=window)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, clip):
        """Build from DRIFT_STATS_PATH; None (monitoring off) if the file is unusable."""
        path = os.environ.get("DRIFT_STATS_PATH", "data/processed/stats.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                monitor = cls(json.load(f), clip)
        except (OSError, ValueError, KeyError) as exc:
            log.warning("Drift monitoring is OFF (%s: %s)", path, exc)
            return None
        log.info("Drift monitoring on, reference statistics from %s", path)
        return monitor

    def observe(self, evals, side, result, predicted_rating):
        """Record one predicted player. side is 'white' or 'black'."""
        try:
            vec = game_features(np.asarray(evals), 0 if side == "white" else 1, result, self.clip)
            with self._lock:
                self._feats.append(vec)
                self._ratings.append(float(predicted_rating))
                self._publish()
        except Exception:  # noqa: BLE001 - never break a prediction
            log.exception("Drift update failed")

    def _publish(self):
        n = len(self._feats)
        WINDOW_SIZE.set(n)
        if n < self.min_samples:
            return
        shift = (np.mean(self._feats, axis=0) - self.mean) / self.std
        for name, value in zip(FEATURE_NAMES, shift):
            FEATURE_SHIFT.labels(name).set(float(value))
        FEATURE_DRIFT_MAX.set(float(np.abs(shift).max()))
        RATING_SHIFT.set((np.mean(self._ratings) - self.rating_mean) / self.rating_std)
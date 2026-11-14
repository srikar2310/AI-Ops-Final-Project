"""MLflow pyfunc wrapper so the trained CNN can be served with `mlflow models serve`.

Input  (DataFrame / JSON records), one row per player to rate:
    evals  : list of White-perspective centipawn evals, one per ply
    side   : "white" or "black"  (which player to rate)
    result : optional PGN result, "1-0", "0-1", "1/2-1/2" (default "*" = unknown)
Output: DataFrame with one column, `predicted_rating`.

It reuses features.build_sample and model.make_inputs (through the model), so
serving sees exactly the same preprocessing as training.
"""
import json

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from features import build_sample, game_features
from model import build_model


def _side_to_int(side):
    s = str(side).strip().lower()
    if s in ("white", "w", "0"):
        return 0
    if s in ("black", "b", "1"):
        return 1
    raise ValueError(f"side must be 'white' or 'black', got {side!r}")


class RatingPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["config"], "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.model = build_model(self.cfg["model"])
        state = torch.load(context.artifacts["weights"], map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, context, model_input, params=None):
        df = pd.DataFrame(model_input)
        max_plies = int(self.cfg["max_plies"])
        clip = float(self.cfg["data"]["cp_clip"])

        results = df["result"] if "result" in df.columns else ["*"] * len(df)

        rows, lengths, sides, feats = [], [], [], []
        for evals, side, result in zip(df["evals"], df["side"], results):
            if evals is None or len(evals) == 0:
                raise ValueError("evals must contain at least one ply")
            s = _side_to_int(side)
            row, n = build_sample(evals, s, max_plies, clip)
            rows.append(row)
            lengths.append(n)
            sides.append(s)
            feats.append(game_features(evals, s, result, clip))

        with torch.no_grad():
            z = self.model(
                torch.from_numpy(np.stack(rows)).float(),
                torch.tensor(sides, dtype=torch.long),
                torch.tensor(lengths, dtype=torch.long),
                torch.from_numpy(np.stack(feats)).float(),
            )
        rating = z.numpy() * self.cfg["rating_std"] + self.cfg["rating_mean"]
        return pd.DataFrame({"predicted_rating": np.round(rating, 1)})
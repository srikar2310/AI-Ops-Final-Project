"""Talks to the trained model.

Two interchangeable ways, chosen with MODEL_MODE:
  rest  (default) call the MLflow model server over HTTP. The API and the model
        stay independent software blocks connected only by a configurable URL:
            mlflow models serve -m models:/chess-rating@champion -p 5001
  local load the registered model inside the API process with
        mlflow.pyfunc.load_model(MODEL_URI). Useful for development or if the
        MLflow server cannot be run.

Both take the same records: {"evals": [...], "side": "white", "result": "1-0"}
and return one predicted rating per record. The feature engineering happens
inside the model (shared features.py), so there is a single source of truth.
"""
import logging
from urllib.parse import urlsplit, urlunsplit

import requests

from .errors import ModelServiceError

log = logging.getLogger("api.model")


def _extract(predictions, expected):
    """MLflow returns either [{"predicted_rating": x}, ...] or [x, ...]."""
    try:
        values = [float(p["predicted_rating"] if isinstance(p, dict) else p)
                  for p in predictions]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelServiceError("The model service returned an unexpected response.") from exc
    if len(values) != expected:
        raise ModelServiceError(
            f"The model service returned {len(values)} predictions for {expected} inputs.")
    return values


class RestModelClient:
    def __init__(self, url, timeout):
        self.url, self.timeout = url, timeout
        self.session = requests.Session()
        parts = urlsplit(url)
        self._ping = urlunsplit((parts.scheme, parts.netloc, "/ping", "", ""))

    def predict(self, records):
        try:
            resp = self.session.post(
                self.url, json={"dataframe_records": records}, timeout=self.timeout)
        except requests.RequestException as exc:
            log.error("Model service unreachable at %s: %r", self.url, exc)
            raise ModelServiceError(
                "The prediction model is not reachable right now. Please try again later."
            ) from exc
        if resp.status_code != 200:
            log.error("Model service HTTP %s: %s", resp.status_code, resp.text[:300])
            raise ModelServiceError(
                f"The prediction model returned an error (HTTP {resp.status_code}).")
        try:
            body = resp.json()
            predictions = body["predictions"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelServiceError("The model service returned an unexpected response.") from exc
        return _extract(predictions, len(records))

    def ready(self):
        try:
            return self.session.get(self._ping, timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def describe(self):
        return {"mode": "rest", "target": self.url}


class LocalModelClient:
    def __init__(self, model_uri):
        import mlflow.pyfunc  # imported here so REST mode doesn't need to load mlflow

        self.uri = model_uri
        log.info("Loading model %s", model_uri)
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, records):
        import pandas as pd

        try:
            out = self.model.predict(pd.DataFrame(records))
            values = out["predicted_rating"].tolist() if hasattr(out, "columns") else list(out)
        except Exception as exc:  # noqa: BLE001
            log.exception("Local model failed")
            raise ModelServiceError("The prediction model failed to run.") from exc
        return _extract(values, len(records))

    def ready(self):
        return self.model is not None

    def describe(self):
        return {"mode": "local", "target": self.uri}


def build_model_client(settings):
    if settings.model_mode == "local":
        return LocalModelClient(settings.model_uri)
    return RestModelClient(settings.mlflow_url, settings.model_timeout_s)

"""The ONLY place the frontend talks to the backend: plain REST calls.

The API address comes from the API_URL environment variable (default
http://localhost:8000), so the frontend and backend stay independent blocks.
"""
import os

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
PREDICT_TIMEOUT_S = 120

MESSAGES = {
    "network": "We can't reach the analysis server. Check that it is running, then try again.",
    "engine_busy": "The analysis service is busy right now. Please try again in a few seconds.",
    "engine_failure": "The chess engine had a problem. Please try again.",
    "analysis_timeout": "This game took too long to analyse. Try a shorter game.",
    "model_unavailable": "The rating model is not available right now. Please try again later.",
    "ops_unavailable": "That information source is not available right now.",
    "internal_error": "Something went wrong on our side.",
}


class ApiError(Exception):
    def __init__(self, code, message=None, request_id=None):
        super().__init__(message or code)
        self.code, self.message, self.request_id = code, message, request_id


def friendly(err):
    """Message for a non-technical user. bad_pgn text is already written for them."""
    if err.code in ("bad_pgn", "invalid_request") and err.message:
        return err.message
    text = MESSAGES.get(err.code, "Something unexpected happened. Please try again.")
    if err.request_id and err.code != "network":
        text += f" (Reference: {err.request_id})"
    return text


def _request(method, path, timeout=15, **kwargs):
    try:
        resp = requests.request(method, API_URL + path, timeout=timeout, **kwargs)
    except requests.RequestException:
        raise ApiError("network") from None
    try:
        data = resp.json()
    except ValueError:
        data = None
    if not resp.ok:
        err = (data or {}).get("error", {})
        raise ApiError(err.get("code", f"http_{resp.status_code}"), err.get("message"),
                       (data or {}).get("request_id"))
    return data


def get_config():
    return _request("GET", "/config")


def validate(pgn):
    return _request("POST", "/validate", json={"pgn": pgn})


def split_games(text):
    return _request("POST", "/games/split", json={"text": text}, timeout=30)


def predict(pgn, player="both"):
    return _request("POST", "/predict", timeout=PREDICT_TIMEOUT_S,
                    json={"pgn": pgn, "player": player})


def ops(path):
    return _request("GET", path)


def ready():
    """/ready answers 503 with a body when degraded; treat that as data, not an error."""
    try:
        return requests.get(API_URL + "/ready", timeout=3).json()
    except (requests.RequestException, ValueError):
        return {"status": "unreachable"}
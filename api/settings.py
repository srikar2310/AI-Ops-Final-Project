"""Runtime settings for the API.

Engine and analysis settings come from params.yaml (section `eval`/`data`), so
the API analyses a game exactly the way the training data was produced.
Deployment settings (ports, pool size, model location, ...) come from
environment variables.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml


def _env(name, default, cast=str):
    value = os.environ.get(name)
    return default if value in (None, "") else cast(value)


@dataclass(frozen=True)
class Settings:
    # --- must match the training data (from params.yaml) ---
    engine_path: str
    depth: int
    hash_mb: int
    mate_score: int
    min_plies: int
    # --- limits ---
    max_plies: int = 400
    max_pgn_bytes: int = 200_000
    pool_size: int = 2
    acquire_timeout_s: float = 5.0      # wait for a free engine before answering 503
    position_timeout_s: float = 30.0    # one Stockfish search; longer = engine is hung
    analysis_timeout_s: float = 90.0    # whole game
    # --- model service ---
    model_mode: str = "rest"            # "rest" = MLflow model server, "local" = load in-process
    mlflow_url: str = "http://127.0.0.1:5001/invocations"
    model_uri: str = "models:/chess-rating@champion"
    model_timeout_s: float = 30.0
    # --- misc ---
    log_level: str = "INFO"
    log_file: Optional[str] = None
    cors_origins: Tuple[str, ...] = ("*",)

    def __post_init__(self):
        if self.model_mode not in ("rest", "local"):
            raise ValueError(f"MODEL_MODE must be 'rest' or 'local', got {self.model_mode!r}")
        if self.pool_size < 1:
            raise ValueError("ENGINE_POOL_SIZE must be at least 1")

    @classmethod
    def load(cls, params_path=None):
        path = params_path or os.environ.get("PARAMS_FILE", "params.yaml")
        with open(path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        ev, data = params["eval"], params["data"]
        origins = _env("CORS_ORIGINS", "*")
        return cls(
            engine_path=_env("STOCKFISH_PATH", ev["engine_path"]),
            depth=int(ev["depth"]),
            hash_mb=int(ev["hash_mb"]),
            mate_score=int(ev["mate_score"]),
            min_plies=int(data["min_plies"]),
            max_plies=_env("API_MAX_PLIES", 400, int),
            max_pgn_bytes=_env("API_MAX_PGN_BYTES", 200_000, int),
            pool_size=_env("ENGINE_POOL_SIZE", 2, int),
            acquire_timeout_s=_env("ENGINE_ACQUIRE_TIMEOUT_S", 5.0, float),
            position_timeout_s=_env("ENGINE_POSITION_TIMEOUT_S", 30.0, float),
            analysis_timeout_s=_env("ANALYSIS_TIMEOUT_S", 90.0, float),
            model_mode=_env("MODEL_MODE", "rest"),
            mlflow_url=_env("MLFLOW_URL", "http://127.0.0.1:5001/invocations"),
            model_uri=_env("MODEL_URI", "models:/chess-rating@champion"),
            model_timeout_s=_env("MODEL_TIMEOUT_S", 30.0, float),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            log_file=_env("LOG_FILE", None),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )

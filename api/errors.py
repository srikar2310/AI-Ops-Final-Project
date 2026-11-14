"""Exceptions the API raises on purpose, each mapped to one HTTP status."""


class APIError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message, headers=None):
        super().__init__(message)
        self.message = message
        self.headers = headers


class BadPGNError(APIError):
    """The uploaded text is not a usable game (syntax, illegal move, too short...)."""
    status_code = 422
    code = "bad_pgn"


class PoolBusyError(APIError):
    """All Stockfish engines are busy and none became free in time."""
    status_code = 503
    code = "engine_busy"

    def __init__(self, message):
        super().__init__(message, headers={"Retry-After": "5"})


class EngineFailure(APIError):
    """Stockfish crashed or hung and a retry on a fresh engine did not help."""
    status_code = 503
    code = "engine_failure"


class AnalysisTimeout(APIError):
    """The whole-game analysis exceeded its time budget."""
    status_code = 504
    code = "analysis_timeout"


class ModelServiceError(APIError):
    """The model service is unreachable or returned something unusable."""
    status_code = 502
    code = "model_unavailable"


class OpsUnavailable(APIError):
    """A source for the pipeline console (dvc files, MLflow tracker) can't be read."""
    status_code = 503
    code = "ops_unavailable"

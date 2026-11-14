"""Prometheus metrics. Scraped from GET /metrics.

Run uvicorn with ONE worker: the Stockfish pool and these in-process metrics
are per process (scale out with more containers, not more workers).
"""
from prometheus_client import Counter, Gauge, Histogram

# ---- HTTP layer
REQUESTS = Counter("api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
REQUEST_SECONDS = Histogram(
    "api_request_latency_seconds", "HTTP request latency", ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120))
IN_PROGRESS = Gauge("api_requests_in_progress", "Requests being handled right now")
ERRORS = Counter("api_errors_total", "Requests that ended in an error",
                 ["endpoint", "error_type"])

# ---- input
PGN_PLIES = Histogram("pgn_plies", "Plies (half-moves) per submitted game",
                      buckets=(10, 20, 40, 60, 80, 100, 150, 200, 300, 400))

# ---- predictions
PREDICTED_RATING = Histogram("predicted_rating", "Distribution of predicted ratings",
                             ["side"], buckets=tuple(range(400, 3201, 200)))
MODEL_SECONDS = Histogram(
    "model_request_seconds", "Time spent in the model service",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10))

# ---- Stockfish
STOCKFISH_ANALYSIS_SECONDS = Histogram(
    "stockfish_analysis_seconds", "Stockfish time to analyse one whole game",
    buckets=(0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60, 90))
STOCKFISH_POSITION_SECONDS = Histogram(
    "stockfish_position_seconds", "Stockfish time per position",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10))
STOCKFISH_POSITIONS = Counter("stockfish_positions_total", "Positions searched")
STOCKFISH_FAILURES = Counter("stockfish_engine_failures_total",
                             "Engine crashes or hangs during analysis")
STOCKFISH_RESTARTS = Counter("stockfish_engine_restarts_total",
                             "Engines restarted after a failure")
ENGINES_AVAILABLE = Gauge("stockfish_engines_available", "Idle engines in the pool")

"""FastAPI layer: PGN in -> Stockfish evals -> model -> predicted ratings.

Run (from the project folder, so params.yaml is found):
    uvicorn api.main:create_app --factory --port 8000

Endpoints
    POST /predict   analyse one game and estimate the player ratings
    POST /validate  check a PGN without running Stockfish (instant feedback)
    POST /games/split  list the games in an uploaded PGN file
    GET  /config    limits the UI needs (min/max length, file size, depth)
    GET  /ops/*     read-only data for the pipeline console (see ops.py)
    GET  /health    liveness: the process is up
    GET  /ready     readiness: Stockfish and the model service are usable
    GET  /metrics   Prometheus metrics
    GET  /docs      interactive API documentation
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import metrics, ops
from .engine import EnginePool
from .errors import APIError, BadPGNError
from .labels import annotate_moves, summarise
from .logging_setup import request_id_var, setup_logging
from .model_client import build_model_client
from .pgn_utils import parse_pgn, split_games
from .schemas import (AnalysisInfo, ConfigResponse, ErrorInfo, GameInfo,
                      GameListItem, ModelInfo, MoveInfo, PlayerPrediction,
                      PredictRequest, PredictResponse, SplitRequest,
                      SplitResponse, ValidateRequest, ValidateResponse)
from .settings import Settings

log = logging.getLogger("api")

QUIET_PATHS = {"/metrics", "/health", "/ready"}  # probes: don't flood the log
SHORT_GAME_PLIES = 20
API_VERSION = "1.1.0"


def _endpoint(request):
    """Route template ('/predict'), never the raw URL, to keep metric labels bounded."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def _error_body(code, message):
    return {"error": {"code": code, "message": message},
            "request_id": request_id_var.get()}


def _game_warnings(game):
    """Plain-language warnings shared by /validate and /predict."""
    warnings = []
    if game.result == "*":
        warnings.append("The game result is unknown, so the estimate is less reliable. "
                        "Add a [Result] tag if the game is finished.")
    elif game.result_inferred:
        warnings.append(f"The result ({game.result}) was inferred from the final position.")
    if game.plies < SHORT_GAME_PLIES:
        warnings.append("This is a very short game, so the estimate is less reliable.")
    return warnings


def create_app(settings=None, engine_pool=None, model_client=None):
    """Build the app. `engine_pool` / `model_client` can be injected (used by tests)."""
    settings = settings or Settings.load()
    setup_logging(settings.log_level, settings.log_file)

    @asynccontextmanager
    async def lifespan(app):
        owns_pool = app.state.pool is None
        if owns_pool:
            app.state.pool = EnginePool(settings)
            app.state.pool.start()          # fails fast if Stockfish can't start
        if app.state.model is None:
            app.state.model = build_model_client(settings)
        log.info("API ready | model: %s", app.state.model.describe())
        try:
            yield
        finally:
            if owns_pool:
                app.state.pool.close()
            log.info("API stopped")

    app = FastAPI(
        title="Chess rating estimator",
        description="Upload a PGN. Stockfish evaluates the game and a trained model "
                    "estimates each player's rating.",
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.pool = engine_pool
    app.state.model = model_client
    app.include_router(ops.router)

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        started = time.monotonic()
        metrics.IN_PROGRESS.inc()
        try:
            try:
                response = await call_next(request)
            except Exception:  # noqa: BLE001 - last line of defence: never leak a traceback
                log.exception("Unhandled error on %s %s", request.method, request.url.path)
                metrics.ERRORS.labels(_endpoint(request), "internal_error").inc()
                response = JSONResponse(
                    status_code=500,
                    content=_error_body("internal_error", "Something went wrong on our side."))
            elapsed = time.monotonic() - started
            endpoint = _endpoint(request)
            metrics.REQUESTS.labels(request.method, endpoint, str(response.status_code)).inc()
            metrics.REQUEST_SECONDS.labels(endpoint).observe(elapsed)
            response.headers["X-Request-ID"] = rid
            if request.url.path not in QUIET_PATHS:
                log.info("%s %s -> %s in %.2fs", request.method, request.url.path,
                         response.status_code, elapsed)
            return response
        finally:
            metrics.IN_PROGRESS.dec()
            request_id_var.reset(token)

    # CORS is added last so it is the outermost layer (error responses get headers too)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins),
        allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-ID"])

    # ------------------------------------------------------ error handlers
    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError):
        metrics.ERRORS.labels(_endpoint(request), exc.code).inc()
        level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
        log.log(level, "%s: %s", exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, headers=exc.headers,
                            content=_error_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}"
            for e in exc.errors())
        metrics.ERRORS.labels(_endpoint(request), "invalid_request").inc()
        log.warning("invalid_request: %s", detail)
        return JSONResponse(status_code=422, content=_error_body("invalid_request", detail))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        code = f"http_{exc.status_code}"
        metrics.ERRORS.labels(_endpoint(request), code).inc()
        return JSONResponse(status_code=exc.status_code,
                            content=_error_body(code, str(exc.detail)))

    # ------------------------------------------------------------- endpoints
    @app.post("/predict", response_model=PredictResponse, tags=["prediction"])
    def predict(req: PredictRequest):
        # A plain `def` endpoint runs in a worker thread, so the blocking Stockfish
        # call does not freeze the server.
        rid = request_id_var.get()
        game = parse_pgn(req.pgn, max_bytes=settings.max_pgn_bytes,
                         min_plies=settings.min_plies, max_plies=settings.max_plies)
        metrics.PGN_PLIES.observe(game.plies)
        log.info("Parsed game: %d plies, result %s, player=%s",
                 game.plies, game.result, req.player)

        analysis = app.state.pool.analyse_game(game.moves, token=rid)
        metrics.STOCKFISH_ANALYSIS_SECONDS.observe(analysis.seconds)

        sides = ["white", "black"] if req.player == "both" else [req.player]
        records = [{"evals": analysis.evals, "side": s, "result": game.result}
                   for s in sides]
        started = time.monotonic()
        ratings = app.state.model.predict(records)
        model_seconds = time.monotonic() - started
        metrics.MODEL_SECONDS.observe(model_seconds)

        players = []
        for side, rating in zip(sides, ratings):
            metrics.PREDICTED_RATING.labels(side).observe(rating)
            players.append(PlayerPrediction(
                side=side,
                name=game.white if side == "white" else game.black,
                predicted_rating=round(rating, 1),
                actual_rating=game.white_elo if side == "white" else game.black_elo))

        warnings = _game_warnings(game)
        moves = annotate_moves(analysis.evals, game.moves_san, settings.cp_clip)

        log.info("Predicted %s in %.2fs (Stockfish %.2fs, model %.2fs)",
                 {p.side: p.predicted_rating for p in players},
                 analysis.seconds + model_seconds, analysis.seconds, model_seconds)
        return PredictResponse(
            request_id=rid,
            players=players,
            game=GameInfo(plies=game.plies, result=game.result,
                          result_known=game.result != "*",
                          result_inferred=game.result_inferred),
            analysis=AnalysisInfo(
                engine=app.state.pool.engine_name, depth=settings.depth,
                positions_searched=analysis.positions,
                seconds=round(analysis.seconds, 2), evals_white_cp=analysis.evals),
            model=ModelInfo(seconds=round(model_seconds, 3), **app.state.model.describe()),
            moves=[MoveInfo(**mv) for mv in moves],
            move_summary=summarise(moves),
            warnings=warnings)

    @app.post("/validate", response_model=ValidateResponse, tags=["input"])
    def validate(req: ValidateRequest):
        """Check a PGN without running Stockfish. A bad game is a normal answer
        (valid=false + a message), not an HTTP error, so the UI can show it live."""
        try:
            game = parse_pgn(req.pgn, max_bytes=settings.max_pgn_bytes,
                             min_plies=settings.min_plies, max_plies=settings.max_plies)
        except BadPGNError as exc:
            return ValidateResponse(valid=False,
                                    error=ErrorInfo(code=exc.code, message=exc.message))
        return ValidateResponse(
            valid=True, plies=game.plies, result=game.result,
            result_known=game.result != "*", result_inferred=game.result_inferred,
            white=game.white, black=game.black,
            white_elo=game.white_elo, black_elo=game.black_elo,
            warnings=_game_warnings(game))

    @app.post("/games/split", response_model=SplitResponse, tags=["input"])
    def games_split(req: SplitRequest):
        """List the games in an uploaded file; each item carries its own PGN."""
        items, truncated = split_games(
            req.text, max_bytes=settings.max_upload_bytes,
            max_games=settings.max_games_per_file, game_max_bytes=settings.max_pgn_bytes,
            min_plies=settings.min_plies, max_plies=settings.max_plies)
        return SplitResponse(games=[GameListItem(**it) for it in items],
                             truncated=truncated, limit=settings.max_games_per_file)

    @app.get("/config", response_model=ConfigResponse, tags=["input"])
    def get_config():
        """Limits the UI shows and checks against (so it hard-codes none of them)."""
        return ConfigResponse(
            api_version=API_VERSION, min_plies=settings.min_plies,
            max_plies=settings.max_plies, max_pgn_bytes=settings.max_pgn_bytes,
            max_upload_bytes=settings.max_upload_bytes,
            max_games_per_file=settings.max_games_per_file,
            short_game_plies=SHORT_GAME_PLIES, engine_depth=settings.depth,
            analysis_timeout_s=settings.analysis_timeout_s)

    @app.get("/health", tags=["operations"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"])
    def ready():
        engine_ok = bool(app.state.pool and app.state.pool.ready())
        model_ok = bool(app.state.model and app.state.model.ready())
        body = {"status": "ready" if engine_ok and model_ok else "not_ready",
                "engine": engine_ok, "model": model_ok}
        return JSONResponse(status_code=200 if engine_ok and model_ok else 503, content=body)

    @app.get("/metrics", tags=["operations"])
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app

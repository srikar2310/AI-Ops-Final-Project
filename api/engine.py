"""A small pool of Stockfish engines with crash and hang recovery.

Each engine is single-threaded, with the same depth and hash size as the
training data (params.yaml), and analyses one game at a time. Concurrent
requests use different engines; if all are busy, a request waits a few
seconds and then gets HTTP 503.

Failure handling:
  * an engine that crashes or hangs mid-game is discarded and replaced, and the
    game is analysed again on a fresh engine (one retry); if that fails too the
    request gets HTTP 503 (EngineFailure)
  * a game that takes too long overall gets HTTP 504 (AnalysisTimeout); the
    engine itself is fine and goes back to the pool
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import List

import chess
import chess.engine

from . import metrics
from .errors import AnalysisTimeout, APIError, EngineFailure, PoolBusyError

log = logging.getLogger("api.engine")

MAX_ATTEMPTS = 2
# Anything that means "the engine process is broken or stuck"
_ENGINE_ERRORS = (chess.engine.EngineError, chess.engine.EngineTerminatedError,
                  TimeoutError, OSError, EOFError)


@dataclass
class Analysis:
    evals: List[int]     # White-perspective centipawns, one per ply (training format)
    seconds: float
    positions: int       # positions actually searched (checkmate/stalemate are not)


def _popen_stockfish(settings):
    engine = chess.engine.SimpleEngine.popen_uci(
        settings.engine_path, timeout=settings.position_timeout_s)
    try:
        engine.configure({"Threads": 1, "Hash": settings.hash_mb})
    except Exception:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass
        raise
    return engine


class EnginePool:
    def __init__(self, settings, engine_factory=None):
        self.s = settings
        self._factory = engine_factory or (lambda: _popen_stockfish(settings))
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._alive = 0
        self._started = False
        self.engine_name = "unknown"

    # ---------------------------------------------------------- lifecycle
    def start(self):
        """Start all engines. Raises if Stockfish cannot be launched (fail fast)."""
        for _ in range(self.s.pool_size):
            self._q.put(self._spawn())
        self._started = True
        metrics.ENGINES_AVAILABLE.set(self._q.qsize())
        log.info("Engine pool ready: %d x %s | depth %d | hash %d MB",
                 self.s.pool_size, self.engine_name, self.s.depth, self.s.hash_mb)

    def close(self):
        while True:
            try:
                engine = self._q.get_nowait()
            except queue.Empty:
                break
            if engine is not None:
                try:
                    engine.quit()
                except Exception:  # noqa: BLE001
                    self._close_quietly(engine)
        self._started = False

    def ready(self):
        """True if at least one engine is alive or can be (re)started right now."""
        if not self._started:
            return False
        if self._alive > 0:
            return True
        try:
            self._release(self._acquire(timeout=0.5), healthy=True)
            return True
        except APIError:
            return False

    # ----------------------------------------------------------- internals
    def _spawn(self):
        engine = self._factory()
        with self._lock:
            self._alive += 1
        self.engine_name = getattr(engine, "id", {}).get("name", "unknown")
        return engine

    @staticmethod
    def _close_quietly(engine):
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass

    def _discard(self, engine):
        self._close_quietly(engine)
        with self._lock:
            self._alive -= 1

    def _acquire(self, timeout=None):
        timeout = self.s.acquire_timeout_s if timeout is None else timeout
        try:
            slot = self._q.get(timeout=timeout)
        except queue.Empty:
            raise PoolBusyError(
                "The analysis service is busy. Please try again in a few seconds.") from None
        metrics.ENGINES_AVAILABLE.set(self._q.qsize())
        if slot is None:  # a previously failed engine: start a replacement now
            try:
                slot = self._spawn()
                metrics.STOCKFISH_RESTARTS.inc()
                log.info("Started a replacement Stockfish engine")
            except Exception as exc:  # noqa: BLE001
                self._q.put(None)
                metrics.ENGINES_AVAILABLE.set(self._q.qsize())
                log.exception("Could not start a replacement engine")
                raise EngineFailure("The chess engine could not be started.") from exc
        return slot

    def _release(self, engine, healthy):
        if healthy:
            self._q.put(engine)
        else:
            self._discard(engine)
            self._q.put(None)  # placeholder: respawned lazily on next acquire
        metrics.ENGINES_AVAILABLE.set(self._q.qsize())

    # ------------------------------------------------------------ analysis
    def _score(self, engine, board, limit, token):
        """(centipawns from White's view, was_searched), same rules as Stockfish_train_eval.py."""
        if board.is_checkmate():
            # Side to move is mated -> the player who just moved won.
            mate = self.s.mate_score
            return (mate if board.turn == chess.BLACK else -mate), False
        if board.is_stalemate() or board.is_insufficient_material():
            return 0, False
        started = time.monotonic()
        info = engine.analyse(board, limit, game=token)
        metrics.STOCKFISH_POSITION_SECONDS.observe(time.monotonic() - started)
        metrics.STOCKFISH_POSITIONS.inc()
        return int(info["score"].white().score(mate_score=self.s.mate_score)), True

    def _run(self, engine, moves, token):
        board = chess.Board()
        limit = chess.engine.Limit(depth=self.s.depth)
        started = time.monotonic()
        evals, searched = [], 0
        for ply, uci in enumerate(moves, 1):
            if time.monotonic() - started >= self.s.analysis_timeout_s:
                raise AnalysisTimeout(
                    f"Analysing the game took longer than {self.s.analysis_timeout_s:.0f} s "
                    f"(stopped at move {ply} of {len(moves)}).")
            board.push_uci(uci)
            score, was_searched = self._score(engine, board, limit, token)
            evals.append(score)
            searched += was_searched
        return Analysis(evals=evals, seconds=time.monotonic() - started, positions=searched)

    def analyse_game(self, moves, token):
        """Evaluate every position of a game. `token` identifies the game to the
        engine (hash is kept within a game and cleared between games)."""
        last_exc = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            engine = self._acquire()
            healthy = True
            try:
                return self._run(engine, moves, token)
            except AnalysisTimeout:
                raise                      # engine is fine; release it as healthy
            except _ENGINE_ERRORS as exc:
                healthy = False
                last_exc = exc
                metrics.STOCKFISH_FAILURES.inc()
                log.error("Engine failure on attempt %d/%d: %r", attempt, MAX_ATTEMPTS, exc)
            finally:
                self._release(engine, healthy)
        raise EngineFailure(
            "The chess engine failed while analysing this game. Please try again."
        ) from last_exc

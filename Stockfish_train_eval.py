"""Evaluate every position of every game in PGN shards with local Stockfish.

Settings come from params.yaml (section `eval`) so DVC can track them.
Output: one JSONL file per input shard (resumable) plus a metrics JSON.
"""
import argparse
import json
import logging
import os
import subprocess
import time
from multiprocessing import Pool

import chess
import chess.engine
import chess.pgn
import yaml

log = logging.getLogger("stockfish_eval")


def load_params(path=None):
    """Load the `eval` section of params.yaml (override path with PARAMS_FILE)."""
    path = path or os.environ.get("PARAMS_FILE", "params.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["eval"]


P = load_params()
ENGINE_PATH = P["engine_path"]
DEPTH = int(P["depth"])
HASH_MB = int(P["hash_mb"])
MATE_SCORE = int(P["mate_score"])
CACHE_PLIES = int(P["cache_plies"])
CACHE_MAX = 500_000

_engine = None
_cache = {}


def init_worker():
    """Start one single-threaded Stockfish per worker process."""
    global _engine
    if _engine is not None:
        try:
            _engine.close()
        except Exception:
            pass
    _engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    _engine.configure({"Threads": 1, "Hash": HASH_MB})


def _eval_game(task):
    shard, gid, w_elo, b_elo, moves = task
    board = chess.Board()
    limit = chess.engine.Limit(depth=DEPTH)
    evals = []

    for ply, uci in enumerate(moves, 1):
        board.push_uci(uci)

        if board.is_checkmate():
            # Side to move is mated -> the player who just moved won.
            score = MATE_SCORE if board.turn == chess.BLACK else -MATE_SCORE
        elif board.is_stalemate() or board.is_insufficient_material():
            score = 0
        else:
            key = board.epd() if ply <= CACHE_PLIES else None
            if key is not None and key in _cache:
                score = _cache[key]
            else:
                info = _engine.analyse(board, limit, game=gid)
                score = info["score"].white().score(mate_score=MATE_SCORE)
                if key is not None and len(_cache) < CACHE_MAX:
                    _cache[key] = score
        evals.append(score)

    return {
        "shard": shard,
        "id": gid,
        "white_elo": w_elo,
        "black_elo": b_elo,
        "moves": moves,
        "evals_white_cp": evals,  # evals[i] = score after ply i+1, White's view
    }


def eval_game(task):
    """Never raise: return an error record so one bad game can't kill the run."""
    try:
        return _eval_game(task)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, (chess.engine.EngineError, chess.engine.EngineTerminatedError)):
            try:
                init_worker()  # engine likely died; restart it
            except Exception:
                pass
        return {"shard": task[0], "id": task[1], "error": repr(exc)}


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass  # ignore a possibly truncated last line
    return done


def task_stream(input_dir, output_dir, limit):
    count = 0
    shards = sorted(f for f in os.listdir(input_dir) if f.endswith(".pgn"))
    for shard in shards:
        done = load_done(os.path.join(output_dir, shard.replace(".pgn", ".jsonl")))
        with open(os.path.join(input_dir, shard), "r",
                  encoding="utf-8", errors="ignore") as f:
            idx = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                idx += 1
                gid = game.headers.get("Site") or f"{shard}:{idx}"
                if gid in done:
                    continue
                moves = [m.uci() for m in game.mainline_moves()]
                yield (shard, gid,
                       game.headers.get("WhiteElo", ""),
                       game.headers.get("BlackElo", ""),
                       moves)
                count += 1
                if limit and count >= limit:
                    return


def engine_name():
    eng = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    try:
        return eng.id.get("name", "unknown")
    finally:
        eng.quit()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_meta(path, name, games, plies, failed, seconds):
    """Write run metrics, accumulating across resumed runs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prev = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            prev = {}
    games += prev.get("games", 0)
    plies += prev.get("positions", 0)
    seconds += prev.get("seconds", 0.0)
    meta = {
        "stockfish": name,
        "depth": DEPTH,
        "hash_mb": HASH_MB,
        "cache_plies": CACHE_PLIES,
        "games": games,
        "positions": plies,
        "failed_games_last_run": failed,
        "seconds": round(seconds, 1),
        "positions_per_sec": round(plies / seconds, 1) if seconds else 0,
        "games_per_sec": round(games / seconds, 3) if seconds else 0,
        "git_commit": git_commit(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset_clean/train")
    ap.add_argument("--output", default="dataset_clean/train_evaluated")
    ap.add_argument("--meta", default="metrics/eval_train.json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--limit", type=int, default=0, help="games to process (benchmark)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    name = engine_name()
    pinned = P.get("stockfish_version", "auto")
    if pinned != "auto" and pinned not in name:
        log.warning("Engine reports %r but params.yaml pins %r", name, pinned)
    log.info("%s | depth %d | %d workers (1 thread each)", name, DEPTH, args.workers)

    outs = {}
    start = time.time()
    games = plies = failed = 0

    with Pool(args.workers, initializer=init_worker) as pool:
        stream = task_stream(args.input, args.output, args.limit)
        for res in pool.imap(eval_game, stream, chunksize=4):
            if "error" in res:
                failed += 1
                log.error("Game %s failed: %s", res["id"], res["error"])
                continue  # not written, so a rerun retries it

            shard = res.pop("shard")
            if shard not in outs:
                outs[shard] = open(
                    os.path.join(args.output, shard.replace(".pgn", ".jsonl")),
                    "a", encoding="utf-8")
            outs[shard].write(json.dumps(res) + "\n")

            games += 1
            plies += len(res["moves"])
            if games % 200 == 0:
                for fh in outs.values():
                    fh.flush()
                el = time.time() - start
                log.info("%d games | %.0f pos/s | %.2f games/s",
                         games, plies / el, games / el)

    for fh in outs.values():
        fh.close()

    seconds = time.time() - start
    meta = write_meta(args.meta, name, games, plies, failed, seconds)
    log.info("Done: %d games, %d positions, %d failed in %.1f min",
             games, plies, failed, seconds / 60)
    log.info("Metrics written to %s: %s", args.meta, meta)


if __name__ == "__main__":
    main()
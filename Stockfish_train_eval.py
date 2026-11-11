"""Evaluate every position of every game in PGN shards with local Stockfish.

Settings come from params.yaml (section `eval`) so DVC can track them.

Run with no arguments to process train, val and test in one go:
    python Stockfish_train_eval.py

Output: one PGN file per input shard (same file name) containing only
WhiteElo, BlackElo, Result, Evals and the moves. Example game:

    [WhiteElo "1500"]
    [BlackElo "1480"]
    [Result "1-0"]
    [Evals "20,15,30,..."]

    1. e4 e5 2. Nf3 Nc6 ... 1-0

Evals holds one centipawn score per ply (White's point of view), taken after
that ply is played. Forced mates are stored as +/-mate_score.
The run is resumable: games already present in an output shard are skipped.
"""
import argparse
import logging
import os
import subprocess
import time
import json
from multiprocessing import Pool

import chess
import chess.engine
import chess.pgn
import yaml

log = logging.getLogger("stockfish_eval")

BLOCK_END = "\n\n\n"  # unique terminator written after every game


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
    shard, uci_moves, w_elo, b_elo, result, movetext = task
    board = chess.Board()
    limit = chess.engine.Limit(depth=DEPTH)
    evals = []

    for ply, uci in enumerate(uci_moves, 1):
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
                info = _engine.analyse(board, limit, game=movetext)
                score = info["score"].white().score(mate_score=MATE_SCORE)
                if key is not None and len(_cache) < CACHE_MAX:
                    _cache[key] = score
        evals.append(score)

    block = (
        f'[WhiteElo "{w_elo}"]\n'
        f'[BlackElo "{b_elo}"]\n'
        f'[Result "{result}"]\n'
        f'[Evals "{",".join(map(str, evals))}"]\n'
        f"\n{movetext}{BLOCK_END}"
    )
    return {"shard": shard, "block": block, "plies": len(evals)}


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
        return {"shard": task[0], "game": task[5][:60], "error": repr(exc)}


def load_done(path):
    """Return the movetexts already written to `path`.

    A partially written trailing game (e.g. after a crash) is dropped so it
    gets evaluated again.
    """
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    parts = data.split(BLOCK_END)
    tail = parts.pop()  # "" if the file ended cleanly, else a partial game
    if tail:
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(p + BLOCK_END for p in parts))
        log.warning("Dropped a partially written game at the end of %s", path)
    for block in parts:
        lines = block.strip().split("\n")
        done.add(lines[-1])  # the movetext line identifies the game
    return done


def task_stream(input_dir, output_dir, limit):
    count = 0
    shards = sorted(f for f in os.listdir(input_dir) if f.endswith(".pgn"))
    for shard in shards:
        done = load_done(os.path.join(output_dir, shard))
        with open(os.path.join(input_dir, shard), "r",
                  encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                if "FEN" in game.headers:
                    continue  # only standard starting positions are supported
                moves = list(game.mainline_moves())
                if not moves:
                    continue
                try:
                    text = game.board().variation_san(moves)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Skipping unparsable game in %s: %r", shard, exc)
                    continue
                result = game.headers.get("Result", "*")
                movetext = f"{text} {result}"
                if movetext in done:
                    continue
                yield (shard,
                       [m.uci() for m in moves],
                       game.headers.get("WhiteElo", "?"),
                       game.headers.get("BlackElo", "?"),
                       result,
                       movetext)
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


def run_split(pool, label, input_dir, output_dir, meta_path, engine, limit):
    """Evaluate every shard in input_dir and write results to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    log.info("[%s] %s -> %s", label, input_dir, output_dir)

    outs = {}
    start = time.time()
    games = plies = failed = 0

    for res in pool.imap(eval_game, task_stream(input_dir, output_dir, limit),
                         chunksize=4):
        if "error" in res:
            failed += 1
            log.error("[%s] game '%s...' failed: %s", label, res["game"], res["error"])
            continue  # not written, so a rerun retries it

        shard = res["shard"]
        if shard not in outs:
            outs[shard] = open(os.path.join(output_dir, shard), "a", encoding="utf-8")
        outs[shard].write(res["block"])

        games += 1
        plies += res["plies"]
        if games % 200 == 0:
            for fh in outs.values():
                fh.flush()
            el = time.time() - start
            log.info("[%s] %d games | %.0f pos/s | %.2f games/s",
                     label, games, plies / el, games / el)

    for fh in outs.values():
        fh.close()

    seconds = time.time() - start
    meta = write_meta(meta_path, engine, games, plies, failed, seconds)
    log.info("[%s] done: %d games, %d positions, %d failed in %.1f min",
             label, games, plies, failed, seconds / 60)
    log.info("[%s] metrics -> %s: %s", label, meta_path, meta)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset_clean",
                    help="folder containing the split folders")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="splits to process when --input is not given")
    ap.add_argument("--metrics-dir", default="metrics")
    ap.add_argument("--input", help="process just this folder (used by the DVC stage)")
    ap.add_argument("--output", help="output folder for --input")
    ap.add_argument("--meta", help="metrics JSON for --input")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--limit", type=int, default=0,
                    help="games per split to process (benchmark)")
    args = ap.parse_args()

    name = engine_name()
    pinned = P.get("stockfish_version", "auto")
    if pinned != "auto" and pinned not in name:
        log.warning("Engine reports %r but params.yaml pins %r", name, pinned)
    log.info("%s | depth %d | %d workers (1 thread each)", name, DEPTH, args.workers)

    if args.input:
        jobs = [(os.path.basename(os.path.normpath(args.input)), args.input,
                 args.output or args.input + "_evaluated",
                 args.meta or os.path.join(args.metrics_dir, "eval_custom.json"))]
    else:
        jobs = [(s, os.path.join(args.root, s),
                 os.path.join(args.root, f"{s}_evaluated"),
                 os.path.join(args.metrics_dir, f"eval_{s}.json"))
                for s in args.splits]

    with Pool(args.workers, initializer=init_worker) as pool:
        for label, inp, out, meta in jobs:
            run_split(pool, label, inp, out, meta, name, args.limit)


if __name__ == "__main__":
    main()
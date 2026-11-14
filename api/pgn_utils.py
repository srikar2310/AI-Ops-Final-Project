"""Parse and validate an uploaded PGN.

Everything that can be wrong with user input is caught here and reported as a
BadPGNError with a message the UI can show to a non-technical user.
"""
import io
from dataclasses import dataclass
from typing import List, Optional

import chess
import chess.pgn

from .errors import BadPGNError

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}


@dataclass
class ParsedGame:
    moves: List[str]               # UCI moves of the main line
    result: str                    # "1-0", "0-1", "1/2-1/2" or "*" (unknown)
    result_inferred: bool          # True if we derived it from the final position
    white: str
    black: str
    white_elo: Optional[int]
    black_elo: Optional[int]

    @property
    def plies(self):
        return len(self.moves)


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_result(board):
    """Result implied by the final position, or '*' if the game isn't over."""
    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"  # side to move is mated
    if board.is_stalemate() or board.is_insufficient_material():
        return "1/2-1/2"
    return "*"


def parse_pgn(text, *, max_bytes, min_plies, max_plies):
    """Return a ParsedGame or raise BadPGNError."""
    if not isinstance(text, str) or not text.strip():
        raise BadPGNError("The PGN is empty. Paste the full text of one game.")
    text = text.lstrip("\ufeff").strip()
    if len(text.encode("utf-8", "ignore")) > max_bytes:
        raise BadPGNError(f"The PGN is too large (the limit is {max_bytes} bytes).")

    stream = io.StringIO(text)
    try:
        game = chess.pgn.read_game(stream)
    except Exception as exc:  # noqa: BLE001 - python-chess can raise several types
        raise BadPGNError(f"Could not read the PGN: {exc}") from exc
    if game is None:
        raise BadPGNError("No game was found in the text.")

    try:
        extra = chess.pgn.read_game(stream)
    except Exception:  # noqa: BLE001
        extra = None
    if extra is not None and extra.next() is not None:
        raise BadPGNError("The text contains more than one game. Send one game at a time.")

    variant = game.headers.get("Variant", "Standard")
    if variant.strip().lower() not in ("standard", "chess"):
        raise BadPGNError(f"Only standard chess is supported (this game is '{variant}').")
    try:
        start = game.board()
    except ValueError as exc:
        raise BadPGNError(f"The starting position is invalid: {exc}") from exc
    if start.fen() != chess.STARTING_FEN:
        raise BadPGNError("Games that start from a custom position are not supported.")

    if game.errors:
        raise BadPGNError(
            f"The game contains an invalid or illegal move: {str(game.errors[0])[:200]}")

    moves = [move.uci() for move in game.mainline_moves()]
    if len(moves) < min_plies:
        raise BadPGNError(
            f"The game is too short: it has {len(moves)} half-moves and the model "
            f"needs at least {min_plies}.")
    if len(moves) > max_plies:
        raise BadPGNError(
            f"The game is too long: it has {len(moves)} half-moves (limit {max_plies}).")

    tag = game.headers.get("Result", "*").strip()
    result, inferred = (tag, False) if tag in VALID_RESULTS else ("*", False)
    if result == "*":
        guess = infer_result(game.end().board())
        if guess != "*":
            result, inferred = guess, True

    return ParsedGame(
        moves=moves,
        result=result,
        result_inferred=inferred,
        white=game.headers.get("White", "White"),
        black=game.headers.get("Black", "Black"),
        white_elo=_int_or_none(game.headers.get("WhiteElo")),
        black_elo=_int_or_none(game.headers.get("BlackElo")),
    )

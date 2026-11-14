"""Request and response bodies (also the API's OpenAPI documentation at /docs)."""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Body of POST /predict."""
    pgn: str = Field(..., description="The full text of ONE game in PGN format.")
    player: Literal["white", "black", "both"] = Field(
        "both", description="Which player(s) to estimate the rating of.")


class PlayerPrediction(BaseModel):
    """Estimated rating of one player, next to the rating in the PGN."""
    side: Literal["white", "black"]
    name: str
    predicted_rating: float
    actual_rating: Optional[int] = Field(
        None, description="The rating from the PGN headers, if present.")


class GameInfo(BaseModel):
    """Facts about the submitted game."""
    plies: int
    result: str
    result_known: bool
    result_inferred: bool


class AnalysisInfo(BaseModel):
    """How Stockfish analysed the game."""
    engine: str
    depth: int
    positions_searched: int
    seconds: float
    evals_white_cp: List[int] = Field(
        ..., description="Engine evaluation after each ply, in centipawns, White's view.")


class ModelInfo(BaseModel):
    """Which model service answered and how long it took."""
    mode: str
    target: str
    seconds: float


class MoveInfo(BaseModel):
    """One half-move with its evaluation and quality label."""
    ply: int = Field(..., description="1-based half-move number.")
    side: Literal["white", "black"]
    san: str
    eval_cp: int = Field(..., description="Evaluation after this move, centipawns, White's view.")
    white_win_prob: float = Field(..., description="White's win chance after this move (0..1).")
    loss: float = Field(..., description="Win chance the mover gave away with this move (0..1).")
    label: Literal["ok", "inaccuracy", "mistake", "blunder"]


class PredictResponse(BaseModel):
    """Body of a successful POST /predict."""
    request_id: str
    players: List[PlayerPrediction]
    game: GameInfo
    analysis: AnalysisInfo
    model: ModelInfo
    moves: List[MoveInfo] = []
    move_summary: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Counts of inaccuracies, mistakes and blunders per side.")
    warnings: List[str] = []


# ---- input helpers
class ErrorInfo(BaseModel):
    """Machine-readable code plus a message for the user."""
    code: str
    message: str


class ValidateRequest(BaseModel):
    """Body of POST /validate."""
    pgn: str = Field(..., description="Text to check. Nothing is analysed.")


class ValidateResponse(BaseModel):
    """Result of POST /validate: valid, or the reason it is not."""
    valid: bool
    error: Optional[ErrorInfo] = None
    plies: Optional[int] = None
    result: Optional[str] = None
    result_known: Optional[bool] = None
    result_inferred: Optional[bool] = None
    white: Optional[str] = None
    black: Optional[str] = None
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    warnings: List[str] = []


class SplitRequest(BaseModel):
    """Body of POST /games/split."""
    text: str = Field(..., description="Contents of a PGN file with one or more games.")


class GameListItem(BaseModel):
    """One game found in an uploaded file."""
    index: int
    white: str
    black: str
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    result: str
    plies: int
    usable: bool
    problem: Optional[str] = None
    pgn: str = Field(..., description="This game alone, ready to send to /predict.")


class SplitResponse(BaseModel):
    """Games found in an uploaded file."""
    games: List[GameListItem]
    truncated: bool = Field(..., description="True if the file had more games than the limit.")
    limit: int


class ConfigResponse(BaseModel):
    """Limits the UI reads from GET /config."""
    api_version: str
    min_plies: int
    max_plies: int
    max_pgn_bytes: int
    max_upload_bytes: int
    max_games_per_file: int
    short_game_plies: int
    engine_depth: int
    analysis_timeout_s: float

"""Request and response bodies (also the API's OpenAPI documentation at /docs)."""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    pgn: str = Field(..., description="The full text of ONE game in PGN format.")
    player: Literal["white", "black", "both"] = Field(
        "both", description="Which player(s) to estimate the rating of.")


class PlayerPrediction(BaseModel):
    side: Literal["white", "black"]
    name: str
    predicted_rating: float
    actual_rating: Optional[int] = Field(
        None, description="The rating from the PGN headers, if present.")


class GameInfo(BaseModel):
    plies: int
    result: str
    result_known: bool
    result_inferred: bool


class AnalysisInfo(BaseModel):
    engine: str
    depth: int
    positions_searched: int
    seconds: float
    evals_white_cp: List[int] = Field(
        ..., description="Engine evaluation after each ply, in centipawns, White's view.")


class ModelInfo(BaseModel):
    mode: str
    target: str
    seconds: float


class MoveInfo(BaseModel):
    ply: int = Field(..., description="1-based half-move number.")
    side: Literal["white", "black"]
    san: str
    eval_cp: int = Field(..., description="Evaluation after this move, centipawns, White's view.")
    white_win_prob: float = Field(..., description="White's win chance after this move (0..1).")
    loss: float = Field(..., description="Win chance the mover gave away with this move (0..1).")
    label: Literal["ok", "inaccuracy", "mistake", "blunder"]


class PredictResponse(BaseModel):
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
    code: str
    message: str


class ValidateRequest(BaseModel):
    pgn: str = Field(..., description="Text to check. Nothing is analysed.")


class ValidateResponse(BaseModel):
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
    text: str = Field(..., description="Contents of a PGN file with one or more games.")


class GameListItem(BaseModel):
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
    games: List[GameListItem]
    truncated: bool = Field(..., description="True if the file had more games than the limit.")
    limit: int


class ConfigResponse(BaseModel):
    api_version: str
    min_plies: int
    max_plies: int
    max_pgn_bytes: int
    max_upload_bytes: int
    max_games_per_file: int
    short_game_plies: int
    engine_depth: int
    analysis_timeout_s: float

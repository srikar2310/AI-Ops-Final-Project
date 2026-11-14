"""Request and response bodies (also the API's OpenAPI documentation at /docs)."""
from typing import List, Literal, Optional

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


class PredictResponse(BaseModel):
    request_id: str
    players: List[PlayerPrediction]
    game: GameInfo
    analysis: AnalysisInfo
    model: ModelInfo
    warnings: List[str] = []

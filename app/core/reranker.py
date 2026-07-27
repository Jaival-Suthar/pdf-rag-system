from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from app.config import Settings


@dataclass(frozen=True)
class RankedChunk:
    index: int
    score: float


class _RerankerModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> object:
        ...


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.re_rank_enabled
        self._model_name = settings.re_rank_model_name
        self._model: _RerankerModel | None = None

    def rank(self, query: str, passages: list[str]) -> list[RankedChunk]:
        if not self._enabled or not passages:
            return [RankedChunk(index=index, score=0.0) for index in range(len(passages))]
        from sentence_transformers import CrossEncoder

        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        pairs = [(query, passage) for passage in passages]
        scores = cast(Sequence[float], self._model.predict(pairs))
        return [RankedChunk(index=index, score=float(score)) for index, score in enumerate(scores)]

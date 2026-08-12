from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.core.reranker import RankedChunk
from app.core.vectorstore import RetrievalFilters, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model_name: str
    embedding_version: str
    candidate_k: int
    top_k: int
    similarity_threshold: float
    reranker_enabled: bool
    reranker_model_name: str | None


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    candidate_chunks: list[RetrievedChunk]
    embedding_latency_ms: int
    retrieval_latency_ms: int
    rerank_latency_ms: int
    retrieval_config: RetrievalConfig


class EmbedderLike(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class RerankerLike(Protocol):
    def rank(self, query: str, passages: list[str]) -> list[RankedChunk]: ...


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderLike,
        vectorstore: VectorStore,
        reranker: RerankerLike | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalResult:
        effective_top_k = top_k if top_k is not None else self._settings.retrieval_top_k_default

        if similarity_threshold is None:
            effective_threshold = self._settings.retrieval_similarity_threshold
        else:
            effective_threshold = similarity_threshold

        reranking_enabled = self._reranker is not None and self._settings.re_rank_enabled

        candidate_k = (
            max(effective_top_k, self._settings.rerank_candidate_k)
            if reranking_enabled
            else effective_top_k
        )

        embedding_start = time.perf_counter()
        query_embedding = self._embedder.embed_texts([query])[0]
        embedding_latency_ms = int((time.perf_counter() - embedding_start) * 1000)

        retrieval_start = time.perf_counter()
        chunks = self._vectorstore.search(
            query_embedding,
            top_k=candidate_k,
            similarity_threshold=effective_threshold,
            filters=filters,
        )
        retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)

        rerank_latency_ms = 0
        candidate_chunks = list(chunks)

        if reranking_enabled and self._reranker is not None:
            rerank_start = time.perf_counter()

            ranked_passages = self._reranker.rank(
                query,
                [chunk.text for chunk in chunks],
            )

            rerank_latency_ms = int((time.perf_counter() - rerank_start) * 1000)

            chunks = [
                chunks[ranked_chunk.index]
                for ranked_chunk in sorted(
                    ranked_passages,
                    key=lambda item: item.score,
                    reverse=True,
                )[:effective_top_k]
            ]

        logger.info(
            "retrieval_completed",
            extra={
                "embedding_latency_ms": embedding_latency_ms,
                "retrieval_latency_ms": retrieval_latency_ms,
                "rerank_latency_ms": rerank_latency_ms,
                "candidate_k": candidate_k,
                "final_top_k": effective_top_k,
                "reranker_enabled": reranking_enabled,
                "total_ms": (embedding_latency_ms + retrieval_latency_ms + rerank_latency_ms),
            },
        )

        return RetrievalResult(
            chunks=chunks,
            candidate_chunks=candidate_chunks,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            retrieval_config=RetrievalConfig(
                embedding_model_name=self._settings.embedding_model_name,
                embedding_version=self._settings.embedding_version,
                candidate_k=candidate_k,
                top_k=effective_top_k,
                similarity_threshold=effective_threshold,
                reranker_enabled=reranking_enabled,
                reranker_model_name=(
                    self._settings.re_rank_model_name if reranking_enabled else None
                ),
            ),
        )

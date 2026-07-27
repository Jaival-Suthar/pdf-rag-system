from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.core.vectorstore import RetrievalFilters, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    embedding_latency_ms: int
    retrieval_latency_ms: int


class EmbedderLike(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderLike,
        vectorstore: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vectorstore = vectorstore

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
        embedding_start = time.perf_counter()
        query_embedding = self._embedder.embed_texts([query])[0]
        embedding_latency_ms = int((time.perf_counter() - embedding_start) * 1000)

        retrieval_start = time.perf_counter()
        chunks = self._vectorstore.search(
            query_embedding,
            top_k=effective_top_k,
            similarity_threshold=effective_threshold,
            filters=filters,
        )
        retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)

        logger.info(
            "retrieval_completed",
            extra={
                "embedding_latency_ms": embedding_latency_ms,
                "retrieval_latency_ms": retrieval_latency_ms,
                "total_ms": embedding_latency_ms + retrieval_latency_ms,
            },
        )
        return RetrievalResult(
            chunks=chunks,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
        )

from __future__ import annotations

import logging
from time import perf_counter
from typing import cast

from fastapi import APIRouter, HTTPException, Request

from app.core.generation_client import GenerationProviderError
from app.core.vectorstore import RetrievalFilters
from app.logging_config import request_id_var
from app.models.schemas import ChatRequest, ChatResponse, LatencyMetrics, SourceChunk
from app.services import Services

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    if payload.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented in Milestone 1")

    services = cast(Services, request.app.state.services)
    request_id = request_id_var.get()
    total_start = perf_counter()

    retrieval_result = services.retriever.retrieve(
        payload.message,
        top_k=payload.top_k,
        similarity_threshold=services.settings.retrieval_similarity_threshold,
        filters=RetrievalFilters(doc_id=payload.doc_id),
    )

    prompt_bundle = services.prompt_builder.build(payload.message, retrieval_result.chunks)

    llm_start = perf_counter()
    try:
        answer = services.generation_client.generate(prompt_bundle.prompt).text
    except GenerationProviderError as exc:
        logger.exception("generation_failed", extra={"request_id": request_id})
        raise HTTPException(status_code=502, detail="Generation provider request failed") from exc
    llm_latency_ms = int((perf_counter() - llm_start) * 1000)
    total_latency_ms = int((perf_counter() - total_start) * 1000)

    latency = LatencyMetrics(
        embedding=retrieval_result.embedding_latency_ms,
        retrieval=retrieval_result.retrieval_latency_ms,
        rerank=retrieval_result.rerank_latency_ms,
        llm=llm_latency_ms,
        total=total_latency_ms,
    )
    source_models = [
        SourceChunk(
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            score=chunk.score,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
        )
        for chunk in prompt_bundle.included_sources
    ]

    logger.info(
        "chat_completed",
        extra={
            "request_id": request_id,
            "embedding_latency_ms": retrieval_result.embedding_latency_ms,
            "retrieval_latency_ms": retrieval_result.retrieval_latency_ms,
            "rerank_latency_ms": retrieval_result.rerank_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "total_ms": total_latency_ms,
        },
    )

    return ChatResponse(answer=answer, sources=source_models, latency_ms=latency)

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ErrorPayload(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorPayload


class HealthResponse(BaseModel):
    status: Literal["ok"]
    qdrant_connected: bool
    inference_lab_connected: bool


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: Literal["ready"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    doc_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


class LatencyMetrics(BaseModel):
    embedding: int
    retrieval: int
    rerank: int
    llm: int
    total: int


class SourceChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    text: str
    score: float
    page_number: int
    section_title: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: LatencyMetrics

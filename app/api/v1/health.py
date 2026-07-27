from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse
from app.services import Services

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    services = cast(Services, request.app.state.services)
    return HealthResponse(
        status="ok",
        qdrant_connected=services.vectorstore.ping(),
        inference_lab_connected=services.generation_client.is_reachable(),
    )

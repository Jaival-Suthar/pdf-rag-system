from __future__ import annotations

import hashlib
import logging
import uuid
from typing import cast

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.ingestion.extractor import PdfExtractionError
from app.models.schemas import UploadResponse
from app.services import Services

router = APIRouter()
logger = logging.getLogger(__name__)


def _is_pdf(file_name: str, content_type: str | None) -> bool:
    return file_name.lower().endswith(".pdf") or content_type == "application/pdf"


def _fingerprint_pdf_bytes(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


@router.post("/upload", response_model=UploadResponse)
def upload_pdf(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    if not _is_pdf(file.filename or "", file.content_type):
        raise HTTPException(status_code=422, detail="Only PDF uploads are supported")

    services = cast(Services, request.app.state.services)
    settings = services.settings
    doc_id = str(uuid.uuid4())
    filename = file.filename or f"{doc_id}.pdf"

    file.file.seek(0)
    pdf_bytes = file.file.read()
    document_fingerprint = _fingerprint_pdf_bytes(pdf_bytes)

    duplicate_exists = services.vectorstore.has_document_fingerprint(document_fingerprint)
    logger.info(
        "upload_received",
        extra={
            "request_id": getattr(request.state, "request_id", "-"),
            "doc_id": doc_id,
            "uploaded_filename": filename,
            "duplicate_exists": duplicate_exists,
            "duplicate_policy": settings.duplicate_upload_policy,
        },
    )

    if duplicate_exists and settings.duplicate_upload_policy == "reject":
        raise HTTPException(status_code=409, detail="Duplicate PDF upload detected")

    # Concurrent uploads of the same fingerprint can both pass the duplicate check before either
    # completes ingestion, so this milestone accepts that race window for simplicity.
    if duplicate_exists and settings.duplicate_upload_policy == "replace":
        services.vectorstore.delete_document_fingerprint(document_fingerprint)

    upload_path = settings.data_dir / "uploads" / f"{doc_id}.pdf"
    upload_path.write_bytes(pdf_bytes)

    try:
        result = services.pipeline.ingest_pdf(
            doc_id=doc_id,
            document_fingerprint=document_fingerprint,
            filename=filename,
            pdf_path=upload_path,
        )
    except PdfExtractionError as exc:
        logger.warning(
            "pdf_ingestion_failed",
            extra={"doc_id": doc_id, "uploaded_filename": filename},
            exc_info=True,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "pdf_ingestion_failed",
            extra={"doc_id": doc_id, "uploaded_filename": filename},
        )
        raise HTTPException(status_code=500, detail="Failed to ingest uploaded PDF") from exc

    return UploadResponse(
        doc_id=result.doc_id,
        filename=result.filename,
        chunk_count=len(result.chunks),
        status="ready",
    )

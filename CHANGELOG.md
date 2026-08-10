# Changelog

All notable changes to this repository will be documented in this file.

## [1.1.0] - 2026-08-10

### Changed

- Updated the README to match the implemented M1 retrieval, reranking, evaluation, and benchmarking flow.
- Clarified the distinction between rerank candidate pools and final `top_k` context.
- Corrected the offline evaluation section to describe the metrics currently computed by the code.
- Added v1.1.0 release notes.

## [1.0.0] - 2026-07-28

### Added

- Production-oriented PDF ingestion pipeline with PyMuPDF extraction, recursive chunking, embedding generation, and Qdrant indexing.
- Deterministic document fingerprints, document IDs, and duplicate handling policies.
- Semantic retrieval with metadata filtering, score normalization, and deterministic ordering.
- Prompt construction and generation orchestration through Inference Lab (M0) over REST.
- FastAPI endpoints for upload, chat, and health.
- Evaluation and benchmarking commands that emit JSON reports.
- Structured logging, request IDs, and typed error responses.

### Documentation

- Public README with architecture, setup, API examples, troubleshooting, and release guidance.
- Release notes for v1.0.0.

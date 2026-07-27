# pdf-rag-system

Local-first PDF RAG service for Milestone 1.

This repository is intentionally small and explicit: extraction, chunking, embeddings, retrieval,
generation, evaluation, and benchmarking are all visible as separate stages.

## Architecture Overview

```text
                          +----------------------+
                          |   POST /v1/upload    |
                          +----------+-----------+
                                     |
                                     v
                           +---------+----------+
                           |  Read PDF bytes    |
                           |  SHA256 fingerprint |
                           +---------+----------+
                                     |
                                     v
                           +---------+----------+
                           | PyMuPDF extraction  |
                           | skip empty pages    |
                           +---------+----------+
                                     |
                                     v
                           +---------+----------+
                           | Recursive chunking  |
                           | token-aware windows |
                           +---------+----------+
                                     |
                                     v
                           +---------+----------+
                           | SentenceTransformer |
                           | bge-small-en-v1.5   |
                           +---------+----------+
                                     |
                                     v
                           +---------+----------+
                           | Qdrant collection   |
                           | document + chunk IDs |
                           +---------+----------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
          +--------+--------+                 +--------+--------+
          |   POST /v1/chat |                 |  GET /v1/health  |
          +--------+--------+                 +--------+--------+
                   |                                   |
                   v                                   v
          +--------+--------+                 +--------+--------+
          | Retrieval +     |                 | Connectivity    |
          | PromptBuilder    |                 | checks          |
          +--------+--------+                 +-----------------+
                   |
                   v
          +--------+--------+
          | Generation API  |
          | localhost:4000  |
          +-----------------+
```

## Request Flow

```text
Upload PDF
  -> compute document_fingerprint
  -> duplicate check
  -> extract text
  -> recursive chunking
  -> embed chunks
  -> index chunks in Qdrant
  -> return document metadata

Chat request
  -> embed user question
  -> retrieve top-k chunks from Qdrant
  -> normalize scores and apply threshold
  -> build prompt with source citations
  -> call POST http://localhost:4000/v1/generate
  -> return answer + sources + latency metrics
```

## Setup

### With `uv`

```bash
uv sync --dev
uv run uvicorn app.main:app --reload --port 8000
```

### With Docker

```bash
docker compose up --build
```

The compose stack starts:

- Qdrant on `localhost:6333`
- the FastAPI app on `localhost:8000`

Milestone 0, Inference Lab, must already be running separately at
`http://localhost:4000/v1/generate`.

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

Key environment variables:

- `GENERATION_PROVIDER_URL`: generation backend URL, defaults to `http://localhost:4000/v1/generate`
- `GENERATION_TIMEOUT_SECONDS`: timeout for provider requests
- `GENERATION_RETRY_COUNT`: retry count for provider failures
- `EMBEDDING_MODEL_NAME`: defaults to `BAAI/bge-small-en-v1.5`
- `EMBEDDING_DEVICE`: defaults to `cpu`
- `CHUNK_MAX_TOKENS`: recursive chunk size
- `CHUNK_OVERLAP_TOKENS`: chunk overlap
- `RETRIEVAL_TOP_K_DEFAULT`: fallback top-k for retrieval
- `RETRIEVAL_SIMILARITY_THRESHOLD`: minimum normalized similarity score
- `DUPLICATE_UPLOAD_POLICY`: `reject`, `replace`, or `allow`
- `PROMPT_TOKEN_BUDGET`: prompt budget for the PromptBuilder

`EMBEDDING_DEVICE=cuda` can be useful during bulk ingestion if embeddings become the bottleneck.

## API Examples

### Health

```bash
curl http://localhost:8000/v1/health
```

### Upload

```bash
curl -F "file=@/path/to/document.pdf" http://localhost:8000/v1/upload
```

### Chat

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does the document say about setup?",
    "doc_id": null,
    "top_k": 5,
    "stream": false
  }'
```

## Evaluation

Run the evaluation suite and emit a JSON report:

```bash
uv run pdf-rag-eval --questions eval/questions.json --output eval-report.json
```

Metrics include:

- Retrieval Precision
- Recall@k
- Latency
- Answer Faithfulness
- Context Utilisation

## Benchmarks

Run the benchmark suite and emit a JSON report:

```bash
uv run pdf-rag-benchmark --output benchmark-report.json
```

The benchmark reports:

- upload latency
- extraction latency
- embedding latency
- retrieval latency
- generation latency
- total end-to-end latency

## Troubleshooting

- If `POST /v1/chat` returns `502`, confirm Inference Lab is still listening on `localhost:4000`.
- If uploads fail with `409`, the fingerprint already exists and `DUPLICATE_UPLOAD_POLICY=reject`.
- If Qdrant is unreachable, confirm the Docker container is running and port `6333` is free.
- If embeddings are slow, try `EMBEDDING_DEVICE=cuda` during local bulk ingestion.


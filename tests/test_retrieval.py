from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.config import Settings
from app.core.reranker import RankedChunk
from app.core.retrieval import Retriever
from app.core.vectorstore import RetrievalFilters, VectorStore, normalize_similarity_score


@dataclass
class _FakeResult:
    score: float
    payload: dict[str, object]


class _FakeClient:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []

    def get_collections(self) -> object:
        return type("Collections", (), {"collections": []})()

    def create_collection(self, **_: object) -> None:
        return None

    def search(self, **kwargs: object) -> list[_FakeResult]:
        self.search_calls.append(kwargs)
        return [
            _FakeResult(
                score=1.4,
                payload={
                    "doc_id": "doc-b",
                    "document_fingerprint": "fingerprint-b",
                    "filename": "b.pdf",
                    "chunk_id": "chunk-b",
                    "chunk_index": 2,
                    "text": "B chunk",
                    "page_number": 2,
                    "section_title": "Beta",
                    "source_path": "/tmp/b.pdf",
                },
            ),
            _FakeResult(
                score=1.0,
                payload={
                    "doc_id": "doc-a",
                    "document_fingerprint": "fingerprint-a",
                    "filename": "a.pdf",
                    "chunk_id": "chunk-a",
                    "chunk_index": 1,
                    "text": "A chunk",
                    "page_number": 1,
                    "section_title": "Alpha",
                    "source_path": "/tmp/a.pdf",
                },
            ),
            _FakeResult(
                score=0.2,
                payload={
                    "doc_id": "doc-c",
                    "document_fingerprint": "fingerprint-c",
                    "filename": "c.pdf",
                    "chunk_id": "chunk-c",
                    "chunk_index": 3,
                    "text": "C chunk",
                    "page_number": 3,
                    "section_title": "Gamma",
                    "source_path": "/tmp/c.pdf",
                },
            ),
        ]

    def scroll(self, **_: object) -> tuple[list[object], None]:
        return ([], None)

    def upsert(self, **_: object) -> None:
        return None

    def delete(self, **_: object) -> None:
        return None


class _FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 0.0] for text in texts]


class _FakeReranker:
    def rank(self, query: str, passages: list[str]) -> list[RankedChunk]:
        del query
        if len(passages) < 2:
            return [RankedChunk(index=index, score=0.0) for index in range(len(passages))]
        return [
            RankedChunk(index=1, score=2.0),
            RankedChunk(index=0, score=1.0),
        ]


def test_similarity_score_is_normalized() -> None:
    assert normalize_similarity_score(-0.3) == 0.0
    assert normalize_similarity_score(0.4) == 0.4
    assert normalize_similarity_score(1.7) == 1.0


def test_retriever_applies_threshold_and_deterministic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(qdrant_url="http://example", retrieval_similarity_threshold=0.3)
    vectorstore = VectorStore(settings)
    retriever = Retriever(settings, _FakeEmbedder(), vectorstore)

    result = retriever.retrieve(
        "query",
        top_k=2,
        similarity_threshold=0.3,
        filters=RetrievalFilters(doc_id="doc-a"),
    )

    assert [chunk.doc_id for chunk in result.chunks] == ["doc-a", "doc-b"]
    assert result.chunks[0].score == 1.0
    assert result.chunks[1].score == 1.0
    assert fake_client.search_calls
    assert fake_client.search_calls[0]["query_filter"] is not None


def test_retriever_preserves_order_when_reranking_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(
        qdrant_url="http://example",
        retrieval_similarity_threshold=0.3,
        re_rank_enabled=False,
    )
    vectorstore = VectorStore(settings)
    retriever = Retriever(settings, _FakeEmbedder(), vectorstore, reranker=_FakeReranker())

    result = retriever.retrieve("query", top_k=2, similarity_threshold=0.3)

    assert [chunk.doc_id for chunk in result.chunks] == ["doc-a", "doc-b"]


def test_retriever_uses_reranker_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeClient()

    def fake_qdrant_client(url: str) -> _FakeClient:
        _ = url
        return fake_client

    monkeypatch.setattr("app.core.vectorstore.QdrantClient", fake_qdrant_client)

    settings = Settings(
        qdrant_url="http://example",
        retrieval_similarity_threshold=0.3,
        re_rank_enabled=True,
    )
    vectorstore = VectorStore(settings)
    retriever = Retriever(settings, _FakeEmbedder(), vectorstore, reranker=_FakeReranker())

    result = retriever.retrieve("query", top_k=2, similarity_threshold=0.3)

    assert [chunk.doc_id for chunk in result.chunks] == ["doc-b", "doc-a"]

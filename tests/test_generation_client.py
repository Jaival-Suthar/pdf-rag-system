from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from app.config import Settings
from app.core.generation_client import GenerationClient


@dataclass
class _FakeResponse:
    text_value: str | None
    response_value: str | None = None
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://localhost:4000/v1/generate")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("bad response", request=request, response=response)

    def json(self) -> dict[str, str | None]:
        return {"text": self.text_value, "response": self.response_value}


class _FakeClient:
    attempts = 0

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc_type, exc, tb
        return None

    def post(self, url: str, json: dict[str, str]) -> _FakeResponse:
        _ = url, json
        type(self).attempts += 1
        if type(self).attempts == 1:
            request = httpx.Request("POST", url)
            raise httpx.ConnectError("temporary failure", request=request)
        return _FakeResponse("final answer")


def test_generation_client_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _FakeClient.attempts = 0

    client = GenerationClient(Settings(generation_retry_count=1, generation_timeout_seconds=1.0))
    result = client.generate("prompt")

    assert result.text == "final answer"
    assert _FakeClient.attempts == 2


class _ResponseFieldClient(_FakeClient):
    def post(self, url: str, json: dict[str, str]) -> _FakeResponse:
        _ = url, json
        return _FakeResponse(text_value=None, response_value="fallback answer")


def test_generation_client_accepts_response_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", _ResponseFieldClient)

    client = GenerationClient(Settings(generation_retry_count=0, generation_timeout_seconds=1.0))
    result = client.generate("prompt")

    assert result.text == "fallback answer"

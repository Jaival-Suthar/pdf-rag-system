from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    text: str


class GenerationProviderError(RuntimeError):
    """Raised when the external generation provider cannot be reached or returns invalid data."""


class GenerationClient:
    def __init__(self, settings: Settings) -> None:
        self._url = settings.generation_provider_url
        self._timeout = settings.generation_timeout_seconds
        self._retry_count = settings.generation_retry_count

    def generate(self, prompt: str) -> GenerationResult:
        last_error: Exception | None = None
        with httpx.Client(timeout=self._timeout) as client:
            for attempt in range(self._retry_count + 1):
                try:
                    response = client.post(self._url, json={"prompt": prompt})
                    response.raise_for_status()
                    payload = response.json()

                    text = payload.get("text")
                    if not isinstance(text, str):
                        raise ValueError("generation provider response missing text")
                    return GenerationResult(text=text)
                except (httpx.HTTPError, ValueError) as exc:
                    last_error = exc
                    if attempt >= self._retry_count:
                        break
                    backoff_seconds = 0.5 * (2**attempt)
                    time.sleep(backoff_seconds)
        raise GenerationProviderError("generation provider request failed") from last_error

    def is_reachable(self) -> bool:
        try:
            self.generate("healthcheck")
            return True
        except Exception:
            logger.exception("generation provider health check failed")
            return False

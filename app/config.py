from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "pdf-rag-system"
    app_env: str = "development"
    debug: bool = False
    data_dir: Path = Path("./data")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pdf_chunks"
    generation_provider_url: str = "http://localhost:4000/v1/generate"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_dimension: int = 384
    embedding_version: str = "bge-small-en-v1.5"
    re_rank_enabled: bool = False
    re_rank_model_name: str = "BAAI/bge-reranker-base"
    generation_timeout_seconds: float = Field(default=30.0, ge=1.0)
    generation_retry_count: int = Field(default=2, ge=0, le=5)
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 80
    prompt_token_budget: int = Field(default=3000, ge=512, le=100000)
    duplicate_upload_policy: Literal["reject", "replace", "allow"] = "reject"
    retrieval_top_k_default: int = Field(default=5, ge=1, le=50)
    rerank_candidate_k: int = Field(default=20, ge=1, le=100)
    retrieval_similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    def ensure_data_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_data_dirs()
    return settings

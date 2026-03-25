"""Centralized configuration for FinDocs using pydantic-settings.

All configuration is loaded from environment variables with an optional .env file.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """FinDocs application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM APIs
    ANTHROPIC_API_KEY: str
    OPENAI_API_KEY: str

    # Infrastructure
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "findocs"
    POSTGRES_URL: str = "postgresql+asyncpg://findocs:findocs@localhost:5432/findocs"
    REDIS_URL: str = "redis://localhost:6379"

    # Langfuse
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str = "http://localhost:3000"

    # HuggingFace
    HF_TOKEN: str
    HF_MODEL_REPO: str = "your-username/findocs-phi3-finetuned"

    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    FINETUNED_MODEL_PATH: str = "./models/phi3-findocs-adapter"
    FALLBACK_MODEL: str = "gpt-4o-mini"
    BASE_MODEL_FOR_FINETUNING: str = "microsoft/Phi-3-mini-4k-instruct"

    # Chunking
    CHUNK_SIZE_TOKENS: int = 512
    CHILD_CHUNK_SIZE_TOKENS: int = 128
    CHUNK_OVERLAP_TOKENS: int = 50

    # Retrieval
    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20
    RERANK_TOP_K: int = 5

    # Evaluation thresholds (CI gate)
    MIN_FAITHFULNESS: float = 0.80
    MIN_CONTEXT_PRECISION: float = 0.70
    MIN_ANSWER_RELEVANCE: float = 0.75

    # Monitoring
    DRIFT_ALERT_THRESHOLD_PCT: float = 5.0
    WEEKLY_EVAL_QUESTION_SAMPLE: int = 50

    # Slack (for drift alerts)
    SLACK_WEBHOOK_URL: str = ""

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()  # type: ignore[call-arg]

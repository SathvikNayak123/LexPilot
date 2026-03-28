from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # LLM Providers (3-tier router)
    groq_api_key: str = Field(..., description="Groq for Tier 2")
    anthropic_api_key: str = Field(..., description="Anthropic for Tier 3 + eval judge")
    openai_api_key: Optional[str] = Field(default=None, description="Optional: OpenAI for Agents SDK default")

    # Vector DB
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="lexpilot_chunks")

    # Knowledge Graph
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="lexpilot")

    # Infrastructure
    postgres_url: str = Field(default="postgresql+asyncpg://lexpilot:lexpilot@localhost:5432/lexpilot")
    redis_url: str = Field(default="redis://localhost:6379/0")
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = Field(default="http://localhost:3001")

    # Embedding
    embedding_model: str = Field(default="all-mpnet-base-v2")
    embedding_dim: int = Field(default=768)

    # Chunking
    child_chunk_tokens: int = Field(default=128)
    parent_chunk_tokens: int = Field(default=512)
    chunk_overlap_tokens: int = Field(default=50)
    semantic_threshold: float = Field(default=0.6)

    # Retrieval
    dense_top_k: int = Field(default=50)
    sparse_top_k: int = Field(default=50)
    rrf_k: int = Field(default=60)
    rerank_top_k: int = Field(default=20)
    final_top_k: int = Field(default=5)

    # General
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_port: int = Field(default=8000)


settings = Settings()

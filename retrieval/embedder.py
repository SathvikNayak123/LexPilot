from sentence_transformers import SentenceTransformer
import numpy as np
from config.config import settings


class EmbeddingService:
    """SentenceTransformer embedding service. Singleton."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer(settings.embedding_model)
        return cls._instance

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (n, 768) numpy array."""
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (768,) numpy array."""
        return self.model.encode(query, normalize_embeddings=True)

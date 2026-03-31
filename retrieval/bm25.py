from rank_bm25 import BM25Okapi
import numpy as np
from typing import Optional
import pickle
import structlog

logger = structlog.get_logger()


class BM25Encoder:
    """Builds BM25 sparse vectors for Qdrant sparse indexing."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.vocab: dict[str, int] = {}

    def fit(self, corpus: list[str]):
        """Fit BM25 on corpus of chunk texts."""
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

        # Build vocabulary mapping
        all_tokens = set()
        for doc in tokenized:
            all_tokens.update(doc)
        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}

        # Pre-compute per-token IDF from BM25's internal doc_freqs
        self.token_idf: dict[str, float] = {}
        n = len(tokenized)
        for token in self.vocab:
            df = sum(1 for doc in tokenized if token in doc)
            # Standard BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            self.token_idf[token] = np.log((n - df + 0.5) / (df + 0.5) + 1.0)

        logger.info("bm25_fitted", vocab_size=len(self.vocab), corpus_size=len(corpus))

    def encode_document(self, text: str) -> dict:
        """Encode a document into a sparse vector (indices + values)."""
        tokens = text.lower().split()
        token_counts = {}
        for t in tokens:
            if t in self.vocab:
                token_counts[t] = token_counts.get(t, 0) + 1

        indices = []
        values = []
        for token, count in token_counts.items():
            idx = self.vocab[token]
            idf = self.token_idf.get(token, 1.0)
            indices.append(idx)
            values.append(count * idf)

        return {"indices": indices, "values": values}

    def encode_query(self, query: str) -> dict:
        """Encode a query into a sparse vector."""
        return self.encode_document(query)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "vocab": self.vocab, "token_idf": self.token_idf}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.vocab = data["vocab"]
            self.token_idf = data.get("token_idf", {})

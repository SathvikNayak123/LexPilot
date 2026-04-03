"""
Build Corpus-Wide BM25 Index
============================
Fits a single BM25 encoder over all child chunks from Qdrant using an
efficient streaming IDF calculation (avoids rank_bm25's O(corpus x vocab)
internal loop). Re-encodes sparse vectors with correct corpus-wide IDF,
uploads them back to Qdrant, and saves the fitted encoder to
data/bm25_encoder.pkl.

Run ONCE after ingestion is complete, and again whenever new documents
are added (idempotent — overwrites the old encoder).

Usage:
    python scripts/build_bm25_index.py

Requires: Qdrant running with indexed chunks.
"""

import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BM25_SAVE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_encoder.pkl"

# BM25 Okapi parameters
K1 = 1.5
B = 0.75


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_efficient(texts: list[str]) -> dict:
    """
    Build BM25 vocabulary and IDF in two passes — O(corpus) not O(corpus x vocab).

    Returns a dict with 'vocab', 'token_idf', 'avgdl' for use by BM25Encoder.
    """
    n = len(texts)
    print(f"  Pass 1: building document-frequency table from {n:,} chunks...")

    # Pass 1: count document frequency per token (one loop)
    df: dict[str, int] = defaultdict(int)
    total_tokens = 0
    tokenized: list[list[str]] = []
    for text in texts:
        toks = _tokenize(text)
        tokenized.append(toks)
        total_tokens += len(toks)
        for tok in set(toks):   # set → count each token once per doc
            df[tok] += 1

    avgdl = total_tokens / n if n else 1.0

    # Build vocab (sorted for deterministic indices)
    vocab = {tok: idx for idx, tok in enumerate(sorted(df.keys()))}

    # Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    token_idf: dict[str, float] = {}
    for tok, freq in df.items():
        token_idf[tok] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

    print(f"  Vocab size: {len(vocab):,}  |  avgdl: {avgdl:.1f} tokens")
    return {"vocab": vocab, "token_idf": token_idf, "avgdl": avgdl, "tokenized": tokenized}


def encode_bm25_document(toks: list[str], vocab: dict, token_idf: dict,
                          avgdl: float, doc_len: int) -> dict:
    """BM25 Okapi TF-IDF sparse vector for a single document."""
    tf: dict[str, int] = defaultdict(int)
    for t in toks:
        if t in vocab:
            tf[t] += 1

    indices = []
    values = []
    for tok, count in tf.items():
        idf = token_idf.get(tok, 1.0)
        # BM25 TF normalisation
        tf_norm = (count * (K1 + 1)) / (count + K1 * (1 - B + B * doc_len / avgdl))
        indices.append(vocab[tok])
        values.append(idf * tf_norm)

    return {"indices": indices, "values": values}


def main():
    from retrieval.qdrant_store import QdrantStore
    from retrieval.bm25 import BM25Encoder

    print("=" * 60)
    print("  BM25 Corpus Index Builder")
    print("=" * 60)

    qdrant = QdrantStore()

    # Step 1: Scroll all points
    print("\n[1/4] Reading all chunks from Qdrant...")
    points = qdrant.scroll_all()
    if not points:
        print("  No points found. Run ingestion first.")
        sys.exit(1)
    print(f"  Found {len(points):,} chunks.")

    point_ids = [p.id for p in points]
    texts = [p.payload.get("content", "") for p in points]

    # Step 2: Build BM25 efficiently (two-pass, no rank_bm25 internal loop)
    print("\n[2/4] Building BM25 vocabulary and IDF...")
    bm25_data = build_bm25_efficient(texts)
    vocab = bm25_data["vocab"]
    token_idf = bm25_data["token_idf"]
    avgdl = bm25_data["avgdl"]
    tokenized = bm25_data["tokenized"]

    # Step 3: Encode sparse vectors for each chunk
    print("\n[3/4] Encoding sparse vectors...")
    updates = []
    empty_count = 0
    for i, (point_id, toks) in enumerate(zip(point_ids, tokenized)):
        sv = encode_bm25_document(toks, vocab, token_idf, avgdl, len(toks))
        if not sv["indices"]:
            empty_count += 1
        updates.append((point_id, sv))
        if (i + 1) % 10000 == 0:
            print(f"    Encoded {i + 1:,}/{len(points):,}...")

    if empty_count:
        print(f"  Warning: {empty_count} chunks produced empty sparse vectors.")

    # Step 4: Upload to Qdrant
    print(f"\n[4/4] Uploading {len(updates):,} sparse vectors to Qdrant (batches of 100)...")
    qdrant.update_sparse_vectors(updates)
    print("  Done.")

    # Save encoder for query-time use
    # Store as a compatible BM25Encoder pickle (vocab + token_idf; bm25=None is fine
    # since query-time only calls encode_document/encode_query which use vocab+token_idf)
    encoder = BM25Encoder()
    encoder.vocab = vocab
    encoder.token_idf = token_idf
    encoder.save(str(BM25_SAVE_PATH))
    print(f"\n  BM25 encoder saved: {BM25_SAVE_PATH}")

    print("\n" + "=" * 60)
    print("  BM25 index built. HybridSearchPipeline loads it automatically.")
    print("=" * 60)


if __name__ == "__main__":
    main()

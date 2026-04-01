"""InLegalBERT-based legal relation extractor.

Uses law-ai/InLegalBERT (BERT pre-trained on Indian legal corpora) to classify
the relation type between a citing judgment and each cited case it mentions.

Approach: zero-shot prototype embedding classification — no fine-tuning required.
  1. Extract the sentence(s) surrounding each citation mention in the text.
  2. Embed that context with InLegalBERT (mean-pooled token embeddings).
  3. Compare against pre-computed prototype embeddings for each relation type.
  4. Assign the closest relation type if similarity >= MIN_CONFIDENCE, else "CITES".

Relation types produced:
  OVERRULES         — "the earlier decision is overruled / no longer good law"
  DISTINGUISHED_FROM — "the present case is distinguishable on facts"
  APPLIED           — "the ratio of the earlier decision is followed/applied"
  CITES             — neutral reference (default fallback)
"""

import re
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Citation patterns (mirrors graph_builder.CITATION_PATTERNS)
# ---------------------------------------------------------------------------
_CITATION_PATTERNS = [
    r'\(\d{4}\)\s+\d+\s+SCC\s+\d+',       # (2017) 10 SCC 1
    r'AIR\s+\d{4}\s+SC\s+\d+',              # AIR 1987 SC 1086
    r'\d{4}\s+SCC\s+OnLine\s+SC\s+\d+',     # 2024 SCC OnLine SC 123
    r'\[\d{4}\]\s+\d+\s+SCR\s+\d+',         # [2017] 1 SCR 1
    r'ILR\s+\d{4}\s+\w+\s+\d+',             # ILR 2020 Kar 567
]
_COMBINED_PATTERN = re.compile("|".join(_CITATION_PATTERNS))

# ---------------------------------------------------------------------------
# Prototype sentences — one cluster per relation type
# Legal phrasings drawn from Indian SC judgment style
# ---------------------------------------------------------------------------
_PROTOTYPES: dict[str, list[str]] = {
    "OVERRULES": [
        "This court overrules the earlier decision and holds it to be no longer good law.",
        "The previous judgment is expressly overruled and cannot be relied upon.",
        "The earlier ruling is overruled by this larger bench.",
        "We overrule the earlier decision to the extent it conflicts with this view.",
    ],
    "DISTINGUISHED_FROM": [
        "The present case is clearly distinguishable from the earlier decision on facts.",
        "The ratio of the earlier case is not applicable as the facts are materially different.",
        "The earlier judgment is distinguished and not followed in the present case.",
        "This case stands on a different footing and must be distinguished.",
    ],
    "APPLIED": [
        "Applying the ratio of the earlier decision, this court follows the principles laid down.",
        "The court followed and applied the ratio of the earlier judgment.",
        "The principles enunciated in the earlier case are directly applicable and we apply them.",
        "Relying on the ratio in the earlier case, the court reached the same conclusion.",
    ],
    "CITES": [
        "The court referred to and cited the earlier decision in the course of its reasoning.",
        "As held in the earlier case, the court noted the proposition of law.",
        "The court took note of the earlier ruling while considering the matter.",
        "Reference was made to the earlier decision by the learned counsel.",
    ],
}

# Minimum cosine similarity to accept a classified relation (else falls back to CITES)
MIN_CONFIDENCE: float = 0.55


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ExtractedRelation:
    citation: str          # Raw citation string extracted from text
    context: str           # Sentence window around the citation
    relation_type: str     # OVERRULES | DISTINGUISHED_FROM | APPLIED | CITES
    confidence: float      # Cosine similarity score (0–1)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
class LegalNERExtractor:
    """InLegalBERT-based relation extractor. Thread-safe for read-only inference."""

    MODEL_NAME = "law-ai/InLegalBERT"
    # Window (in chars) around citation for context extraction fallback
    CONTEXT_WINDOW = 300

    def __init__(self):
        logger.info("loading_inlegalbert", model=self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        self._prototype_embeddings: dict[str, np.ndarray] = self._build_prototypes()
        logger.info("inlegalbert_loaded")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_relations(self, text: str) -> list[ExtractedRelation]:
        """Return one ExtractedRelation per unique citation found in text."""
        citations = list({m.group() for m in _COMBINED_PATTERN.finditer(text)})
        if not citations:
            return []

        relations: list[ExtractedRelation] = []
        for citation in citations:
            context = self._extract_context(text, citation)
            if context:
                rel_type, confidence = self._classify(context)
            else:
                rel_type, confidence = "CITES", 0.5

            relations.append(ExtractedRelation(
                citation=citation,
                context=context,
                relation_type=rel_type,
                confidence=confidence,
            ))
            logger.debug(
                "relation_classified",
                citation=citation[:40],
                type=rel_type,
                confidence=round(confidence, 3),
            )

        return relations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pool InLegalBERT last hidden state, masked for padding tokens."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self.model(**enc)
        # (batch, seq, hidden) — mask padding tokens before mean
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (batch, seq, 1)
        embeddings = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return embeddings.cpu().numpy()

    def _build_prototypes(self) -> dict[str, np.ndarray]:
        """Pre-compute L2-normalised mean embedding for each relation type."""
        result: dict[str, np.ndarray] = {}
        for rel_type, sentences in _PROTOTYPES.items():
            embs = self._embed(sentences)          # (n_sentences, hidden)
            mean_emb = embs.mean(axis=0)            # (hidden,)
            result[rel_type] = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
        return result

    def _classify(self, context: str) -> tuple[str, float]:
        """Return (relation_type, confidence) for a context sentence."""
        emb = self._embed([context])[0]
        emb_norm = emb / (np.linalg.norm(emb) + 1e-9)

        best_type, best_score = "CITES", 0.0
        for rel_type, proto in self._prototype_embeddings.items():
            score = float(np.dot(emb_norm, proto))
            if score > best_score:
                best_score = score
                best_type = rel_type

        # If the top-scoring relation is too weak, fall back to neutral CITES
        if best_score < MIN_CONFIDENCE:
            return "CITES", best_score
        return best_type, best_score

    def _extract_context(self, text: str, citation: str) -> str:
        """Return up to 3 sentences surrounding the citation mention."""
        # Try sentence-based extraction first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for i, sent in enumerate(sentences):
            if citation in sent:
                window = sentences[max(0, i - 1): i + 2]
                return " ".join(window).strip()

        # Fallback: character window around first occurrence
        idx = text.find(citation)
        if idx == -1:
            return ""
        start = max(0, idx - self.CONTEXT_WINDOW)
        end = min(len(text), idx + len(citation) + self.CONTEXT_WINDOW)
        return text[start:end].strip()

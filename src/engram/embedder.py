"""Sentence-transformers wrapper with singleton model loading."""

from __future__ import annotations

from typing import List

_model = None
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_DIMENSIONS = 384


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Encode a list of texts into embedding vectors."""
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [vec.tolist() for vec in embeddings]


def embed_text(text: str) -> List[float]:
    """Encode a single text into an embedding vector."""
    return embed_texts([text])[0]

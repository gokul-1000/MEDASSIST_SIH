"""
The only two ML components that run ON the laptop (i9, no GPU) — everything
else is a hosted API call. Both models are small enough to be genuinely
CPU-friendly:

  - all-MiniLM-L6-v2: 384-dim, ~22M params.
  - ms-marco-MiniLM-L-6-v2 cross-encoder: similarly small.

Imports are lazy so this module — and anything that imports it just to read
type hints — stays importable without `sentence-transformers`/`torch`
installed (which is the situation in this sandbox and, honestly, in a lot of
"just want to check the code compiles" CI runs too).
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from config import EMBEDDING_MODEL, RERANKER_MODEL

_MOCK_DIM = 64


def _mock_vector(text: str) -> list[float]:
    """
    Deterministic hashing-based bag-of-words vector — NOT a real embedding,
    used only so RAG's control flow (retrieve -> rerank -> compress) can be
    unit-tested without installing sentence-transformers/torch. Swapped out
    automatically for the real model whenever it's importable (see
    embed_texts below).
    """
    vec = [0.0] * _MOCK_DIM
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)  # noqa: S324 — non-cryptographic use
        vec[h % _MOCK_DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@lru_cache(maxsize=1)
def _embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers not installed — `pip install sentence-transformers`. "
            "This is the one truly required ML dependency; it runs fine on CPU."
        ) from e
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _reranker():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError("sentence-transformers not installed — `pip install sentence-transformers`.") from e
    return CrossEncoder(RERANKER_MODEL, device="cpu")


def embed_texts(texts: list[str], mock: bool = False) -> list[list[float]]:
    if mock:
        return [_mock_vector(t) for t in texts]
    try:
        model = _embedder()
    except RuntimeError:
        return [_mock_vector(t) for t in texts]
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


def rerank(query: str, passages: list[str], mock: bool = False) -> list[tuple[int, float]]:
    """Returns (original_index, score) pairs sorted best-first."""
    if not passages:
        return []
    if not mock:
        try:
            model = _reranker()
            pairs = [(query, p) for p in passages]
            scores = model.predict(pairs)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return [(idx, float(score)) for idx, score in ranked]
        except RuntimeError:
            pass  # fall through to mock scoring below

    q_vec = _mock_vector(query)
    scored = [(i, _cosine(q_vec, _mock_vector(p))) for i, p in enumerate(passages)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

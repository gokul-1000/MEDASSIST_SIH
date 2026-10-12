"""
Multi-stage RAG (architecture doc §3.7): Retrieve -> Rerank -> Compress.
Single-stage retrieve-and-stuff tends to inject noisy context that reduces
classification accuracy — the rerank stage exists specifically to fix that.

Also implements:
  - Conditional RAG Decision (§3.6): skip for common/high-confidence,
    invoke for rare/ambiguous/low-confidence/guideline-dependent cases.
  - Async RAG prefetch (§3.4): kicked off the moment a follow-up question is
    emitted, using whatever symptoms are already known, so retrieval latency
    overlaps the user's think-time instead of being paid for serially.

Corpus note: for this prototype the "knowledge base" is the versioned KB
JSON files (kb_loader.py) turned into short retrievable passages. That is
intentionally minimal — see README "Honest Rating" for what a real
deployment would need instead (a licensed, continuously-updated clinical
corpus, not five hand-written JSON entries per domain).
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

from knowledge_base.kb_loader import KnowledgeBase, load_kb
from models.embeddings_reranker import embed_texts, rerank

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-prefetch")


@dataclass(frozen=True)
class RetrievedPassage:
    text: str
    source_disease: str
    kb_metadata: dict[str, Any]


@dataclass(frozen=True)
class RagResult:
    context: str
    sources: list[dict[str, Any]]
    similarity: float  # top rerank score, fed into Confidence Calibration


def _build_corpus(kb: KnowledgeBase) -> list[RetrievedPassage]:
    passages = []
    meta = {
        "guideline_version": kb.metadata.guideline_version,
        "publication_date": kb.metadata.publication_date,
        "retrieval_timestamp": kb.metadata.retrieval_timestamp,
        "document_source": kb.metadata.document_source,
    }
    for d in kb.diseases:
        required = ", ".join(s.replace("_", " ") for s in d.get("required_symptoms", []))
        supporting = ", ".join(s.replace("_", " ") for s in d.get("supporting_symptoms", []))
        text = f"{d['name'].replace('_', ' ')}: typically presents with {required}"
        if supporting:
            text += f", often alongside {supporting}"
        if d.get("red_flag"):
            text += f". Red flag: {d.get('red_flag_reason', 'requires urgent evaluation')}"
        passages.append(RetrievedPassage(text=text, source_disease=d["name"], kb_metadata=meta))
    return passages


def should_invoke_rag(
    leading_candidate_score: float,
    confidence_threshold: float,
    candidate_is_rare: bool,
    candidate_is_red_flag: bool,
) -> bool:
    """
    Conditional RAG Decision (§3.6). Skip for common/high-confidence cases,
    invoke for rare/ambiguous/low-confidence/guideline-dependent ones.
    """
    if candidate_is_red_flag:
        return True  # always ground red-flag/emergency calls in the guideline text
    if candidate_is_rare:
        return True
    if leading_candidate_score < confidence_threshold:
        return True
    return False


def retrieve_rerank_compress(domain: str, query: str, top_k_retrieve: int = 15, top_k_final: int = 3, mock: bool = False) -> RagResult:
    """The full three-stage pipeline, run synchronously (used both for the
    normal Conditional-RAG path and inside the async prefetch worker)."""
    kb = load_kb(domain)
    corpus = _build_corpus(kb)
    if not corpus:
        return RagResult(context="", sources=[], similarity=0.0)

    # Stage 1: Retrieve (broad semantic search, top-k ~15-20)
    query_vec = embed_texts([query], mock=mock)[0]
    corpus_vecs = embed_texts([p.text for p in corpus], mock=mock)
    sims = [sum(a * b for a, b in zip(query_vec, cv)) for cv in corpus_vecs]
    ranked_idx = sorted(range(len(corpus)), key=lambda i: sims[i], reverse=True)[:top_k_retrieve]
    retrieved = [corpus[i] for i in ranked_idx]

    # Stage 2: Rerank (small cross-encoder reorders by actual relevance)
    rerank_scores = rerank(query, [p.text for p in retrieved], mock=mock)
    top = rerank_scores[:top_k_final]

    # Stage 3: Compress (extractive — no LLM call, just keep the best passages)
    selected = [retrieved[idx] for idx, _score in top]
    context = " ".join(p.text for p in selected)
    sources = [{"disease": p.source_disease, **p.kb_metadata} for p in selected]
    top_similarity = top[0][1] if top else 0.0

    return RagResult(context=context, sources=sources, similarity=round(float(top_similarity), 4))


class AsyncRagPrefetcher:
    """
    Wraps the async prefetch behavior from §3.4: kick off retrieval the
    moment a follow-up question is emitted, reuse the result if the leading
    candidate hasn't changed by the time the answer comes back, discard and
    re-retrieve if it has.
    """

    def __init__(self):
        self._pending: dict[str, tuple[str, Future]] = {}  # session_id -> (candidate_name, future)

    def kick_off(self, session_id: str, domain: str, leading_candidate: str, mock: bool = False) -> None:
        future = _executor.submit(retrieve_rerank_compress, domain, leading_candidate.replace("_", " "), mock=mock)
        self._pending[session_id] = (leading_candidate, future)

    def resolve(self, session_id: str, current_leading_candidate: str, timeout_s: float = 2.0) -> RagResult | None:
        """Returns the prefetched result if it still matches the current
        leading candidate, else None (caller should do a fresh synchronous
        retrieval).

        Catches both concurrent.futures.TimeoutError (Python ≤ 3.10) and the
        built-in TimeoutError (Python 3.11+, where the two were unified).
        Also catches any other exception from the background worker so a
        failed prefetch never crashes the active graph session.
        """
        pending = self._pending.pop(session_id, None)
        if pending is None:
            return None
        prefetched_candidate, future = pending
        if prefetched_candidate != current_leading_candidate:
            future.cancel()
            return None
        try:
            return future.result(timeout=timeout_s)
        except (FuturesTimeoutError, TimeoutError):
            # Prefetch didn't finish in time — caller will do a fresh retrieval.
            future.cancel()
            return None
        except Exception:
            # Any other worker failure (embedding error, etc.) — degrade
            # gracefully rather than propagating into the graph runner.
            return None


# Module-level singleton — one prefetcher per process is fine for this
# prototype's scale; a multi-worker deployment would key this off a shared
# cache (e.g. Redis) instead.
prefetcher = AsyncRagPrefetcher()

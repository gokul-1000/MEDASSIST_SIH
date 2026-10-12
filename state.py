"""
Graph state schema — single source of truth for what flows through every node.

Design note (matches architecture doc §2): conversation memory is represented
entirely through this typed state, not a raw chat transcript. Nodes read
`extracted_symptoms`, `age`/`gender`, `follow_up_history`, `confidence_score`,
and the rest of these fields — never a message log.

This is a plain TypedDict (not pydantic) on purpose: LangGraph's StateGraph
works natively with TypedDict + Annotated reducers, and keeping this module
free of third-party imports means state.py, and everything that only reads
type hints from it, can be unit-tested without installing langgraph/torch.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict


class QAPair(TypedDict):
    question: str
    answer: str
    asked_for: str  # the missing_info key this question targeted


class CandidateDisease(TypedDict):
    name: str
    score: float  # 0-1, running plausibility given extracted_symptoms


class SafetyFlag(TypedDict):
    code: str
    detail: str
    severity: Literal["info", "warning", "critical"]


class State(TypedDict, total=False):
    # --- identity / thread ---
    session_id: str
    raw_query: str

    # --- routing ---
    domain: Literal["eye", "general"]
    candidate_diseases: list[CandidateDisease]  # ranked, probabilistic router output

    # --- symptom pipeline ---
    raw_symptom_text: Annotated[list[str], operator.add]
    extracted_symptoms: dict[str, Any]
    age: Optional[int]
    gender: Optional[str]
    missing_info: list[str]  # Planner output, ranked by expected information gain

    # --- follow-up loop control ---
    follow_up_count: int
    follow_up_history: Annotated[list[QAPair], operator.add]
    max_follow_ups: int  # config, default 3
    pending_question: str  # written by follow_up_generate, consumed by follow_up_interrupt
    pending_asked_for: str  # the missing_info key the pending question targets

    # --- confidence (two distinct signals — see confidence.py) ---
    confidence_score: float  # in-loop, deterministic, "keep asking or stop?"
    confidence_history: Annotated[list[float], operator.add]  # confidence_score at each turn, feeds Calibration
    calibrated_confidence: Optional[float]  # post-diagnosis, "trust this answer?"
    confidence_threshold: float  # domain-specific

    # --- RAG ---
    rag_required: bool
    rag_prefetch_context: Optional[str]
    rag_prefetch_candidate: Optional[str]  # which disease the prefetch was keyed to
    rag_context: Optional[str]
    rag_sources: list[dict[str, Any]]  # versioned KB metadata for whatever was retrieved
    rag_similarity: Optional[float]  # top rerank score, feeds Confidence Calibration

    # --- diagnosis ---
    predicted_disease: Optional[str]
    raw_model_confidence: Optional[float]
    differential_diagnosis: Optional[list[dict[str, Any]]]  # ranked differential: top-N DifferentialEntry dicts
    kb_constrained: bool  # True = all predictions are from KB; False = LLM hallucination detected and corrected

    # --- safety / escalation ---
    safety_flags: Annotated[list[SafetyFlag], operator.add]
    escalated: bool
    escalation_reason: Optional[str]

    # --- rule-engine outputs ---
    emergency_level: Optional[Literal[1, 2, 3]]
    recommended_department: Optional[str]

    # --- telemetry ---
    model_tier_used: list[str]  # accumulates which tier served each LLM call this turn
    fallback_applied: bool
    node_latencies_ms: dict[str, float]

    # --- terminal output ---
    final_response: Optional[str]

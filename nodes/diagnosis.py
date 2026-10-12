"""
Conditional RAG Decision (§3.6), Direct Reasoning (§3.8), and
Diagnosis/Classification (§3.9) — grouped in one file because they form a
single linear decision ("do we need grounding, then commit to an answer"),
but still exposed as three separately callable functions so graph.py can
show them as distinct nodes on the mermaid diagram, matching the doc.

Improvements in this version:

1. KB CONSTRAINT ENFORCEMENT — After the LLM responds, the predicted disease
   name is validated against the candidate_diseases list (which only contains
   KB entries). If the LLM returns a name that is not in the KB, the system
   falls back to the top-ranked KB candidate and sets `kb_constrained=False`
   as an audit flag. The final diagnosis is always a KB entry.

2. DIFFERENTIAL DIAGNOSIS — After committing to a top disease, the node calls
   `build_differential()` from confidence.py to construct a deterministic
   top-3 ranked list with matched/missing/contradictory symptoms and reasoning.
   No extra LLM call required — the differential is built from symptom math.

3. NO-MATCH HANDLING — If the LLM explicitly returns "no_match" or the
   candidate list is empty, the predicted_disease is set to the sentinel
   string "no_match_in_kb" and confidence is 0. The synthesis node
   translates this into the user-facing message:
   "No reliable match was found within the current knowledge base."
"""
from __future__ import annotations

import json
from typing import Any

from config import Settings
from confidence import build_differential
from knowledge_base.kb_loader import load_kb
from models.model_router import ModelRouter
from rag.rag_pipeline import prefetcher, retrieve_rerank_compress, should_invoke_rag
from state import State

_RARE_DISEASE_BASE_RATE_CUTOFF = 0.05

# Sentinel returned by LLM (or injected by us) when no KB disease matches
_NO_MATCH_SENTINEL = "no_match_in_kb"

_DIAGNOSIS_SYSTEM_TEMPLATE = """You are a differential-diagnosis assistant for a self-assessment tool. You do
NOT provide treatment, dosages, or medication advice — only which condition
best fits the symptom pattern, from the given candidate list only.
task=diagnosis
"leading_candidate": "{leading_candidate}"
Candidate diseases (name: score): {candidates}
Extracted symptoms: {symptoms}
{rag_block}
CRITICAL RULES:
- You MUST only select a disease from the candidate list above. Do NOT invent new disease names.
- If none of the candidates fit reasonably, return exactly: {{"predicted_disease": "no_match", "confidence": 0.0}}
- Do not rely on general medical knowledge to name conditions not in the list.
- Self-report your confidence honestly as a 0-1 float. Do not default to a high number.
- Confidence above 0.90 should only be used for extremely clear-cut cases.

Respond with strict JSON: {{"predicted_disease": "...", "confidence": 0.0}}"""


def conditional_rag_decision_node(state: State) -> dict[str, Any]:
    candidates = state.get("candidate_diseases", [])
    if not candidates:
        return {"rag_required": True}

    domain = state["domain"]
    kb = load_kb(domain)
    leading = candidates[0]
    entry = kb.disease_by_name(leading["name"])
    is_rare = bool(entry and entry.get("base_rate", 1.0) < _RARE_DISEASE_BASE_RATE_CUTOFF)
    is_red_flag = bool(entry and entry.get("red_flag"))

    required = should_invoke_rag(
        leading_candidate_score=leading["score"],
        confidence_threshold=state.get("confidence_threshold", 0.60),
        candidate_is_rare=is_rare,
        candidate_is_red_flag=is_red_flag,
    )
    return {"rag_required": required}


def make_rag_retrieval_node(settings: Settings):
    def rag_retrieval_node(state: State) -> dict[str, Any]:
        candidates = state.get("candidate_diseases", [])
        leading_name = candidates[0]["name"] if candidates else state["domain"]

        prefetched = prefetcher.resolve(
            session_id=state.get("session_id", "default"),
            current_leading_candidate=leading_name,
        )
        if prefetched is not None:
            result = prefetched  # reuse — the leading candidate didn't change since the follow-up was asked
        else:
            query = " ".join(k for k, v in state.get("extracted_symptoms", {}).items() if v is True) or leading_name
            try:
                result = retrieve_rerank_compress(state["domain"], query, mock=settings.use_mock_provider)
            except RuntimeError:
                # Retrieval fallback (§5): degrade gracefully to Direct Reasoning
                # rather than hard-failing the session.
                return {"rag_context": None, "rag_sources": [], "rag_required": False, "fallback_applied": True}

        return {"rag_context": result.context, "rag_sources": result.sources, "rag_similarity": result.similarity}

    return rag_retrieval_node


def direct_reasoning_node(state: State) -> dict[str, Any]:
    """Marker node used when rag_required == False — the actual diagnosis
    call is shared with the RAG path in diagnosis_node below; this node just
    makes the "skip RAG" branch visible on the graph, per the mermaid
    diagram in the architecture doc."""
    return {}


def _validate_kb_prediction(
    predicted: str,
    candidate_names: set[str],
    fallback: str,
) -> tuple[str, bool]:
    """
    Validate that the LLM's predicted disease is in the KB candidate set.

    Returns (validated_name, kb_constrained) where:
    - kb_constrained=True  means the prediction came from the KB (normal)
    - kb_constrained=False means the LLM hallucinated a name and we fell back
    """
    # Explicit no-match from LLM or empty prediction
    if predicted in ("no_match", "", _NO_MATCH_SENTINEL):
        return _NO_MATCH_SENTINEL, True  # legitimate "no match" — not a hallucination

    # Predicted name is in the KB — all good
    if predicted in candidate_names:
        return predicted, True

    # LLM hallucinated a name not in the KB — fall back to top KB candidate
    return fallback, False


def make_diagnosis_node(model_router: ModelRouter):
    def diagnosis_node(state: State) -> dict[str, Any]:
        domain = state["domain"]
        kb = load_kb(domain)
        candidates = state.get("candidate_diseases", [])
        extracted = state.get("extracted_symptoms", {})

        # Safety: if no candidates, still produce a minimal response
        leading = candidates[0]["name"] if candidates else "unspecified_condition"
        candidate_names = {c["name"] for c in candidates}

        rag_context = state.get("rag_context")
        rag_block = f"Guideline context: {rag_context}" if rag_context else ""

        system_prompt = _DIAGNOSIS_SYSTEM_TEMPLATE.format(
            leading_candidate=leading,
            candidates=json.dumps(candidates),
            symptoms=json.dumps(extracted),
            rag_block=rag_block,
        )

        tier = "large" if state.get("rag_required") else "medium"
        routed = model_router.call(
            tier=tier,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Provide the diagnosis."},
            ],
            escalate_if_low_confidence=0.5,
        )
        parsed = routed.response.parsed or {}
        raw_predicted = parsed.get("predicted_disease", leading)
        raw_confidence = float(parsed.get("confidence", 0.5))

        # --- KB Constraint Enforcement ---
        # Validate the LLM's choice is in the KB. If not, fall back to the
        # top-ranked KB candidate and flag kb_constrained=False for auditing.
        fallback_name = candidates[0]["name"] if candidates else _NO_MATCH_SENTINEL
        predicted, kb_constrained = _validate_kb_prediction(
            raw_predicted, candidate_names, fallback_name
        )

        # If no match in KB, zero out confidence
        if predicted == _NO_MATCH_SENTINEL:
            raw_confidence = 0.0

        # --- Differential Diagnosis Construction ---
        # Build the top-3 ranked differential using only KB diseases.
        # This is deterministic (no LLM) and guaranteed to be KB-constrained.
        differential_entries = build_differential(
            extracted_symptoms=extracted,
            diseases=kb.diseases,
            top_n=3,
        )
        differential_dicts = [e.to_dict() for e in differential_entries]

        model_tier_used = state.get("model_tier_used", []) + [routed.tier_served_by]

        return {
            "predicted_disease": predicted,
            "raw_model_confidence": raw_confidence,
            "differential_diagnosis": differential_dicts,
            "kb_constrained": kb_constrained,
            "model_tier_used": model_tier_used,
            "fallback_applied": state.get("fallback_applied", False) or routed.fallback_applied,
        }

    return diagnosis_node

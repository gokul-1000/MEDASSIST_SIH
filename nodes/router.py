"""
Router (split out from the old monolithic "Supervisor" per your first
refinement). Two responsibilities, both cheap:

1. Domain classification (eye vs. general) — small model, one call.
2. Probabilistic disease candidate ranking — your second refinement.
   Instead of handing the Planner nothing but a domain label, we seed
   `candidate_diseases` immediately using the deterministic scorer in
   confidence.py against every disease in that domain's KB. This is what
   lets the Planner ask "which of my top 2 candidates does this question
   separate?" from turn one, instead of waiting for a full loop to build a
   candidate list from scratch.

Loop-control, retries, and checkpoint coordination are deliberately NOT
here — that's the Orchestrator (nodes/orchestrator.py). Router only routes.
"""
from __future__ import annotations

import logging
from typing import Any

from confidence import rank_candidates
from config import CONFIDENCE_THRESHOLDS, MAX_FOLLOW_UPS
from knowledge_base.kb_loader import load_kb
from models.model_router import ModelRouter
from state import State

logger = logging.getLogger("medassist.nodes.router")

_DOMAIN_SYSTEM_PROMPT = """You are a routing classifier for a medical symptom assessment tool.
task=classify_domain
Given the user's raw query, decide whether it is primarily an EYE complaint
(vision, redness, eye pain, discharge, etc.) or a GENERAL complaint (anything
else — fever, headache, chest pain, digestive issues, etc.).
Respond with strict JSON: {"domain": "eye"} or {"domain": "general"}."""

_EYE_KEYWORDS = ("eye", "vision", "sight", "blurry", "blurred", "pupil", "cornea", "eyelid")


def make_router_node(model_router: ModelRouter):
    def router_node(state: State) -> dict[str, Any]:
        raw_query = state.get("raw_query", "")

        domain = _classify_domain_heuristic(raw_query)
        if domain is None:
            # Ambiguous on keywords alone — ask the small model to decide.
            resp = model_router.call(
                tier="small",
                messages=[
                    {"role": "system", "content": _DOMAIN_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_query},
                ],
            )
            domain = (resp.response.parsed or {}).get("domain", "general")
            if domain not in ("eye", "general"):
                domain = "general"

        kb = load_kb(domain)
        # Seed candidates from whatever's in extracted_symptoms so far (usually
        # nothing yet on the very first call — that's fine, rank_candidates
        # falls back to each disease's base_rate).
        candidates = rank_candidates(state.get("extracted_symptoms", {}), kb.diseases)

        return {
            "domain": domain,
            "candidate_diseases": candidates,
            "confidence_threshold": CONFIDENCE_THRESHOLDS[domain],
            "max_follow_ups": state.get("max_follow_ups", MAX_FOLLOW_UPS),
            "follow_up_count": state.get("follow_up_count", 0),
            "escalated": False,
            "safety_flags": [],
            "raw_symptom_text": [],
        }

    return router_node


def _classify_domain_heuristic(raw_query: str) -> str | None:
    """Cheap keyword pre-filter so most eye complaints never need an LLM call
    at all. Returns None when it's genuinely ambiguous."""
    text = raw_query.lower()
    if any(k in text for k in _EYE_KEYWORDS):
        return "eye"
    # Only fall through to the model for queries with no clear signal either way.
    general_signal_words = ("fever", "chest", "stomach", "head", "throat", "cough", "pain")
    if any(k in text for k in general_signal_words):
        return "general"
    return None

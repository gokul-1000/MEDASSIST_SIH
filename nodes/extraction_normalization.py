"""
Two-step, not one (§3.3):

1. Extractor  — small model pulls raw symptom phrases from raw_query/latest
   answer into raw_symptom_text.
2. Normalizer — maps free-text phrases onto the domain's fixed symptom
   vocabulary and writes booleans into extracted_symptoms.

Kept as two separate node functions (and two separate LangGraph nodes) on
purpose: the Confidence Engine, Planner, and rule engines all key off the
*normalized* vocabulary. If extraction and normalization were fused, every
downstream node would need to handle arbitrary free text.
"""
from __future__ import annotations

import json
from typing import Any

from knowledge_base.kb_loader import load_kb
from models.model_router import ModelRouter
from nodes.symptom_lexicon import follow_up_target_mapping, symptom_phrases_in_text, symptoms_in_text
from state import State

_EXTRACT_SYSTEM_TEMPLATE = """You are a symptom-phrase extractor for a medical assessment tool.
task=extract_symptoms
Pull out every symptom-related phrase the user mentioned, verbatim or lightly
cleaned up (e.g. "eye hurts a lot" stays close to the original wording — do
NOT diagnose or normalize yet, just extract phrases).
Respond with strict JSON: {{"raw_symptom_phrases": ["phrase1", "phrase2", ...]}}"""

_NORMALIZE_SYSTEM_TEMPLATE = """You are a symptom normalizer for a medical assessment tool.
task=normalize_symptoms
The fixed symptom vocabulary for this domain is: {vocabulary}
Given the raw phrases below, map each one to zero or more vocabulary terms
and output booleans. If a phrase clearly DENIES a symptom (e.g. "no itching"),
map it to false. If a phrase doesn't match any vocabulary term, ignore it.
"raw_symptom_phrases": {phrases}
Respond with strict JSON mapping vocabulary terms to true/false, e.g.
{{"eye_redness": true, "itching": false}}"""


def make_extractor_node(model_router: ModelRouter):
    def extractor_node(state: State) -> dict[str, Any]:
        # The text to extract from is either the very first raw_query, or —
        # once the loop is underway — the latest follow-up answer.
        history = state.get("follow_up_history", [])
        text = history[-1]["answer"] if history else state.get("raw_query", "")

        resp = model_router.call(
            tier="small",
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM_TEMPLATE},
                {"role": "user", "content": text},
            ],
        )
        phrases = (resp.response.parsed or {}).get("raw_symptom_phrases", [])
        domain = state.get("domain")
        if domain in ("eye", "general"):
            kb = load_kb(domain)
            phrases = list(dict.fromkeys([*phrases, *symptom_phrases_in_text(text, set(kb.symptom_vocabulary))]))
        return {"raw_symptom_text": phrases}

    return extractor_node


def make_normalizer_node(model_router: ModelRouter):
    def normalizer_node(state: State) -> dict[str, Any]:
        domain = state["domain"]
        kb = load_kb(domain)
        phrases = state.get("raw_symptom_text", [])

        system_prompt = _NORMALIZE_SYSTEM_TEMPLATE.format(
            vocabulary=", ".join(kb.symptom_vocabulary),
            phrases=json.dumps(phrases),
        )
        resp = model_router.call(
            tier="small",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Normalize the phrases above."},
            ],
        )
        normalized_raw = resp.response.parsed or {}
        # Defensive filter: only keep keys that are actually in this
        # domain's vocabulary, and only real booleans — a model occasionally
        # hallucinates a field name or returns a string instead of a bool.
        normalized = {
            k: bool(v)
            for k, v in normalized_raw.items()
            if k in kb.symptom_vocabulary and isinstance(v, bool)
        }
        vocabulary = set(kb.symptom_vocabulary)
        for phrase in phrases:
            normalized.update(symptoms_in_text(str(phrase), vocabulary))

        history = state.get("follow_up_history", [])
        if history:
            latest = history[-1]
            normalized.update(
                follow_up_target_mapping(
                    latest.get("asked_for"),
                    latest.get("answer", ""),
                    vocabulary,
                )
            )

        merged = {**state.get("extracted_symptoms", {}), **normalized}
        return {"extracted_symptoms": merged}

    return normalizer_node

"""
Safety Validator (§3.11). Sits before Emergency Classification, not folded
into it, so the emergency rule engine only ever sees validated,
contradiction-checked state.

Two independent checks, both run every time:
  1. Deterministic red-flag combination check (rules/rule_engine.py) — no
     LLM, can't be talked out of firing.
  2. A moderation-tier model pass (openai/gpt-oss-safeguard-20b — a
     purpose-built policy-following moderation model, not the main LLM)
     that looks for contradictions, hallucinated details, and any drift
     toward unsafe advice (dosages, treatment instructions — this system
     must never emit those).

A non-empty red-flag list, or a "critical" flag from either check, can force
emergency_level = 3 downstream, overriding whatever the rule engine would
have computed from the diagnosis alone.
"""
from __future__ import annotations

import json
from typing import Any

from models.model_router import ModelRouter
from rules.rule_engine import check_red_flags
from state import State

_VALIDATION_SYSTEM_PROMPT = """You are a safety validator for a medical self-assessment tool's output. You do
NOT generate new medical content — you only check what's already been produced.
task=safety_validation
Predicted disease: {predicted_disease}
Extracted symptoms: {symptoms}
Check for:
1. Internal contradictions (symptom set inconsistent with the predicted disease)
2. Signs the model hallucinated a detail not present in the symptoms above
3. Any drift toward unsafe advice (dosages, treatment/medication instructions)
   — this system must never emit those, so flag if the diagnosis text implies any.
Respond with strict JSON:
{{"contradictions": [...], "hallucination_risk": "low"|"medium"|"high", "unsafe_advice_detected": true|false}}"""


def make_safety_validator_node(model_router: ModelRouter):
    def safety_validator_node(state: State) -> dict[str, Any]:
        extracted = state.get("extracted_symptoms", {})
        predicted = state.get("predicted_disease")

        flags: list[dict[str, str]] = []

        red_flags = check_red_flags(extracted)
        if red_flags.triggered:
            for combo_id, reason in zip(red_flags.combination_ids, red_flags.reasons):
                flags.append({"code": f"red_flag:{combo_id}", "detail": reason, "severity": "critical"})

        system_prompt = _VALIDATION_SYSTEM_PROMPT.format(
            predicted_disease=predicted, symptoms=json.dumps(extracted)
        )
        resp = model_router.call(
            tier="moderation",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Validate the diagnosis above."},
            ],
        )
        parsed = resp.response.parsed or {}

        for contradiction in parsed.get("contradictions", []):
            flags.append({"code": "contradiction", "detail": str(contradiction), "severity": "warning"})

        if parsed.get("hallucination_risk") == "high":
            flags.append(
                {"code": "hallucination_risk", "detail": "model flagged high hallucination risk", "severity": "critical"}
            )
        elif parsed.get("hallucination_risk") == "medium":
            flags.append(
                {"code": "hallucination_risk", "detail": "model flagged medium hallucination risk", "severity": "warning"}
            )

        if parsed.get("unsafe_advice_detected"):
            flags.append(
                {"code": "unsafe_advice", "detail": "possible dosage/treatment instruction detected", "severity": "critical"}
            )

        return {"safety_flags": flags}

    return safety_validator_node

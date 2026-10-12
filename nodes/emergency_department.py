"""
Emergency Classification (§3.13) and Department Recommendation (§3.14).
Both are rule-based lookups with no LLM call, and both are mutually
independent of Confidence Calibration once Diagnosis + Safety Validation
have produced a finalized predicted_disease and safety_flags — so graph.py
fans all three out in parallel and joins them at Response Synthesis (§4).
"""
from __future__ import annotations

from typing import Any

from knowledge_base.kb_loader import load_kb
from rules.rule_engine import classify_emergency_level, recommend_department
from state import State


def emergency_classification_node(state: State) -> dict[str, Any]:
    domain = state["domain"]
    kb = load_kb(domain)
    entry = kb.disease_by_name(state.get("predicted_disease", ""))
    has_critical_flag = any(f["severity"] == "critical" for f in state.get("safety_flags", []))

    level = classify_emergency_level(
        predicted_disease=state.get("predicted_disease"),
        extracted_symptoms=state.get("extracted_symptoms", {}),
        kb_disease_entry=entry,
        safety_flags_critical=has_critical_flag,
    )
    return {"emergency_level": level}


def department_recommendation_node(state: State) -> dict[str, Any]:
    domain = state["domain"]
    kb = load_kb(domain)
    entry = kb.disease_by_name(state.get("predicted_disease", ""))
    department = recommend_department(state.get("predicted_disease"), domain, entry)
    return {"recommended_department": department}

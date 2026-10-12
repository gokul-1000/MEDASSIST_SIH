"""
Everything that carries direct clinical-adjacent weight is a lookup table,
not a generation task (architecture doc §10). No LLM call touches this file
— full determinism, full auditability, and fully unit-testable with zero
dependencies.

Three responsibilities, kept in one module because they're small and tightly
coupled by the same red-flag vocabulary, but each is exposed as its own
function so nodes/safety_validator.py and nodes/escalation.py can call them
independently and the graph can still show them as parallel fan-out targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

EmergencyLevel = Literal[1, 2, 3]

# Hard red-flag COMBINATIONS that short-circuit straight to Level 3,
# bypassing the rest of the loop (architecture doc §9). Each entry is a set
# of extracted_symptoms keys that must ALL be true.
RED_FLAG_COMBINATIONS: list[dict[str, Any]] = [
    {
        "id": "chest_pain_breathlessness",
        "requires": {"chest_pain", "shortness_of_breath"},
        "reason": "chest pain + shortness of breath — possible acute coronary syndrome / PE",
    },
    {
        "id": "stroke_pattern",
        "requires": {"facial_droop", "slurred_speech"},
        "reason": "facial droop + slurred speech — stroke-pattern presentation (FAST criteria)",
    },
    {
        "id": "stroke_pattern_deficit",
        "requires": {"neurological_deficit", "sudden_worst_headache_of_life"},
        "reason": "sudden focal neurological deficit + thunderclap headache",
    },
    {
        "id": "acute_angle_closure",
        "requires": {"severe_eye_pain", "halos_around_lights"},
        "reason": "severe eye pain + halos around lights — possible acute angle-closure glaucoma",
    },
    {
        "id": "retinal_detachment_pattern",
        "requires": {"curtain_shadow_vision", "flashes_of_light"},
        "reason": "curtain/shadow vision + flashes of light — possible retinal detachment",
    },
    {
        "id": "anaphylaxis_pattern_rash",
        "requires": {"hives_or_rash", "shortness_of_breath"},
        "reason": "rash/hives with breathing difficulty — possible anaphylaxis, needs immediate emergency care",
    },
    {
        "id": "anaphylaxis_pattern_swelling",
        "requires": {"facial_swelling", "shortness_of_breath"},
        "reason": "facial/airway swelling with breathing difficulty — possible anaphylaxis, needs immediate emergency care",
    },
    {
        "id": "thunderclap_headache",
        "requires": {"sudden_worst_headache_of_life"},
        "reason": ("sudden worst headache of life (thunderclap headache) "
        "requires immediate emergency evaluation"),
    },
    {
        "id": "heart_attack_pattern",
        "requires": {
        "chest_pain",
        "left_arm_pain",
        "sweating"
        },
        "reason": "possible heart attack pattern",
    },
    {
        "id": "appendicitis_complete",
        "requires": {
        "right_lower_quadrant_pain",
        "fever",
        "vomiting"
        },
        "reason": "classic appendicitis pattern",
    },
    {
        "id": "neurological_deficit",
        "requires": {"neurological_deficit"},
        "reason": "new neurological deficit requires immediate emergency evaluation",
    }
]


@dataclass(frozen=True)
class RedFlagResult:
    triggered: bool
    combination_ids: list[str]
    reasons: list[str]


def check_red_flags(extracted_symptoms: dict[str, Any]) -> RedFlagResult:
    """Pure function: symptom booleans in, red-flag verdict out. No LLM."""
    present = {k for k, v in extracted_symptoms.items() if v is True}
    triggered_ids: list[str] = []
    reasons: list[str] = []
    for combo in RED_FLAG_COMBINATIONS:
        if combo["requires"].issubset(present):
            triggered_ids.append(combo["id"])
            reasons.append(combo["reason"])
    return RedFlagResult(triggered=bool(triggered_ids), combination_ids=triggered_ids, reasons=reasons)


def classify_emergency_level(
    predicted_disease: str | None,
    extracted_symptoms: dict[str, Any],
    kb_disease_entry: dict[str, Any] | None,
    safety_flags_critical: bool = False,
) -> EmergencyLevel:
    """
    Deterministic lookup keyed on predicted_disease + red-flag symptoms +
    safety_flags. No LLM call. Outputs exactly Level 1/2/3.

    A non-empty red-flag list, OR a critical Safety Validator flag, forces
    Level 3 directly — overriding whatever the rule engine would have
    computed from the diagnosis alone (safety overrides normal routing,
    per architecture doc §3.11).
    """
    red_flags = check_red_flags(extracted_symptoms)
    if red_flags.triggered or safety_flags_critical:
        return 3
    if kb_disease_entry and kb_disease_entry.get("red_flag"):
        return 3
    if kb_disease_entry and kb_disease_entry.get("department", "").endswith("(Emergency)"):
        return 2
    return 1


def recommend_department(
    predicted_disease: str | None,
    domain: str,
    kb_disease_entry: dict[str, Any] | None,
) -> str:
    """
    Static predicted_disease -> department mapping. Eye subgraph defaults to
    Ophthalmology always. Never names individual doctors.
    """
    if kb_disease_entry:
        return kb_disease_entry["department"]
    # Coarse fallback used by Human Escalation: still rule-based, just
    # keyed on domain instead of a specific (unconfirmed) disease.
    return "Ophthalmology" if domain == "eye" else "General Medicine"

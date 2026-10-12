"""
Budget-aware adaptive question planner.

The planner is intentionally deterministic and model-free. It ranks every
unknown KB symptom as a possible follow-up by expected utility under a strict
three-question budget:

* diagnostic discrimination: expected entropy reduction across all diseases
* contradiction handling: expected candidate mass ruled out by yes/no answers
* safety: expected value of red-flag combinations and red-flag diseases
* coverage: how much current candidate mass the symptom can inform

This avoids the old feedback loop where only already-winning candidates could
generate questions. Unknown required symptoms are treated as unknown, not as
negative evidence; explicit denials still rule candidates out.
"""
from __future__ import annotations

import math
from typing import Any

from confidence import rank_candidates
from knowledge_base.kb_loader import load_kb
from rules.rule_engine import RED_FLAG_COMBINATIONS
from state import State

_EPSILON = 1e-9

_FIELD_LIKELIHOOD = {
    "required": 0.88,
    "supporting": 0.62,
    "contraindicating": 0.06,
    "unrelated": 0.12,
}

_UTILITY_WEIGHTS = {
    "info_gain": 0.34,
    "elimination": 0.22,
    "safety": 0.32,
    "coverage": 0.08,
    "margin_gain": 0.04,
}

_RELATION_STOP_TOKENS = {
    "pain",
    "severe",
    "sudden",
    "worst",
    "life",
    "symptom",
    "sensation",
}


def planner_node(state: State) -> dict[str, Any]:
    domain = state["domain"]
    extracted = state.get("extracted_symptoms", {})
    kb = load_kb(domain)

    candidates = rank_candidates(extracted, kb.diseases)
    known_or_asked = set(extracted.keys()) | _asked_fields(state)
    present = {key for key, value in extracted.items() if value is True}
    candidate_fields = _candidate_question_fields(kb.diseases, present)
    unknown_fields = sorted(field for field in candidate_fields if field not in known_or_asked)

    distribution = _disease_distribution(kb.diseases, extracted)
    current_entropy = _entropy(distribution)
    current_margin = _top_margin(distribution)
    remaining_budget = max(state.get("max_follow_ups", 3) - state.get("follow_up_count", 0), 1)
    confirmation_fields = _confirmation_mode_fields(kb.diseases, extracted, distribution, known_or_asked)
    if confirmation_fields:
        unknown_fields = sorted(confirmation_fields)

    scored_fields = [
        _score_question(
            field=field,
            diseases=kb.diseases,
            extracted=extracted,
            distribution=distribution,
            current_entropy=current_entropy,
            current_margin=current_margin,
            remaining_budget=remaining_budget,
        )
        for field in unknown_fields
    ]
    scored_fields.sort(
        key=lambda item: (
            -item["score"],
            -item["safety"],
            -item["info_gain"],
            -item["elimination"],
            item["field"],
        )
    )

    return {
        "missing_info": [item["field"] for item in scored_fields],
        "candidate_diseases": candidates,
    }


def _asked_fields(state: State) -> set[str]:
    asked = {
        qa.get("asked_for", "")
        for qa in state.get("follow_up_history", [])
        if isinstance(qa, dict) and qa.get("asked_for")
    }
    pending = state.get("pending_asked_for")
    if pending:
        asked.add(pending)
    return asked


def _candidate_question_fields(diseases: list[dict[str, Any]], present: set[str]) -> set[str]:
    fields: set[str] = set()
    for entry in diseases:
        required = set(entry.get("required_symptoms", []))
        supporting = set(entry.get("supporting_symptoms", []))
        contraindicating = set(entry.get("contraindicating_symptoms", []))
        relevant = required | supporting | contraindicating
        overlap = relevant & present
        if present and not (required & present or len(overlap) >= 2):
            continue
        new_fields = relevant - present

        fields |= new_fields
    for combo in RED_FLAG_COMBINATIONS:
        requires = set(combo["requires"])
        if not present or requires & present :
            fields |= requires
    if not fields:
        for entry in diseases:
            fields |= set(entry.get("required_symptoms", []))
    return fields


def _disease_distribution(diseases: list[dict[str, Any]], extracted: dict[str, Any]) -> dict[str, float]:
    weights = {entry["name"]: _soft_evidence_weight(entry, extracted) for entry in diseases}
    total = sum(weights.values())
    if total <= _EPSILON:
        uniform = 1.0 / max(len(diseases), 1)
        return {entry["name"]: uniform for entry in diseases}
    return {name: weight / total for name, weight in weights.items()}


def _soft_evidence_weight(entry: dict[str, Any], extracted: dict[str, Any]) -> float:
    required = set(entry.get("required_symptoms", []))
    supporting = set(entry.get("supporting_symptoms", []))
    contraindicating = set(entry.get("contraindicating_symptoms", []))
    present = {key for key, value in extracted.items() if value is True}
    absent = {key for key, value in extracted.items() if value is False}

    if required & absent:
        return 0.0
    if contraindicating & present:
        return 0.0

    weight = max(float(entry.get("base_rate", 0.05)), 0.001)
    relevant = required | supporting | contraindicating
    if present and not (relevant & present):
        weight *= 0.03

    # Approximate a naive-Bayes likelihood ratio over the fixed KB fields.
    # Priors still matter, but direct evidence should dominate when a disease's
    # defining required pattern is fully present.
    for field in required:
        if field in present:
            weight *= 12.0
    for field in supporting:
        if field in present:
            weight *= 4.0
        elif field in absent:
            weight *= 0.55
    for field in contraindicating:
        if field in absent:
            weight *= 1.4

    if required and required.issubset(present):
        weight *= 8.0 if entry.get("red_flag") else 4.0
    return weight


def _confirmation_mode_fields(
    diseases: list[dict[str, Any]],
    extracted: dict[str, Any],
    distribution: dict[str, float],
    known_or_asked: set[str],
) -> set[str]:
    present = {key for key, value in extracted.items() if value is True}
    if not present:
        return set()

    by_name = {entry["name"]: entry for entry in diseases}
    ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return set()

    leading_entry = by_name[ordered[0][0]]
    leading_required = set(leading_entry.get("required_symptoms", []))
    if not leading_required or not leading_required.issubset(present):
        return set()

    leading_prob = ordered[0][1]
    second_prob = ordered[1][1] if len(ordered) > 1 else 0.0
    if leading_prob < 0.18 and not leading_entry.get("red_flag"):
        return set()
    if leading_prob - second_prob < 0.04 and not leading_entry.get("red_flag"):
        return set()

    fields = set(leading_entry.get("supporting_symptoms", [])) | set(leading_entry.get("contraindicating_symptoms", []))
    for name, probability in ordered[1:5]:
        entry = by_name[name]
        competitor_floor = leading_prob * (0.10 if entry.get("red_flag") else 0.25)
        if probability < max(0.01, competitor_floor):
            continue
        fields |= set(entry.get("required_symptoms", []))
        fields |= set(entry.get("supporting_symptoms", []))
        fields |= set(entry.get("contraindicating_symptoms", []))

    return {field for field in fields if field not in known_or_asked}


def _score_question(
    field: str,
    diseases: list[dict[str, Any]],
    extracted: dict[str, Any],
    distribution: dict[str, float],
    current_entropy: float,
    current_margin: float,
    remaining_budget: int,
) -> dict[str, float | str]:
    p_true = _estimated_true_probability(field, diseases, distribution)
    true_dist = _disease_distribution(diseases, {**extracted, field: True})
    false_dist = _disease_distribution(diseases, {**extracted, field: False})

    expected_entropy = p_true * _entropy(true_dist) + (1.0 - p_true) * _entropy(false_dist)
    max_entropy = math.log2(max(len(diseases), 2))
    info_gain = max(0.0, current_entropy - expected_entropy) / max_entropy

    elimination = _expected_eliminated_mass(field, diseases, distribution, p_true)
    safety = _safety_value(field, diseases, distribution, extracted, p_true)
    coverage = _coverage_mass(field, diseases, distribution)
    expected_margin = p_true * _top_margin(true_dist) + (1.0 - p_true) * _top_margin(false_dist)
    margin_gain = max(0.0, expected_margin - current_margin)

    weights = dict(_UTILITY_WEIGHTS)
    if remaining_budget <= 1:
        weights["info_gain"] += 0.08
        weights["safety"] += 0.05
        weights["coverage"] -= 0.04

    score = (
        weights["info_gain"] * info_gain
        + weights["elimination"] * elimination
        + weights["safety"] * safety
        + weights["coverage"] * coverage
        + weights["margin_gain"] * margin_gain
    )

    return {
        "field": field,
        "score": score,
        "info_gain": info_gain,
        "elimination": elimination,
        "safety": safety,
        "coverage": coverage,
    }


def _estimated_true_probability(field: str, diseases: list[dict[str, Any]], distribution: dict[str, float]) -> float:
    p_true = 0.0
    by_name = {entry["name"]: entry for entry in diseases}
    for name, probability in distribution.items():
        entry = by_name[name]
        p_true += probability * _field_likelihood(field, entry)
    return min(0.95, max(0.05, p_true))


def _field_likelihood(field: str, entry: dict[str, Any]) -> float:
    if field in set(entry.get("required_symptoms", [])):
        return _FIELD_LIKELIHOOD["required"]
    if field in set(entry.get("supporting_symptoms", [])):
        return _FIELD_LIKELIHOOD["supporting"]
    if field in set(entry.get("contraindicating_symptoms", [])):
        return _FIELD_LIKELIHOOD["contraindicating"]
    return _FIELD_LIKELIHOOD["unrelated"]


def _expected_eliminated_mass(
    field: str,
    diseases: list[dict[str, Any]],
    distribution: dict[str, float],
    p_true: float,
) -> float:
    by_name = {entry["name"]: entry for entry in diseases}
    eliminated_if_true = 0.0
    eliminated_if_false = 0.0
    for name, probability in distribution.items():
        entry = by_name[name]
        if field in set(entry.get("contraindicating_symptoms", [])):
            eliminated_if_true += probability
        if field in set(entry.get("required_symptoms", [])):
            eliminated_if_false += probability
    return p_true * eliminated_if_true + (1.0 - p_true) * eliminated_if_false


def _coverage_mass(field: str, diseases: list[dict[str, Any]], distribution: dict[str, float]) -> float:
    by_name = {entry["name"]: entry for entry in diseases}
    mass = 0.0
    for name, probability in distribution.items():
        entry = by_name[name]
        relevant = (
            set(entry.get("required_symptoms", []))
            | set(entry.get("supporting_symptoms", []))
            | set(entry.get("contraindicating_symptoms", []))
        )
        if field in relevant:
            mass += probability
    return mass


def _safety_value(
    field: str,
    diseases: list[dict[str, Any]],
    distribution: dict[str, float],
    extracted: dict[str, Any],
    p_true: float,
) -> float:
    present = {key for key, value in extracted.items() if value is True}
    known = set(extracted.keys())
    combo_value = 0.0
    for combo in RED_FLAG_COMBINATIONS:
        requires = set(combo["requires"])
        if field not in requires:
            continue
        others = requires - {field}
        if others.issubset(present):
            combo_value = max(combo_value, 1.0)
        elif others & present and not others.issubset(known):
            combo_value = max(combo_value, 0.65)
        elif _related_symptom_present(field, present):
            combo_value = max(combo_value, 0.45)

    by_name = {entry["name"]: entry for entry in diseases}
    red_flag_mass = 0.0
    for name, probability in distribution.items():
        entry = by_name[name]
        if not entry.get("red_flag"):
            continue
        relevant = set(entry.get("required_symptoms", [])) | set(entry.get("supporting_symptoms", []))
        required_overlap = bool(set(entry.get("required_symptoms", [])) & present)
        if field in relevant and (required_overlap or _related_symptom_present(field, present)):
            red_flag_mass += probability

    return min(1.0, combo_value + red_flag_mass)


def _entropy(distribution: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in distribution.values() if p > _EPSILON)


def _top_margin(distribution: dict[str, float]) -> float:
    values = sorted(distribution.values(), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return values[0] - values[1]


def _ratio(numerator: set[str], denominator: set[str]) -> float:
    if not denominator:
        return 0.0
    return len(numerator) / len(denominator)


def _related_symptom_present(field: str, present: set[str]) -> bool:
    field_tokens = {
        token for token in field.split("_") if len(token) > 3 and token not in _RELATION_STOP_TOKENS
    }
    for symptom in present:
        symptom_tokens = {
            token for token in symptom.split("_") if len(token) > 3 and token not in _RELATION_STOP_TOKENS
        }
        if field_tokens & symptom_tokens:
            return True
    return False

"""
Orchestrator (the second half of the old "Supervisor" split). Everything
here is a pure decision function consumed by conditional edges in graph.py —
none of it calls an LLM, so all of it is unit-testable directly.

Responsibilities:
  - Loop control: should the follow-up loop continue or stop?
  - Circuit breaker: is the state malformed in a way that should route
    straight to Human Escalation instead of crashing? (§3.1)
  - Escalation policy (recalibrated): the original system had 6 triggers,
    4 of which fired on normal diagnostic ambiguity (low confidence, missing
    symptoms, unsupported combo, missing red-flag info). This caused constant
    over-escalation even when meaningful differential diagnoses were possible.

    The new policy escalates ONLY on genuine safety signals:
        * safety_validator_contradiction — critical flag from the safety model
        * repeated_conflicting_answers   — user directly contradicted themselves

    And one true last-resort:
        * low_calibrated_confidence — confidence stayed below 0.40 even after
          both follow-up turns AND RAG grounding, meaning no reliable KB
          diagnosis can be produced at all.

    Normal cases (confidence 0.40–0.85, missing symptoms, ambiguous combos)
    are handled by the differential diagnosis path — not escalation.
"""
from __future__ import annotations

from typing import Any

from config import ESCALATION_CONFIDENCE_FLOOR
from state import State

REQUIRED_STATE_KEYS = ("domain", "extracted_symptoms", "follow_up_count")

# Minimum KB floor score: if every candidate is below this, there is genuinely
# no match in the knowledge base and last-resort escalation is warranted.
_KB_FLOOR_SCORE = 0.10


def is_state_malformed(state: State) -> bool:
    """Circuit breaker (§3.1): malformed subgraph state routes to Human
    Escalation rather than crashing."""
    for key in REQUIRED_STATE_KEYS:
        if key not in state:
            return True
    if state.get("domain") not in ("eye", "general"):
        return True
    if not isinstance(state.get("extracted_symptoms"), dict):
        return True
    return False


def should_continue_follow_up(confidence_score: float, threshold: float, follow_up_count: int, max_follow_ups: int) -> bool:
    """The bounded follow-up strategy. Continue only if we're below
    threshold AND haven't hit the cap."""
    return confidence_score < threshold and follow_up_count < max_follow_ups


def detect_repeated_conflicting_answers(follow_up_history: list[dict[str, Any]]) -> bool:
    """
    Flags when the normalized answer to the same `asked_for` field flips
    between turns (e.g. user says 'no itching' then later 'yes it itches').

    This is a genuine safety signal: a self-assessment tool should defer
    when the patient's own account is contradictory, rather than silently
    picking whichever answer came last.

    A single flip fires the trigger — one contradiction is enough to warrant
    human review when patient safety is involved.
    """
    seen: dict[str, str] = {}
    for qa in follow_up_history:
        field = qa.get("asked_for", "")
        answer_norm = qa.get("answer", "").strip().lower()
        if not field or not answer_norm:
            continue
        polarity = "yes" if answer_norm in ("yes", "true", "y") else ("no" if answer_norm in ("no", "false", "n") else None)
        if polarity is None:
            continue
        if field in seen and seen[field] != polarity:
            return True
        seen[field] = polarity
    return False


def _no_kb_match(candidate_diseases: list[dict[str, Any]]) -> bool:
    """Returns True when every candidate disease scores below the KB floor —
    meaning the symptom set genuinely doesn't match anything in the knowledge
    base, not just that confidence is low."""
    if not candidate_diseases:
        return True
    return max(c["score"] for c in candidate_diseases) < _KB_FLOOR_SCORE


def evaluate_escalation_triggers(state: State) -> list[str]:
    """
    Runs the (now narrowed) set of escalation triggers and returns the names
    of those that fired. An empty list means: proceed to normal differential
    diagnosis output — do NOT escalate.

    Recalibrated trigger set (from 6 → 3):

    KEPT (genuine safety signals):
      - safety_validator_contradiction: critical model-flagged contradiction
      - repeated_conflicting_answers: patient gave contradictory answers
      - low_calibrated_confidence: truly no KB match after full pipeline

    REMOVED (were causing over-escalation on normal ambiguity):
      - incomplete_critical_info: handled by follow-up questions instead
      - unsupported_symptom_combination: handled by differential output instead
      - missing_red_flag_information: handled by follow-up questions instead
    """
    fired: list[str] = []

    # 1. Safety contradiction — always escalate, no confidence floor applies
    if any(f["severity"] == "critical" for f in state.get("safety_flags", [])):
        fired.append("safety_validator_contradiction")

    # 2. User self-contradiction — always escalate, cannot trust symptom data
    if detect_repeated_conflicting_answers(state.get("follow_up_history", [])):
        fired.append("repeated_conflicting_answers")

    # 3. Last-resort: truly no KB match after the full pipeline completes.
    # Only fire when:
    #   a) calibrated_confidence is below the escalation floor (< 0.40), AND
    #   b) all follow-up turns are exhausted (can't improve with more questions), AND
    #   c) no disease scores above the KB floor (genuinely no match, not just low confidence).
    # This avoids escalating merely because confidence is 0.45–0.65 — those
    # cases should produce a differential, not a human referral.
    calibrated = state.get("calibrated_confidence")
    follow_up_exhausted = state.get("follow_up_count", 0) >= state.get("max_follow_ups", 2)
    candidates = state.get("candidate_diseases", [])
    if (
        calibrated is not None
        and calibrated < ESCALATION_CONFIDENCE_FLOOR
        and follow_up_exhausted
        and _no_kb_match(candidates)
    ):
        fired.append("low_calibrated_confidence")

    return fired

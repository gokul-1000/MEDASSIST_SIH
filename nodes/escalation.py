"""
Human Escalation (§3.12). Triggered by the expanded policy in
nodes/orchestrator.py::evaluate_escalation_triggers. Instead of forcing a
prediction, emits an explicit "insufficient confidence" message plus a
coarse, still-rule-based department recommendation, and an explicit
recommendation to see a specialist.

A healthcare system that knows when to defer is safer than one that always
commits to an answer — this is a first-class terminal state, not a failure
path, and the graph should never feel bad about landing here.
"""
from __future__ import annotations

from typing import Any

from config import ESCALATION_TRIGGERS
from nodes.orchestrator import evaluate_escalation_triggers, is_state_malformed
from rules.rule_engine import recommend_department
from state import State


def escalation_node(state: State) -> dict[str, Any]:
    domain = state.get("domain") if state.get("domain") in ("eye", "general") else "general"

    if is_state_malformed(state):
        # Reached via the circuit breaker (§3.1), not the normal escalation
        # policy — don't claim a confidence-related reason that isn't true.
        reason_text = "internal state was inconsistent for this session and could not be safely processed"
    else:
        fired_triggers = evaluate_escalation_triggers(state)
        reasons = [ESCALATION_TRIGGERS.get(t, t) for t in fired_triggers] or ["confidence remained below threshold"]
        reason_text = "; ".join(reasons)

    department = recommend_department(predicted_disease=None, domain=domain, kb_disease_entry=None)

    urgency_note = ""
    emergency_level = state.get("emergency_level")
    if emergency_level in (2, 3):
        urgency_note = " This also looks like it may need prompt attention — please don't delay seeking care."

    message = (
        "I don't have enough confidence to predict a specific condition from what you've told me "
        f"({reason_text}). Based on the pattern of symptoms, I'd recommend seeing a specialist in "
        f"{department} for a proper evaluation.{urgency_note}"
    )

    return {
        "escalated": True,
        "escalation_reason": reason_text,
        "recommended_department": department,
        "final_response": message,
    }

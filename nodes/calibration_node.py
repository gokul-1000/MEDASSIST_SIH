"""
Confidence Calibration (§3.10), as a graph node. This is deliberately a
DIFFERENT node from confidence_engine_node.py — see confidence.py's module
docstring for why the two must never be conflated:

  - Confidence Engine  -> "keep asking questions, or stop?" (in-loop)
  - Confidence Calibration -> "is this specific finished prediction
    trustworthy?" (post-diagnosis, terminal)

Runs once, right after Diagnosis + Safety Validator, in parallel with
Emergency Classification and Department Recommendation (§4).

Change from original: `candidate_diseases` is now passed to
`calibrate_confidence()` so it can compute the score-margin component and
competition penalties — both key to preventing overconfidence when multiple
diseases score similarly. Previously this context was unavailable to the
calibrator, causing it to miss the most common source of diagnostic uncertainty.
"""
from __future__ import annotations

from typing import Any

from confidence import calibrate_confidence
from knowledge_base.kb_loader import load_kb
from state import State


def calibration_node(state: State) -> dict[str, Any]:
    domain = state["domain"]
    kb = load_kb(domain)
    entry = kb.disease_by_name(state.get("predicted_disease", ""))

    result = calibrate_confidence(
        llm_confidence=state.get("raw_model_confidence", 0.5),
        extracted_symptoms=state.get("extracted_symptoms", {}),
        disease_entry=entry,
        rag_similarity=state.get("rag_similarity"),
        historical_confidence_trend=state.get("confidence_history", []),
        # Pass candidate_diseases so the calibrator can compute score-margin
        # and competition penalties — the primary fix for overconfidence.
        candidate_diseases=state.get("candidate_diseases", []),
    )
    return {"calibrated_confidence": result.calibrated_confidence}

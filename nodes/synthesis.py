"""
Response Synthesis (§3.15). Small model, template-constrained — this is a
formatting task, not reasoning, so it never needs a large model. Joins the
parallel fan-out of Confidence Calibration, Emergency Classification, and
Department Recommendation (or, if the graph routed to Human Escalation
instead, just passes that message through unchanged).

Improved from original: the synthesis prompt now receives the full differential
diagnosis list and calibrated confidence tier, and produces tiered output:

  - High confidence   (> 0.85): single best-disease answer with brief reasoning
  - Medium confidence (0.60–0.85): differential with uncertainty explanation
  - Lower confidence  (< 0.60): differential plus explicit follow-up advice

This ensures the system never pretends certainty it doesn't have, and always
explains what additional information would raise confidence further.

The "no_match_in_kb" sentinel is translated here into the user-facing message:
"No reliable match was found within the current knowledge base."
"""
from __future__ import annotations

from typing import Any

from config import PREDICTION_THRESHOLDS
from models.model_router import ModelRouter
from state import State

_NO_MATCH_SENTINEL = "no_match_in_kb"

# ---------------------------------------------------------------------------
# Synthesis prompt — tiered by confidence level
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_TEMPLATE = """You are formatting a final response for a medical self-assessment tool user.
task=response_synthesis
Predicted condition: {predicted_disease}
Calibrated confidence: {calibrated_confidence_pct}% ({confidence_tier})
Differential diagnosis (ranked): {differential_summary}
Emergency level: {emergency_level} (1=routine, 2=prompt care advised, 3=seek immediate/emergency care)
Recommended department: {recommended_department}

Response rules by confidence tier:
- HIGH confidence (>85%): State the most likely condition clearly. Give 1-2 sentences of reasoning from the matched symptoms.
- MEDIUM confidence (60-85%): State the most likely condition, then list 1-2 alternatives. Explain what is uncertain.
- LOW confidence (<60%): List the top candidates. Explain that more information is needed. Recommend a follow-up appointment.

For all tiers:
- Do NOT include any dosage, medication, or treatment instructions.
- Be plain-language and empathetic — this is a self-assessment tool, not a clinical report.
- Keep to 3-5 sentences maximum.
- If predicted_disease is "no_match_in_kb", say: "No reliable match was found within the current knowledge base." and recommend seeing a physician.

Respond with strict JSON: {{"response": "..."}}"""


def _build_confidence_tier(calibrated_confidence: float) -> str:
    """Map calibrated confidence to the tier label used in the synthesis prompt."""
    if calibrated_confidence >= PREDICTION_THRESHOLDS["high_confidence"]:
        return "HIGH"
    if calibrated_confidence >= PREDICTION_THRESHOLDS["medium_confidence"]:
        return "MEDIUM"
    return "LOW"


def _summarise_differential(differential: list[dict[str, Any]] | None) -> str:
    """
    Produce a compact, readable summary of the differential for the synthesis
    prompt. Keeps the prompt focused — full matched/missing symptom lists
    would overwhelm the small synthesis model.
    """
    if not differential:
        return "No differential available."

    parts: list[str] = []
    for entry in differential:
        rank = entry.get("rank", "?")
        name = entry.get("disease_name", "Unknown")
        score = entry.get("score", 0.0)
        matched = entry.get("matched_symptoms", [])
        missing = entry.get("missing_required_symptoms", [])

        matched_str = ", ".join(s.replace("_", " ") for s in matched[:3]) or "none confirmed"
        missing_str = ", ".join(s.replace("_", " ") for s in missing[:2])

        line = f"#{rank} {name} (score={score:.2f}): matched=[{matched_str}]"
        if missing_str:
            line += f", missing required=[{missing_str}]"
        parts.append(line)

    return " | ".join(parts)


def make_synthesis_node(model_router: ModelRouter):
    def synthesis_node(state: State) -> dict[str, Any]:
        if state.get("escalated"):
            # escalation_node already wrote a complete, safe message.
            return {"final_response": state.get("final_response", "")}

        predicted_disease = state.get("predicted_disease", "")
        calibrated_confidence = state.get("calibrated_confidence", 0.0)
        differential = state.get("differential_diagnosis") or []

        # Handle the "no KB match" sentinel explicitly before calling the LLM
        if predicted_disease == _NO_MATCH_SENTINEL:
            dept = state.get("recommended_department", "General Medicine")
            no_match_msg = (
                "No reliable match was found within the current knowledge base for your symptom pattern. "
                f"I recommend consulting a physician in {dept} for a proper evaluation."
            )
            return {"final_response": no_match_msg}

        confidence_tier = _build_confidence_tier(calibrated_confidence)
        confidence_pct = int(round(calibrated_confidence * 100))
        differential_summary = _summarise_differential(differential)

        system_prompt = _SYNTHESIS_SYSTEM_TEMPLATE.format(
            predicted_disease=predicted_disease,
            calibrated_confidence_pct=confidence_pct,
            confidence_tier=confidence_tier,
            differential_summary=differential_summary,
            emergency_level=state.get("emergency_level"),
            recommended_department=state.get("recommended_department"),
        )
        resp = model_router.call(
            tier="small",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Write the response."},
            ],
        )

        # Fallback template preserves the differential even if the LLM fails
        if differential:
            top = differential[0]
            alt_names = [e["disease_name"] for e in differential[1:]]
            alt_str = f" (also consider: {', '.join(alt_names)})" if alt_names else ""
            fallback = (
                f"Most likely: {predicted_disease}{alt_str}. "
                f"Confidence: {confidence_pct}%. "
                f"Recommended department: {state.get('recommended_department')}."
            )
        else:
            fallback = (
                f"Likely condition: {predicted_disease}. "
                f"Recommended department: {state.get('recommended_department')}."
            )

        response_text = (resp.response.parsed or {}).get("response", fallback)
        return {"final_response": response_text}

    return synthesis_node

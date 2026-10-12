"""
Two distinct confidence mechanisms, deliberately kept in one file so the
distinction (the refinement you flagged — "clarify the confidence
architecture") is impossible to miss:

1. `compute_loop_confidence`  — the Confidence Engine (§3.5). Runs every
   turn inside the follow-up loop. Question it answers: "do we have enough
   to stop asking questions?" Deterministic, no LLM call.

2. `calibrate_confidence`     — Confidence Calibration (§3.10). Runs once,
   after Diagnosis. Question it answers: "is this specific prediction
   trustworthy enough to show the user?" Penalty-based formula — starts
   from an evidence-weighted base score, then subtracts uncertainty penalties
   so the output reflects real diagnostic certainty rather than model
   self-reported confidence (a known overconfidence failure mode).

3. `build_differential`       — Deterministic differential ranking. Constructs
   a top-3 ranked differential diagnosis list directly from KB symptom
   matching, with per-disease reasoning, without any LLM call. All diseases
   in the output are guaranteed to be KB entries (no hallucination).

All are pure functions over plain dicts/lists — no LangGraph, no network,
fully unit-testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from config import CALIBRATION_PENALTIES, CALIBRATION_WEIGHTS


# ---------------------------------------------------------------------------
# Internal symptom scoring
# ---------------------------------------------------------------------------

def _symptom_match_score(extracted_symptoms: dict[str, Any], disease_entry: dict[str, Any]) -> float:
    """How well does what we know match this one candidate's expected pattern?"""
    required = set(disease_entry.get("required_symptoms", []))
    supporting = set(disease_entry.get("supporting_symptoms", []))
    contraindicating = set(disease_entry.get("contraindicating_symptoms", []))

    present = {k for k, v in extracted_symptoms.items() if v is True}
    absent_known = {k for k, v in extracted_symptoms.items() if v is False}

    if required & absent_known:
        return 0.0  # a required symptom is missing or explicitly denied — ruled out
    if contraindicating & present:
        return 0.0  # a contraindicating symptom is present — ruled out

    base = disease_entry.get("base_rate", 0.1)
    required_hits = len(required & present)
    required_total = max(len(required), 1)
    required_ratio = required_hits / required_total
    support_hits = len(supporting & present)
    support_total = max(len(supporting), 1)
    support_ratio = support_hits / support_total
    support_denial_penalty = 0.08 * len(supporting & absent_known)

    # Denying a CONTRAINDICATING symptom is positive evidence for this
    # candidate (ruling out a more alarming alternative's warning sign
    # supports the milder one) — this matters most for diagnoses-by-
    # exclusion (e.g. subconjunctival_hemorrhage, whose entire signature IS
    # the absence of pain/discharge/vision change), which would otherwise
    # have no mechanism to accumulate any score at all.
    #
    # NOTE: an earlier version of this function also gave a bonus for
    # denying a candidate's own SUPPORTING symptoms, on the theory that "a
    # no narrows things down too." That was backwards in practice — it
    # rewarded a candidate for lacking its own supportive evidence, which
    # let candidates whose supporting list happened to overlap with a rarer
    # candidate's contraindicating list win purely on priors. Removed.
    contraindication_denial_bonus = 0.05 * len(contraindicating & absent_known)

    score = base + 0.35 * required_ratio + 0.45 * support_ratio + contraindication_denial_bonus
    score -= support_denial_penalty
    return max(0.0, min(1.0, score))


def rank_candidates(extracted_symptoms: dict[str, Any], diseases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Used by the probabilistic disease router (nodes/router.py) and re-run by the Planner every turn."""
    scored = [
        {"name": d["name"], "score": _symptom_match_score(extracted_symptoms, d)}
        for d in diseases
    ]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Confidence Engine (§3.5) — in-loop stop/continue decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoopConfidenceResult:
    score: float
    should_stop: bool
    leading_candidate: str | None
    margin: float  # gap between #1 and #2 candidate — a big margin means we're not just guessing


def compute_loop_confidence(
    candidates: list[dict[str, Any]],
    threshold: float,
    follow_up_count: int,
    max_follow_ups: int,
    extracted_symptoms: dict[str, Any] | None = None,
) -> LoopConfidenceResult:
    """
    Confidence Engine (§3.5). Deterministic function over extracted_symptoms
    vs. candidate_diseases — no LLM call. Writes confidence_score; the
    conditional edge in graph.py compares it against confidence_threshold.

    Improvements over the original:
    - Information-sparsity cap: fewer than 4 known symptoms → score capped at
      0.70 regardless of match quality. The raw symptom picture is too thin
      to support high confidence at this stage.
    - Entropy-based competitive cluster penalty: when multiple candidates sit
      within 0.15 of the top score, the effective score is pulled downward
      proportionally so the loop continues to differentiate them rather than
      stopping prematurely on an ambiguous result.
    """
    if not candidates:
        return LoopConfidenceResult(score=0.0, should_stop=False, leading_candidate=None, margin=0.0)

    top = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    margin = top["score"] - second_score

    # Margin factor: a high raw score with a thin margin (two diseases are
    # neck-and-neck) means we're not actually confident — the loop should
    # continue to discriminate them. A large margin earns the full score.
    margin_factor = min(margin / 0.30, 1.0)  # fully saturates at a 0.30 gap
    effective_score = top["score"] * (0.65 + 0.35 * margin_factor)

    # Competitive cluster penalty: count how many non-top candidates are
    # within 0.15 of the leading score. Each close competitor reduces the
    # effective score an additional 5%, encouraging more questions.
    close_competitors = sum(
        1 for c in candidates[1:] if (top["score"] - c["score"]) <= 0.15
    )
    if close_competitors > 0:
        effective_score *= max(0.5, 1.0 - 0.05 * close_competitors)

    # Information-sparsity cap: if we know fewer than 4 symptoms total,
    # we don't have enough evidence to be confident regardless of match score.
    known_count = len(extracted_symptoms) if extracted_symptoms else 0
    if known_count < 4:
        effective_score = min(effective_score, 0.70)

    hit_cap = follow_up_count >= max_follow_ups
    should_stop = effective_score >= threshold or hit_cap

    return LoopConfidenceResult(
        score=round(effective_score, 4),
        should_stop=should_stop,
        leading_candidate=top["name"],
        margin=round(margin, 4),
    )


# ---------------------------------------------------------------------------
# Confidence Calibration (§3.10) — post-diagnosis trustworthiness signal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationResult:
    calibrated_confidence: float
    components: dict[str, float]  # kept for observability / debugging
    penalty_breakdown: dict[str, float]  # which penalties fired and by how much


def calibrate_confidence(
    llm_confidence: float,
    extracted_symptoms: dict[str, Any],
    disease_entry: dict[str, Any] | None,
    rag_similarity: float | None,
    historical_confidence_trend: list[float],
    candidate_diseases: list[dict[str, Any]] | None = None,
    weights: dict[str, float] | None = None,
    penalties: dict[str, float] | None = None,
) -> CalibrationResult:
    """
    Confidence Calibration (§3.10). Penalty-based formula — runs once, right
    after Diagnosis. Produces a calibrated score that reflects real uncertainty
    rather than model overconfidence.

    Algorithm:
      1. Compute a weighted base from evidence signals.
      2. Apply a margin bonus if the top candidate has a clear lead.
      3. Subtract uncertainty penalties:
           - Competition penalty  — close competitors signal ambiguity
           - Missing-required penalty — we haven't confirmed key symptoms
           - Sparsity penalty  — too little information was provided overall
      4. Clamp result to [0.0, 0.97]. Confidence should almost never reach 1.0.

    All parameters are configurable from config.py so thresholds can be
    tuned without touching algorithm code.
    """
    w = weights or CALIBRATION_WEIGHTS
    p = penalties or CALIBRATION_PENALTIES

    # --- Base signal components ---

    if disease_entry:
        relevant_fields = (
            set(disease_entry.get("required_symptoms", []))
            | set(disease_entry.get("supporting_symptoms", []))
        )
        known = sum(1 for f in relevant_fields if f in extracted_symptoms)
        completeness = known / max(len(relevant_fields), 1)
    else:
        completeness = 0.0

    # RAG similarity defaults to LLM confidence (neutral) when RAG wasn't used
    rag_component = rag_similarity if rag_similarity is not None else llm_confidence

    trend = historical_confidence_trend or [llm_confidence]
    historical = sum(trend) / len(trend)

    # Margin between #1 and #2 candidate (requires candidate_diseases context)
    top_score = 0.0
    second_score = 0.0
    if candidate_diseases:
        top_score = candidate_diseases[0]["score"] if candidate_diseases else 0.0
        second_score = candidate_diseases[1]["score"] if len(candidate_diseases) > 1 else 0.0
    score_margin = top_score - second_score

    components = {
        "llm_confidence": llm_confidence,
        "symptom_completeness": completeness,
        "rag_similarity": rag_component,
        "score_margin": score_margin,
        "historical_trend": historical,
    }

    # Compute weighted base — uses only the keys in CALIBRATION_WEIGHTS
    base = sum(components[k] * w[k] for k in w if k in components)

    # --- Margin bonus/penalty ---
    # A clear lead between #1 and #2 scores (≥ 0.30) earns a small bonus;
    # a neck-and-neck result (< 0.10 margin) earns a penalty instead.
    if score_margin >= p["margin_bonus_threshold"]:
        margin_adjustment = p["margin_bonus_max"] * min(score_margin / p["margin_bonus_threshold"], 1.0)
    elif score_margin < p["margin_penalty_threshold"]:
        margin_adjustment = -p["margin_penalty"]
    else:
        # Linear interpolation between penalty and neutral
        margin_adjustment = -p["margin_penalty"] * (
            1.0 - (score_margin - p["margin_penalty_threshold"])
            / (p["margin_bonus_threshold"] - p["margin_penalty_threshold"])
        )

    # --- Competition penalty ---
    # Each close competitor (within 0.15 of the top score) signals diagnostic
    # ambiguity and reduces confidence. Ambiguous results should be shown as
    # differentials, not high-confidence single-disease predictions.
    close_competitors = 0
    if candidate_diseases:
        close_competitors = sum(
            1 for c in candidate_diseases[1:] if (top_score - c["score"]) <= 0.15
        )
    competition_penalty = p["competing_disease"] * close_competitors

    # --- Missing required symptom penalty ---
    # Each required symptom for the predicted disease that was never confirmed
    # (not present in extracted_symptoms) reduces confidence. We don't know if
    # the patient has it — absence of data is not absence of symptom.
    missing_required = 0
    if disease_entry:
        required_symptoms = set(disease_entry.get("required_symptoms", []))
        present = {k for k, v in extracted_symptoms.items() if v is True}
        missing_required = len(required_symptoms - present - set(extracted_symptoms.keys()))
    raw_missing_penalty = p["missing_required_symptom"] * missing_required
    missing_penalty = min(raw_missing_penalty, p["missing_required_cap"])

    # --- Information sparsity penalty ---
    # If fewer than `sparsity_threshold` fraction of the disease's relevant
    # symptoms are known, the diagnosis is based on very sparse information.
    sparsity_penalty = 0.0
    if disease_entry:
        relevant_fields = (
            set(disease_entry.get("required_symptoms", []))
            | set(disease_entry.get("supporting_symptoms", []))
        )
        known_count = sum(1 for f in relevant_fields if f in extracted_symptoms)
        known_fraction = known_count / max(len(relevant_fields), 1)
        if known_fraction < p["sparsity_threshold"]:
            sparsity_penalty = p["info_sparsity"]

    # --- Final calibrated score ---
    calibrated = base + margin_adjustment - competition_penalty - missing_penalty - sparsity_penalty

    penalty_breakdown = {
        "margin_adjustment": round(margin_adjustment, 4),
        "competition_penalty": round(competition_penalty, 4),
        "missing_required_penalty": round(missing_penalty, 4),
        "sparsity_penalty": round(sparsity_penalty, 4),
        "total_penalty": round(competition_penalty + missing_penalty + sparsity_penalty - margin_adjustment, 4),
    }

    # Clamp to [0.0, 0.97] — the system should almost never claim certainty
    final = round(max(0.0, min(0.97, calibrated)), 4)

    return CalibrationResult(
        calibrated_confidence=final,
        components=components,
        penalty_breakdown=penalty_breakdown,
    )


# ---------------------------------------------------------------------------
# Differential Diagnosis Builder — deterministic, KB-constrained
# ---------------------------------------------------------------------------

@dataclass
class DifferentialEntry:
    """One ranked entry in the differential diagnosis list."""
    rank: int
    disease_name: str
    score: float                          # raw KB match score (0–1)
    matched_symptoms: list[str]           # symptoms present that support this disease
    missing_required_symptoms: list[str]  # required symptoms not yet confirmed
    contradictory_symptoms: list[str]     # symptoms present that contraindicate this disease
    reasoning: str                        # why this disease matches the symptom picture
    confidence_ceiling_reason: str        # why confidence isn't higher (what would raise it)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "disease_name": self.disease_name,
            "score": self.score,
            "matched_symptoms": self.matched_symptoms,
            "missing_required_symptoms": self.missing_required_symptoms,
            "contradictory_symptoms": self.contradictory_symptoms,
            "reasoning": self.reasoning,
            "confidence_ceiling_reason": self.confidence_ceiling_reason,
        }


def _build_reasoning(
    entry: dict[str, Any],
    matched: list[str],
    missing_required: list[str],
    contradictory: list[str],
) -> tuple[str, str]:
    """
    Construct human-readable reasoning and confidence-ceiling explanation
    from symptom intersection data. Pure string logic — no LLM.

    Returns: (reasoning, confidence_ceiling_reason)
    """
    name = entry["name"]

    if matched:
        symptom_list = ", ".join(s.replace("_", " ") for s in matched[:4])
        reasoning = f"{name} matches because the following symptoms are present: {symptom_list}."
        if entry.get("base_rate", 0.1) > 0.15:
            reasoning += f" It is also a relatively common condition ({int(entry['base_rate']*100)}% base rate)."
    else:
        # Score came from contraindicating-denial bonus or prior only
        reasoning = (
            f"{name} is a candidate primarily because the symptom pattern does not rule it out, "
            "and its contraindicating symptoms were not reported."
        )

    # Confidence ceiling reasons — ordered by descending impact
    ceiling_parts: list[str] = []
    if missing_required:
        keys = ", ".join(s.replace("_", " ") for s in missing_required[:3])
        ceiling_parts.append(f"required symptoms not yet confirmed: {keys}")
    if contradictory:
        keys = ", ".join(s.replace("_", " ") for s in contradictory[:2])
        ceiling_parts.append(f"contradictory symptoms present: {keys}")
    if not matched:
        ceiling_parts.append("no directly matching symptoms confirmed yet")
    if not ceiling_parts:
        ceiling_parts.append("some supporting symptoms are still unconfirmed")

    confidence_ceiling_reason = "Confidence is not higher because: " + "; ".join(ceiling_parts) + "."
    return reasoning, confidence_ceiling_reason


def build_differential(
    extracted_symptoms: dict[str, Any],
    diseases: list[dict[str, Any]],
    top_n: int = 3,
) -> list[DifferentialEntry]:
    """
    Build a ranked differential diagnosis from the KB. Guaranteed to contain
    only diseases that exist in `diseases` (the KB list) — no hallucination.

    Each entry includes:
    - Matched symptoms (support this disease)
    - Missing required symptoms (needed for stronger evidence)
    - Contradictory symptoms (present but shouldn't be for this disease)
    - Deterministic reasoning text explaining the match
    - Explanation of why confidence isn't higher

    The final diagnosis list is always a subset of retrieved KB diseases.
    """
    present = {k for k, v in extracted_symptoms.items() if v is True}
    absent_known = {k for k, v in extracted_symptoms.items() if v is False}

    scored_entries = []
    for entry in diseases:
        score = _symptom_match_score(extracted_symptoms, entry)
        scored_entries.append((score, entry))

    # Sort descending by score; break ties by disease name for determinism
    scored_entries.sort(key=lambda x: (-x[0], x[1]["name"]))

    # Filter out zero-scored diseases (hard ruled out by required/contraindication)
    # then take top_n
    viable = [(score, e) for score, e in scored_entries if score > 0.0]
    top_entries = viable[:top_n]

    differential: list[DifferentialEntry] = []
    for rank, (score, entry) in enumerate(top_entries, start=1):
        required = set(entry.get("required_symptoms", []))
        supporting = set(entry.get("supporting_symptoms", []))
        contraindicating = set(entry.get("contraindicating_symptoms", []))

        matched = sorted((required | supporting) & present)
        missing_required = sorted(required - present)  # required but not yet confirmed
        contradictory = sorted(contraindicating & present)   # should rule this out

        reasoning, ceiling_reason = _build_reasoning(entry, matched, missing_required, contradictory)

        differential.append(DifferentialEntry(
            rank=rank,
            disease_name=entry["name"],
            score=round(score, 4),
            matched_symptoms=matched,
            missing_required_symptoms=missing_required,
            contradictory_symptoms=contradictory,
            reasoning=reasoning,
            confidence_ceiling_reason=ceiling_reason,
        ))

    return differential

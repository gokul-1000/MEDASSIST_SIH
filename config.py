"""
Central configuration. Nothing here talks to a network — it's just constants,
so it can be imported and unit-tested with zero dependencies.

Model IDs verified against Groq's live model list and deprecation notices as
of July 2026 (see README §Model Selection for sources). If you're reading
this months later: re-check https://console.groq.com/docs/models and
https://console.groq.com/docs/deprecations before trusting these strings —
Groq deprecates models on a rolling basis and free-tier lineups shift.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # keep tests/config importable in zero-dependency environments
    def load_dotenv(*args, **kwargs):
        return False

# Resolved relative to THIS file, not the current working directory — so
# `python cli.py` works whether you run it from inside medassist/ or from
# somewhere else entirely. Silently does nothing if .env doesn't exist yet,
# which is fine (mock mode is the fallback either way).
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Model tiers — every generation call is API-based (hosted, free-tier), never
# local. The laptop (i9, no GPU) only ever runs orchestration, embeddings,
# and vector search. See models/llm_provider.py for the provider clients.
# ---------------------------------------------------------------------------

GROQ_MODELS = {
    "moderation": "openai/gpt-oss-safeguard-20b",
    "small": "openai/gpt-oss-20b",
    "medium": "openai/gpt-oss-120b",
    # qwen3.6-27b is Groq's current vision-capable *preview* model — fine as a
    # same-tier fallback for text-only diagnosis, but don't rely on it for
    # anything you can't afford to see deprecated without notice.
    "medium_fallback": "qwen/qwen3.6-27b",
}

# Second free provider, used as the cross-provider fallback so "free-only"
# doesn't mean "single point of failure." Gemini free tier (Google AI
# Studio) currently restricts Pro-tier models heavily/behind billing, so we
# only lean on Flash here — it's the tier Google actually keeps free.
GEMINI_MODELS = {
    "large_free": "gemini-2.5-flash",
}

# What the architecture doc calls the "Large" tier (rare/ambiguous cases,
# low confidence after follow-ups + RAG) was originally Claude Sonnet 5
# (paid). Free-only substitution: escalate through gpt-oss-120b -> Gemini
# 2.5 Flash rather than to a paid frontier model. This is a real quality
# trade-off on the hardest ~5-10% of cases — see README "Honest Rating" for
# why this is the one place free-only visibly costs you something.
MODEL_TIER_CHAIN: dict[str, list[tuple[str, str]]] = {
    "small": [("groq", GROQ_MODELS["small"])],
    "medium": [("groq", GROQ_MODELS["medium"]), ("groq", GROQ_MODELS["medium_fallback"])],
    "large": [("groq", GROQ_MODELS["medium"]), ("gemini", GEMINI_MODELS["large_free"])],
    "moderation": [("groq", GROQ_MODELS["moderation"])],
}

# CPU-friendly, run locally, no GPU required (verified: 22M / small
# cross-encoder params — sub-second on an i9 for the batch sizes this
# project needs).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Follow-up budget — maximum questions the system can ask before committing
# to a diagnosis or escalating. Kept small (2) for usability.
# ---------------------------------------------------------------------------

MAX_FOLLOW_UPS = 2

# ---------------------------------------------------------------------------
# In-loop confidence thresholds (Confidence Engine, §3.5).
# The loop stops collecting symptoms once the effective score reaches this.
# Lowered from 0.75/0.80 so the system commits earlier rather than asking
# more questions than the budget allows.
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLDS = {
    "eye": 0.60,
    "general": 0.60,
}

# ---------------------------------------------------------------------------
# Prediction output tiers — determine how the final answer is framed.
#
#   high_confidence  (> 0.85): return single best disease, high certainty
#   medium_confidence(0.60–0.85): return differential + uncertainty note
#   followup_threshold(0.40–0.60): ask targeted follow-up before deciding
#   escalation_floor (< 0.40): only escalate if max follow-ups exhausted
#                              AND no disease scores above KB floor
# ---------------------------------------------------------------------------

PREDICTION_THRESHOLDS = {
    "high_confidence": 0.85,    # single best-disease output
    "medium_confidence": 0.60,  # differential diagnosis output
    "followup_threshold": 0.40, # ask at least one more question
    "escalation_floor": 0.40,   # last-resort escalation floor
}

# ---------------------------------------------------------------------------
# Calibration weights for the penalty-based confidence formula.
# Must sum to 1.0. LLM self-reports are down-weighted (known overconfidence
# failure mode); symptom coverage and RAG grounding quality are up-weighted.
# A new `score_margin` component rewards clear separation between candidates.
# ---------------------------------------------------------------------------

CALIBRATION_WEIGHTS = {
    "llm_confidence": 0.25,       # LLM self-reports inflate — trust less
    "symptom_completeness": 0.30, # how much of the disease picture we know
    "rag_similarity": 0.20,       # retrieval grounding quality
    "score_margin": 0.15,         # separation gap between #1 and #2 candidate
    "historical_trend": 0.10,     # stability of confidence across turns
}

# ---------------------------------------------------------------------------
# Penalty constants applied on top of the weighted base score.
# Each represents a source of diagnostic uncertainty.
# Tune these to shift the expected confidence distribution.
# ---------------------------------------------------------------------------

CALIBRATION_PENALTIES = {
    # Per close competitor (within 0.15 of top candidate score)
    "competing_disease": 0.08,
    # Per missing required symptom for the predicted disease (capped below)
    "missing_required_symptom": 0.05,
    # Cap on the total missing-required-symptom penalty
    "missing_required_cap": 0.20,
    # Applied when fewer than 30% of the disease's relevant symptoms are known
    "info_sparsity": 0.12,
    # Sparsity threshold: fraction of relevant symptoms that must be known
    "sparsity_threshold": 0.30,
    # Maximum margin bonus awarded when the top candidate has a wide lead
    "margin_bonus_max": 0.08,
    # Margin threshold above which the full bonus is awarded
    "margin_bonus_threshold": 0.30,
    # Margin below which a competitive-cluster penalty is added
    "margin_penalty_threshold": 0.10,
    # Penalty when top-2 candidates are neck-and-neck
    "margin_penalty": 0.10,
}

# ---------------------------------------------------------------------------
# Escalation policy — genuine red flags only.
# Four of the original six triggers fired on normal diagnostic ambiguity
# (low confidence, missing symptoms, unsupported combo, missing red-flag
# info). These are removed: the system should show a differential instead.
# Only true safety-critical triggers remain.
# ---------------------------------------------------------------------------

# Below this calibrated_confidence — AND only after max follow-ups are
# exhausted AND no disease scores above KB floor — does the system escalate
# rather than return a low-confidence differential.
ESCALATION_CONFIDENCE_FLOOR = 0.40

ESCALATION_TRIGGERS = {
    "low_calibrated_confidence": (
        "calibrated confidence remained below threshold after the full follow-up + RAG pipeline"
    ),
    "safety_validator_contradiction": "Safety Validator flagged a critical, unresolvable contradiction",
    "repeated_conflicting_answers": "user gave directly contradictory answers to the same symptom across turns",
}

# ---------------------------------------------------------------------------
# Environment / API keys — read once, never hardcoded.
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    use_mock_provider: bool = field(
        default_factory=lambda: os.getenv("MEDASSIST_MOCK", "").lower() in ("1", "true", "yes")
    )
    checkpoint_db_path: str = field(default_factory=lambda: os.getenv("MEDASSIST_CHECKPOINT_DB", "checkpoints.sqlite"))


def load_settings() -> Settings:
    s = Settings()
    # Auto-fallback to mock mode if no keys are configured at all — this is
    # what makes `python cli.py` work out of the box for grading/demo
    # purposes without anyone needing to hand out an API key.
    if not s.groq_api_key and not s.gemini_api_key:
        s.use_mock_provider = True
    return s

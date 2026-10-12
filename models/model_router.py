"""
Implements two of your requested refinements at once:

1. Tiered model routing with fallbacks (already in the architecture doc §5,
   §11) — primary model -> same-tier fallback -> cross-provider fallback.
2. Confidence-aware model routing (your new refinement) — after follow-ups,
   if confidence is still low, the NEXT call (typically Diagnosis) is
   escalated to the "large" tier + RAG rather than staying on "medium".

`fallback_applied` is written purely for observability — it never changes
graph control flow, only which model served the call (mirrors §5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from config import MODEL_TIER_CHAIN, Settings
from models.llm_provider import GeminiProvider, GroqProvider, LLMProvider, MockProvider, ProviderError, ProviderResponse

logger = logging.getLogger("medassist.model_router")

Tier = str  # "moderation" | "small" | "medium" | "large"


@dataclass
class RoutedResponse:
    response: ProviderResponse
    tier_requested: Tier
    tier_served_by: Tier
    fallback_applied: bool


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._providers: dict[str, LLMProvider] = {}
        if settings.use_mock_provider:
            mock = MockProvider()
            self._providers = {"groq": mock, "gemini": mock}
        else:
            if settings.groq_api_key:
                self._providers["groq"] = GroqProvider(settings.groq_api_key)
            if settings.gemini_api_key:
                self._providers["gemini"] = GeminiProvider(settings.gemini_api_key)
        if not self._providers:
            raise RuntimeError(
                "No LLM provider configured and mock mode is off. Set GROQ_API_KEY / GEMINI_API_KEY "
                "in your environment, or set MEDASSIST_MOCK=1 for offline dev."
            )

    def call(
        self,
        tier: Tier,
        messages: list[dict[str, str]],
        json_schema: dict | None = None,
        escalate_if_low_confidence: float | None = None,
    ) -> RoutedResponse:
        """
        Walks MODEL_TIER_CHAIN[tier] in order until one provider succeeds.
        If `escalate_if_low_confidence` is given and the winning response's
        self-reported confidence is below it, retries once more against the
        "large" tier chain before giving up — this is the confidence-aware
        upgrade path (High confidence -> stay put; low confidence -> bigger
        model + implicitly more likely to trigger RAG upstream).
        """
        chain = MODEL_TIER_CHAIN.get(tier)
        if not chain:
            raise ValueError(f"Unknown model tier: {tier!r}")

        last_error: Exception | None = None
        for i, (provider_name, model_id) in enumerate(chain):
            provider = self._providers.get(provider_name)
            if provider is None:
                continue
            try:
                resp = provider.generate(messages, model=model_id, json_schema=json_schema)
                fallback_applied = i > 0
                served_tier = tier

                if (
                    escalate_if_low_confidence is not None
                    and resp.raw_confidence is not None
                    and resp.raw_confidence < escalate_if_low_confidence
                    and tier != "large"
                ):
                    logger.info(
                        "confidence %.2f < %.2f threshold — escalating this call to 'large' tier",
                        resp.raw_confidence,
                        escalate_if_low_confidence,
                    )
                    escalated = self.call(tier="large", messages=messages, json_schema=json_schema)
                    escalated.fallback_applied = True
                    return escalated

                return RoutedResponse(
                    response=resp, tier_requested=tier, tier_served_by=served_tier, fallback_applied=fallback_applied
                )
            except ProviderError as e:
                logger.warning("provider %s/%s failed: %s", provider_name, model_id, e)
                last_error = e
                continue

        raise RuntimeError(f"All providers in tier {tier!r} failed. Last error: {last_error}")

"""
Unified provider interface. Every concrete provider implements the same
`generate(messages, json_schema=None) -> ProviderResponse` signature so
model_router.py can fall back across providers without any node caring which
one actually answered.

Import discipline matters here: `groq` and `google-generativeai` are only
imported *inside* the methods that need them (not at module load time), so
this file — and everything that imports it — stays importable in an
environment that only has stdlib + requests, which is what let this whole
project get unit-tested without installing the full stack. Production runs
still need `pip install groq google-generativeai` (see requirements.txt).
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from nodes.symptom_lexicon import symptom_phrases_in_text


@dataclass
class ProviderResponse:
    text: str
    parsed: dict[str, Any] | None
    provider: str
    model: str
    latency_ms: float
    raw_confidence: float | None = None  # only populated when we asked the model to self-report


class LLMProvider(Protocol):
    name: str

    def generate(
        self, messages: list[dict[str, str]], model: str, json_schema: dict[str, Any] | None = None
    ) -> ProviderResponse: ...


class ProviderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Groq — OpenAI-compatible chat completions endpoint.
# ---------------------------------------------------------------------------


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(
        self, messages: list[dict[str, str]], model: str, json_schema: dict[str, Any] | None = None
    ) -> ProviderResponse:
        try:
            from groq import Groq  # lazy import — see module docstring
        except ImportError as e:
            raise ProviderError("groq package not installed — `pip install groq`") from e

        client = Groq(api_key=self.api_key)
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        start = time.monotonic()
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — deliberately broad; model_router decides what to do
            raise ProviderError(f"Groq call failed for model={model}: {e}") from e
        latency_ms = (time.monotonic() - start) * 1000

        text = completion.choices[0].message.content or ""
        parsed = _try_parse_json(text)
        return ProviderResponse(text=text, parsed=parsed, provider=self.name, model=model, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Gemini — free-tier cross-provider fallback for the "large" tier.
# ---------------------------------------------------------------------------


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(
        self, messages: list[dict[str, str]], model: str, json_schema: dict[str, Any] | None = None
    ) -> ProviderResponse:
        try:
            import google.generativeai as genai  # lazy import
        except ImportError as e:
            raise ProviderError("google-generativeai package not installed — `pip install google-generativeai`") from e

        genai.configure(api_key=self.api_key)
        # Gemini's SDK wants a single system instruction + a turn list, not
        # OpenAI-style role dicts — flatten our messages into one prompt to
        # keep the provider interface uniform across Groq/Gemini/Mock.
        prompt = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
        gen_config = {"response_mime_type": "application/json"} if json_schema else {}

        start = time.monotonic()
        try:
            gmodel = genai.GenerativeModel(model)
            result = gmodel.generate_content(prompt, generation_config=gen_config)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"Gemini call failed for model={model}: {e}") from e
        latency_ms = (time.monotonic() - start) * 1000

        text = result.text or ""
        parsed = _try_parse_json(text)
        return ProviderResponse(text=text, parsed=parsed, provider=self.name, model=model, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Mock — deterministic, keyword-driven stand-in used for offline dev/tests
# and as the zero-setup default when no API key is configured. This is NOT a
# medical model of any kind — it exists purely so the graph's control flow
# (routing, loop termination, escalation, rule engines) can be exercised and
# unit-tested without a network call or a GPU.
# ---------------------------------------------------------------------------

_SYMPTOM_KEYWORDS: dict[str, list[str]] = {
    "eye_redness": ["red eye", "redness", "bloodshot"],
    "eye_pain": ["eye hurts", "eye pain", "pain in eye", "aching eye"],
    "severe_eye_pain": ["severe eye pain", "excruciating eye"],
    "eye_discharge": ["discharge", "gunky eye", "crusty eye"],
    "itching": ["itchy", "itching"],
    "eye_dryness": ["dry eye", "gritty"],
    "blurred_vision": ["blurry", "blurred vision", "everything looks blurry"],
    "vision_loss_sudden": ["sudden vision loss", "went blind", "lost vision suddenly"],
    "halos_around_lights": ["halos", "rings around lights"],
    "foreign_body_sensation": ["something in my eye", "foreign body"],
    "tearing": ["watery eye", "tearing up"],
    "light_sensitivity": ["light sensitivity", "photophobia", "bright light hurts"],
    "floaters": ["floaters", "spots in vision"],
    "flashes_of_light": ["flashes of light", "seeing flashes"],
    "curtain_shadow_vision": ["curtain over my eye", "shadow in vision"],
    "fever": ["fever", "temperature"],
    "cough": ["cough", "coughing"],
    "chest_pain": ["chest pain", "chest hurts"],
    "shortness_of_breath": ["shortness of breath", "can't breathe", "breathless"],
    "headache": ["headache", "head hurts"],
    "nausea": ["nausea", "nauseous", "feel sick"],
    "abdominal_pain": ["stomach pain", "abdominal pain", "belly hurts"],
    "diarrhea": ["diarrhea", "loose stools"],
    "runny_nose": ["runny nose", "stuffy nose"],
    "sore_throat": ["sore throat"],
    "facial_droop": ["face drooping", "facial droop"],
    "slurred_speech": ["slurred speech", "can't speak clearly"],
    "neurological_deficit": ["can't move my arm", "numbness one side", "weakness one side"],
    "throbbing_pain": ["throbbing", "pulsing", "pulsating", "pounding"],
    "visual_aura": ["visual aura", "zigzags", "shimmering lights", "blind spots"],
    "sudden_worst_headache_of_life": ["worst headache of my life", "worst headache", "thunderclap"],
}


class MockProvider:
    """Deterministic dev/offline provider — see module docstring."""

    name = "mock"

    def __init__(self, seed: int = 0):
        self._rand = random.Random(seed)

    def generate(
        self, messages: list[dict[str, str]], model: str, json_schema: dict[str, Any] | None = None
    ) -> ProviderResponse:
        start = time.monotonic()
        user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
        task = _infer_task(messages)

        if task == "extract_symptoms":
            found = [key for key, phrases in _SYMPTOM_KEYWORDS.items() if any(p in user_text for p in phrases)]
            found = list(dict.fromkeys([*found, *symptom_phrases_in_text(user_text, set(_SYMPTOM_KEYWORDS.keys()))]))
            parsed = {"raw_symptom_phrases": found}
        elif task == "normalize_symptoms":
            # In real use the normalizer maps free text to booleans via the
            # LLM; the mock just echoes back True for every phrase the
            # extractor already found by keyword match, which is enough to
            # drive the rest of the graph deterministically in tests.
            phrases = _extract_json_field(messages, "raw_symptom_phrases") or []
            parsed = {p: True for p in phrases}
        elif task == "follow_up_question":
            gap = _extract_json_field(messages, "missing_info_top") or "your symptoms"
            parsed = {"question": f"Can you tell me more about {gap.replace('_', ' ')}?"}
        elif task == "diagnosis":
            candidate = _extract_json_field(messages, "leading_candidate") or "unspecified_condition"
            parsed = {"predicted_disease": candidate, "confidence": 0.72}
        elif task == "safety_validation":
            parsed = {"contradictions": [], "hallucination_risk": "low", "unsafe_advice_detected": False}
        elif task == "response_synthesis":
            parsed = {"response": "Here is a summary of the assessment based on what you've told me."}
        else:
            parsed = {"note": "mock provider: unrecognized task, returning empty structured response"}

        latency_ms = (time.monotonic() - start) * 1000
        return ProviderResponse(
            text=json.dumps(parsed),
            parsed=parsed,
            provider=self.name,
            model=f"mock::{model}",
            latency_ms=latency_ms,
            raw_confidence=parsed.get("confidence") if isinstance(parsed, dict) else None,
        )


def _infer_task(messages: list[dict[str, str]]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    for marker in (
        "extract_symptoms",
        "normalize_symptoms",
        "follow_up_question",
        "diagnosis",
        "safety_validation",
        "response_synthesis",
    ):
        if marker in system:
            return marker
    return "unknown"


def _extract_json_field(messages: list[dict[str, str]], field_name: str) -> Any:
    """Nodes embed small JSON hints in the system prompt (see nodes/*.py) so the
    mock provider can behave sensibly without real reasoning."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    # List fields first: e.g. "raw_symptom_phrases": ["a", "b"]. This MUST be
    # tried before the scalar pattern below — the scalar pattern's character
    # class excludes `"` but not `[`, so against a list value it would match
    # just the leading "[" and return a truthy-but-wrong single-bracket
    # string instead of falling through to this branch.
    match_list = re.search(rf'"{field_name}"\s*:\s*(\[[^\]]*\])', system)
    if match_list:
        try:
            return json.loads(match_list.group(1))
        except json.JSONDecodeError:
            return None
    match = re.search(rf'"{field_name}"\s*:\s*"?([^",\n}}]+)"?', system)
    if match:
        return match.group(1).strip()
    return None


def _try_parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

"""
Minimal interactive CLI — run `python cli.py` to try a full session
end-to-end. Works with zero setup (falls back to the Mock provider if no
API keys are set — see config.load_settings) or against real free-tier
Groq/Gemini once you export GROQ_API_KEY / GEMINI_API_KEY.

This is the one script in the project that actually needs `langgraph`,
`groq`/`google-generativeai` (unless running in mock mode), installed.
"""
from __future__ import annotations

import sys
import uuid

from config import load_settings
from graph import build_graph
from observability import MetricsSink, SessionMetrics


def main() -> None:
    settings = load_settings()
    if settings.use_mock_provider:
        print("[medassist] No API keys found — running in MOCK mode (deterministic, offline, not a real medical model).\n")

    app = build_graph(settings=settings)
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    metrics_sink = MetricsSink()
    metrics = SessionMetrics(session_id=session_id)

    print("Describe your symptoms (or 'quit' to exit):")
    raw_query = input("> ").strip()
    if raw_query.lower() in ("quit", "exit"):
        return

    initial_state = {
        "session_id": session_id,
        "raw_query": raw_query,
        "extracted_symptoms": {},
        "follow_up_history": [],
        "follow_up_count": 0,
        "confidence_history": [],
        "model_tier_used": [],
        "fallback_applied": False,
        "safety_flags": [],
    }

    result = app.invoke(initial_state, config=config)

    # Drain any interrupts (follow-up questions) until the graph reaches END.
    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        print(f"\n{interrupt_payload['question']}")
        answer = input("> ").strip()
        if answer.lower() in ("quit", "exit"):
            print("Session ended early.")
            sys.exit(0)
        from langgraph.types import Command

        result = app.invoke(Command(resume=answer), config=config)

    print("\n--- Assessment ---")
    print(result.get("final_response", "(no response generated)"))
    if result.get("escalated"):
        print(f"[escalated — reason: {result.get('escalation_reason')}]")
    else:
        print(f"[predicted: {result.get('predicted_disease')} | calibrated confidence: {result.get('calibrated_confidence')}]")
    print(f"[emergency level: {result.get('emergency_level')} | department: {result.get('recommended_department')}]")

    metrics.finalize_from_state(result)
    metrics_sink.record(metrics)
    print(f"\n[telemetry] {metrics.as_dict()}")


if __name__ == "__main__":
    main()

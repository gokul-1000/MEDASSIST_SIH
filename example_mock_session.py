"""
A scripted, non-interactive example — no typing required, no API keys
needed. Run it with:

    python example_mock_session.py

This is deliberately NOT the same thing as the offline tests in tests/. The
tests chain node functions directly in Python, bypassing LangGraph entirely
(see tests/test_end_to_end_offline.py's docstring for why). This script
calls the REAL build_graph() and the REAL app.invoke()/Command(resume=...)
— it's the actual LangGraph engine running the actual StateGraph, just with
MockProvider standing in for the LLM so no API key is needed.

If this script runs to completion and prints a final assessment, that's the
first real confirmation that graph.py's wiring works end-to-end through
LangGraph itself, on your machine — which is exactly the thing I could not
verify in the sandbox this project was built in (no network there, so
`langgraph` couldn't be installed to test against).

Two scripted cases are included:
  1. A straightforward eye complaint — walks through 1-2 follow-up
     questions, answered automatically from a canned list, then reaches a
     diagnosis.
  2. A hard-red-flag case (chest pain + can't breathe) — should reach a
     result with ZERO follow-up questions, because of the short-circuit in
     graph.py's `_route_after_confidence_engine` (see that module's
     docstring, "correction #3").
"""
from __future__ import annotations

import uuid

from config import Settings
from graph import build_graph


def run_scripted_session(app, raw_query: str, canned_answers: list[str], label: str) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"User: {raw_query}")

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    answers = iter(canned_answers)

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

    # Drain interrupts automatically using the canned answers, instead of
    # cli.py's input() prompts — this is what makes the script non-interactive.
    while "__interrupt__" in result:
        question = result["__interrupt__"][0].value["question"]
        try:
            answer = next(answers)
        except StopIteration:
            answer = "no"  # run out of canned answers — just decline further follow-ups
        print(f"  Follow-up asked: {question}")
        print(f"  Answered: {answer}")

        from langgraph.types import Command

        result = app.invoke(Command(resume=answer), config=config)

    print(f"\nFollow-up questions actually asked: {result.get('follow_up_count', 0)}")
    print(f"Predicted disease: {result.get('predicted_disease')}")
    print(f"Emergency level: {result.get('emergency_level')}  (1=routine, 2=prompt, 3=immediate)")
    print(f"Escalated to human: {result.get('escalated')}")
    print(f"Final response: {result.get('final_response')}")


def main() -> None:
    settings = Settings(use_mock_provider=True)  # force mock mode regardless of any keys in your environment
    print("Building graph in MOCK mode (deterministic keyword-matching, not a real medical model)...")
    app = build_graph(settings=settings)
    print("Graph built and compiled successfully.")

    run_scripted_session(
        app,
        raw_query="my eye is red and itchy",
        canned_answers=["no discharge", "no vision loss"],
        label="Case 1: straightforward eye complaint (expects 1-2 follow-ups)",
    )

    run_scripted_session(
        app,
        raw_query="I have chest pain and I'm having shortness of breath",
        canned_answers=["this should never be asked"],
        label="Case 2: hard red-flag pattern (expects ZERO follow-ups, emergency_level 3)",
    )


if __name__ == "__main__":
    main()

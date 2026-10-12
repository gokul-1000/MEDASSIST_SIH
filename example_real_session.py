"""
Tests the REAL Groq/Gemini-backed graph against known expected outcomes —
this is what actually tells you "is it giving the right answer," as opposed
to example_mock_session.py, which only proves the wiring runs.

    python example_real_session.py

Needs your .env to have GROQ_API_KEY set (see README "Real setup"). Will
make real API calls — a handful of calls against Groq's free tier, so this
costs you nothing, but it does need network.

WHY AN "ANSWER BANK" INSTEAD OF A CANNED ANSWER LIST:
With the real LLM, the follow-up QUESTION TEXT is generated live and won't
match any fixed script — but WHICH SYMPTOM each question is targeting
(`asked_for`) is still fully deterministic (the Planner computes that with
no LLM call at all, see nodes/planner.py). So instead of answering "the
2nd follow-up" a fixed way, each test case supplies a dict of
{symptom_name: answer_text} and this script answers whatever's actually
asked, however many times, in whatever order the model picks.

HOW TO READ THE OUTPUT:
Real LLM output isn't deterministic — don't expect a byte-perfect match
every run. What you're checking for:
  1. Does the PREDICTED DISEASE match (or reasonably overlap with) EXPECTED?
  2. For the red-flag case: is EMERGENCY LEVEL 3 and FOLLOW-UPS still 0?
     (That part IS deterministic — it's a rule-engine short-circuit, not an
     LLM judgment call — so it should match every single time. If it
     doesn't, that's a real bug, not "the model was having an off day.")
  3. For the thin-margin case: does it actually ASK about the
     differentiating symptom (nausea/light-sensitivity for migraine vs.
     tension headache) instead of guessing immediately?
A mismatch on (1) sometimes just means the KB entry or prompt wording needs
tightening — that's normal iteration, not a crisis. A mismatch on (2) means
stop and look, because that's the safety-critical path.
"""
from __future__ import annotations

import uuid

from config import load_settings
from graph import build_graph

TEST_CASES = [
    {
        "label": "Easy case — should reach a confident, non-escalated diagnosis",
        "raw_query": "My eye is really red and itchy, and there's some discharge too",
        "expected_disease": "conjunctivitis",
        "answer_bank": {
            "vision_loss_sudden": "no, my vision is fine",
            "severe_eye_pain": "no, it's not that painful, just irritated",
            "eye_pain": "just mild irritation, not real pain",
            "tearing": "a little bit, yes",
        },
    },
    {
        "label": "Thin-margin case — watch whether it asks the RIGHT differentiating question",
        "raw_query": "I have a headache",
        "expected_disease": "migraine",  # only correct if it asks about nausea/light-sensitivity/aura and gets yes
        "answer_bank": {
            "nausea": "yes, I feel a bit nauseous",
            "light_sensitivity": "yes, bright light makes it worse",
            "visual_aura": "yes, I saw some shimmering lights before it started",
            "neck_tightness": "not really",
            "stress": "not particularly stressed",
        },
    },
    {
        "label": "Hard red-flag case — the safety-critical one. MUST show 0 follow-ups and emergency_level 3.",
        "raw_query": "I have chest pain and I'm having shortness of breath",
        "expected_disease": None,  # not the point of this case — see WHY AN ANSWER BANK note above
        "answer_bank": {},
        "expect_zero_follow_ups": True,
        "expect_emergency_level": 3,
    },
    {
        "label": "Rare/red-flag eye case — should trigger RAG and route to Ophthalmology (Emergency)",
        "raw_query": "I suddenly can't see well out of one eye, and I'm seeing floaters and flashes of light",
        "expected_disease": "retinal_detachment",
        "answer_bank": {
            "curtain_shadow_vision": "yes, it's like a curtain coming down over part of my vision",
            "eye_pain": "no pain at all",
        },
    },
]


def run_case(app, case: dict) -> None:
    print(f"\n{'=' * 78}\n{case['label']}\n{'=' * 78}")
    print(f"User: {case['raw_query']}")

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "session_id": session_id,
        "raw_query": case["raw_query"],
        "extracted_symptoms": {},
        "follow_up_history": [],
        "follow_up_count": 0,
        "confidence_history": [],
        "model_tier_used": [],
        "fallback_applied": False,
        "safety_flags": [],
    }

    result = app.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        question, asked_for = payload["question"], payload["asked_for"]
        answer = case["answer_bank"].get(asked_for, "not that I've noticed")
        print(f"  Asked about '{asked_for}': {question}")
        print(f"  Answered: {answer}")

        from langgraph.types import Command

        result = app.invoke(Command(resume=answer), config=config)

    follow_ups = result.get("follow_up_count", 0)
    predicted = result.get("predicted_disease")
    emergency_level = result.get("emergency_level")
    escalated = result.get("escalated")

    print(f"\n  Follow-ups asked: {follow_ups}")
    print(f"  Predicted: {predicted}  (expected: {case.get('expected_disease', 'n/a')})")
    print(f"  Emergency level: {emergency_level}  |  Escalated: {escalated}")
    print(f"  Final response: {result.get('final_response')}")

    if case.get("expect_zero_follow_ups") is not None:
        ok = follow_ups == 0
        print(f"  [{'PASS' if ok else 'FAIL'}] expected zero follow-ups")
    if case.get("expect_emergency_level") is not None:
        ok = emergency_level == case["expect_emergency_level"]
        print(f"  [{'PASS' if ok else 'FAIL'}] expected emergency_level={case['expect_emergency_level']}")
    if case.get("expected_disease") is not None:
        match = predicted == case["expected_disease"]
        print(f"  [{'MATCH' if match else 'DIFFERENT — inspect, not necessarily wrong'}] disease prediction")


def main() -> None:
    settings = load_settings()
    if settings.use_mock_provider:
        print(
            "!! Still running in MOCK mode — your .env isn't being picked up. "
            "Check that medassist/.env exists and has GROQ_API_KEY set, then rerun."
        )
        return

    print("Real API mode confirmed (GROQ_API_KEY found). Building graph...")
    app = build_graph(settings=settings)
    print("Graph built. Running scripted test cases against the real model...\n")

    for case in TEST_CASES:
        run_case(app, case)

    print(f"\n{'=' * 78}\nDone. Re-read the safety-critical case (case 3) output above first.")


if __name__ == "__main__":
    main()

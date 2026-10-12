"""
Adaptive Follow-Up (§3.4). Split into two graph nodes on purpose, for a
reason that's easy to miss and expensive to get wrong:

LangGraph re-executes a node's function body from the top every time it
resumes from an interrupt() inside that same node — only the interrupt()
call itself is "replayed" (it returns the resume value instead of pausing
again). Anything ELSE in that node — including an LLM call — re-runs on
every resume. Put the question-generation LLM call in the same node as
interrupt() and you silently pay for (and could get a *different*) question
on every resume.

So:
  - `make_follow_up_generate_node` does the one LLM call (small tier),
    writes the question into state, and kicks off the async RAG prefetch
    for whatever's currently the leading candidate.
  - `follow_up_interrupt_node` does nothing but call interrupt() and, once
    resumed, record the QA pair. Cheap enough to safely re-run.
"""
from __future__ import annotations

from typing import Any

from config import Settings
from models.model_router import ModelRouter
from nodes.question_bank import question_for_symptom
from rag.rag_pipeline import prefetcher
from state import State

_QUESTION_SYSTEM_TEMPLATE = """You are generating ONE targeted follow-up question for a medical symptom assessment tool.
task=follow_up_question
"missing_info_top": "{top_gap}"
"domain": "{domain}"
Ask a single, plain-language yes/no or short-answer question that would help
determine whether the user has this symptom. Keep the body location implied by
the symptom and domain. Do not ask about anything else.
Respond with strict JSON: {{"question": "..."}}"""


def make_follow_up_generate_node(model_router: ModelRouter, settings: Settings):
    def follow_up_generate_node(state: State) -> dict[str, Any]:
        missing_info = state.get("missing_info", [])
        top_gap = missing_info[0] if missing_info else "how you're feeling overall"

        question = question_for_symptom(top_gap, extracted_symptoms=state.get("extracted_symptoms", {}))
        if question is None:
            resp = model_router.call(
                tier="small",
                messages=[
                    {
                        "role": "system",
                        "content": _QUESTION_SYSTEM_TEMPLATE.format(
                            top_gap=top_gap,
                            domain=state.get("domain", "general"),
                        ),
                    },
                    {"role": "user", "content": "Generate the question."},
                ],
            )
            question = (resp.response.parsed or {}).get("question") or (
                f"Can you tell me more about {top_gap.replace('_', ' ')}?"
            )

        candidates = state.get("candidate_diseases", [])
        if candidates:
            prefetcher.kick_off(
                session_id=state["session_id"],
                domain=state["domain"],
                leading_candidate=candidates[0]["name"],
                mock=settings.use_mock_provider,
            )

        return {"pending_question": question, "pending_asked_for": top_gap}

    return follow_up_generate_node


def follow_up_interrupt_node(state: State) -> dict[str, Any]:
    """No LLM call — safe to re-execute on every resume."""
    from langgraph.types import interrupt  # lazy import, see module docstring in llm_provider.py for rationale

    question = state.get("pending_question", "Can you tell me more about your symptoms?")
    asked_for = state.get("pending_asked_for", "symptoms")

    answer = interrupt({"question": question, "asked_for": asked_for})

    qa = {"question": question, "answer": str(answer), "asked_for": asked_for}
    return {
        "follow_up_history": [qa],
        "follow_up_count": state.get("follow_up_count", 0) + 1,
    }

"""Thin adapter around the real LangGraph triage engine (graph.py). Everything else in the UI consumes the `summary` dict."""
from __future__ import annotations
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_graph():
    from langgraph.checkpoint.memory import MemorySaver
    from config import load_settings
    from graph import build_graph
    settings = load_settings()
    return build_graph(settings=settings, checkpointer=MemorySaver()), settings

def engine_mode() -> str:
    try: return "Mock engine (offline demo)" if get_graph()[1].use_mock_provider else "Live LLM engine"
    except Exception: return "Unavailable"

def initial_state(tid, query, age=None, gender=None) -> dict:
    s = dict(session_id=tid, raw_query=query, extracted_symptoms={}, follow_up_history=[], follow_up_count=0, confidence_history=[], model_tier_used=[], fallback_applied=False, safety_flags=[])
    if age: s["age"] = int(age)
    if gender: s["gender"] = {"M": "male", "F": "female"}.get(gender, gender)
    return s

def invoke(payload, tid):
    app, _ = get_graph()
    return app.invoke(payload, config={"configurable": {"thread_id": tid}})

def resume(answer):
    from langgraph.types import Command
    return Command(resume=answer)

def summarize(result: dict, complaint: str, age=None, gender=None) -> dict:
    """Graph result → the compact clinical summary stored on a visit and shown to the doctor."""
    sym = result.get("extracted_symptoms") or {}
    diff = [dict(name=d.get("disease_name") or d.get("name"), score=d.get("score", 0), matched=d.get("matched_symptoms", []), missing=d.get("missing_required_symptoms", [])) for d in (result.get("differential_diagnosis") or [])]
    return dict(complaint=complaint, domain=result.get("domain", "general"), age=result.get("age", age), gender=result.get("gender", gender),
                symptoms_present=[k for k, v in sym.items() if v is True], symptoms_absent=[k for k, v in sym.items() if v is False],
                attributes={k: v for k, v in sym.items() if not isinstance(v, bool) and v not in (None, "")},
                followups=[dict(q=h.get("question"), a=h.get("answer")) for h in (result.get("follow_up_history") or [])],
                predicted=result.get("predicted_disease"), differential=diff, confidence=result.get("calibrated_confidence"),
                level=result.get("emergency_level") or 1, department_raw=result.get("recommended_department") or "General Medicine",
                escalated=bool(result.get("escalated")), escalation_reason=result.get("escalation_reason"), flags=result.get("safety_flags") or [],
                sources=sorted({f"{s.get('disease','').replace('_',' ')} · {s.get('guideline_version','')}" for s in (result.get("rag_sources") or [])}),
                final_response=result.get("final_response"))

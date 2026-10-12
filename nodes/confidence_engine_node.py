"""
Confidence Engine (§3.5) as a graph node. Wraps the pure function in
confidence.py so it can sit in the LangGraph wiring; the actual logic has
zero LangGraph/network dependency and is unit-tested directly in
tests/test_confidence.py.
"""
from __future__ import annotations

from typing import Any

from confidence import compute_loop_confidence
from state import State


def confidence_engine_node(state: State) -> dict[str, Any]:
    result = compute_loop_confidence(
        candidates=state.get("candidate_diseases", []),
        threshold=state.get("confidence_threshold", 0.75),
        follow_up_count=state.get("follow_up_count", 0),
        max_follow_ups=state.get("max_follow_ups", 3),
    )
    return {
        "confidence_score": result.score,
        "confidence_history": [result.score],
    }

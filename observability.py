"""
Observability (§6). Tracked per-session and aggregated. This is a plain
in-process tracker (dict + list accumulation) rather than a real metrics
backend — swap `MetricsSink.emit` for a StatsD/Prometheus/OpenTelemetry call
when this leaves prototype stage. Kept dependency-free so it's usable from
tests without standing up any infrastructure.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from state import State


@dataclass
class SessionMetrics:
    session_id: str
    raw_confidence_scores: list[float] = field(default_factory=list)
    calibrated_confidence: float | None = None
    follow_up_count: int = 0
    rag_invoked: bool = False
    escalated: bool = False
    model_tiers_used: list[str] = field(default_factory=list)
    fallback_applied: bool = False
    node_latencies_ms: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    def finalize_from_state(self, state: State) -> None:
        self.raw_confidence_scores = state.get("confidence_history", [])
        self.calibrated_confidence = state.get("calibrated_confidence")
        self.follow_up_count = state.get("follow_up_count", 0)
        self.rag_invoked = bool(state.get("rag_required"))
        self.escalated = bool(state.get("escalated"))
        self.model_tiers_used = state.get("model_tier_used", [])
        self.fallback_applied = bool(state.get("fallback_applied"))
        self.node_latencies_ms = state.get("node_latencies_ms", {})
        self.ended_at = time.time()

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "avg_raw_confidence": _safe_mean(self.raw_confidence_scores),
            "calibrated_confidence": self.calibrated_confidence,
            "follow_up_count": self.follow_up_count,
            "rag_invoked": self.rag_invoked,
            "escalated": self.escalated,
            "model_tiers_used": self.model_tiers_used,
            "fallback_applied": self.fallback_applied,
            "duration_s": round((self.ended_at or time.time()) - self.started_at, 3),
        }


class MetricsSink:
    """Accumulates SessionMetrics and can compute the aggregate table from §6."""

    def __init__(self, log_path: str | None = None):
        self._sessions: list[SessionMetrics] = []
        self._log_path = Path(log_path) if log_path else None

    def record(self, metrics: SessionMetrics) -> None:
        self._sessions.append(metrics)
        if self._log_path:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics.as_dict()) + "\n")

    def aggregate(self) -> dict[str, Any]:
        if not self._sessions:
            return {}
        n = len(self._sessions)
        return {
            "sessions": n,
            "avg_raw_confidence": _safe_mean(
                [m.as_dict()["avg_raw_confidence"] for m in self._sessions if m.raw_confidence_scores]
            ),
            "avg_follow_up_count": _safe_mean([m.follow_up_count for m in self._sessions]),
            "rag_invocation_rate": sum(m.rag_invoked for m in self._sessions) / n,
            "escalation_rate": sum(m.escalated for m in self._sessions) / n,
            "fallback_trigger_rate": sum(m.fallback_applied for m in self._sessions) / n,
        }


def _safe_mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None

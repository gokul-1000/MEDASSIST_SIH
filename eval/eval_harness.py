"""
Evaluation mode (your requested refinement). A parallel pipeline that
compares predictions against a labeled dataset and logs precision, recall,
F1, and calibration error — completing the lifecycle alongside the
operational metrics in observability.py, which only tell you the system is
running, not whether it's *right*.

Two ways to use this:

1. `run_offline_eval()` — zero dependencies, zero API keys. Runs the
   deterministic candidate-ranking heuristic (confidence.rank_candidates)
   against knowledge_base/*_kb.json as the "prediction," which is what lets
   this whole harness (and its tests) run without a network call. This is
   evaluating the KB-matching heuristic itself, not the LLM — genuinely
   useful (it tells you if your KB's required/supporting symptom lists are
   well-formed), but it is NOT the same thing as evaluating end-to-end
   diagnostic accuracy.

2. `evaluate_predictions()` — the general-purpose metric function. Feed it
   real (true_label, predicted_label, confidence) triples from actual graph
   runs (e.g. by logging `predicted_disease` + `calibrated_confidence` next
   to the eval set's `true_disease` for each case) once you have API keys
   configured, and it computes the same metrics against real model output.
   This is the one to wire into CI once you're past prototype stage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from confidence import rank_candidates
from knowledge_base.kb_loader import load_kb

_EVAL_DIR = Path(__file__).parent


@dataclass
class PredictionRecord:
    true_label: str
    predicted_label: str
    confidence: float  # 0-1, the model's (or heuristic's) confidence in predicted_label


@dataclass
class EvalReport:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: dict[str, dict[str, float]]
    calibration_error: float  # Expected Calibration Error (ECE), lower is better
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "calibration_error": self.calibration_error,
            "per_class": self.per_class,
        }


def evaluate_predictions(records: list[PredictionRecord], ece_bins: int = 10) -> EvalReport:
    if not records:
        raise ValueError("Need at least one prediction record to evaluate.")

    labels = sorted({r.true_label for r in records} | {r.predicted_label for r in records})
    per_class: dict[str, dict[str, float]] = {}

    for label in labels:
        tp = sum(1 for r in records if r.predicted_label == label and r.true_label == label)
        fp = sum(1 for r in records if r.predicted_label == label and r.true_label != label)
        fn = sum(1 for r in records if r.predicted_label != label and r.true_label == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    accuracy = sum(1 for r in records if r.predicted_label == r.true_label) / len(records)
    precision_macro = sum(c["precision"] for c in per_class.values()) / len(per_class)
    recall_macro = sum(c["recall"] for c in per_class.values()) / len(per_class)
    f1_macro = sum(c["f1"] for c in per_class.values()) / len(per_class)

    ece = _expected_calibration_error(records, n_bins=ece_bins)

    return EvalReport(
        accuracy=round(accuracy, 4),
        precision_macro=round(precision_macro, 4),
        recall_macro=round(recall_macro, 4),
        f1_macro=round(f1_macro, 4),
        per_class=per_class,
        calibration_error=round(ece, 4),
        n=len(records),
    )


def _expected_calibration_error(records: list[PredictionRecord], n_bins: int = 10) -> float:
    """
    ECE: bin predictions by confidence, compare each bin's average confidence
    to its actual accuracy, weight by bin size. 0 = perfectly calibrated.
    """
    bins: list[list[PredictionRecord]] = [[] for _ in range(n_bins)]
    for r in records:
        idx = min(int(r.confidence * n_bins), n_bins - 1)
        bins[idx].append(r)

    total = len(records)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_confidence = sum(r.confidence for r in bucket) / len(bucket)
        accuracy = sum(1 for r in bucket if r.predicted_label == r.true_label) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_confidence - accuracy)
    return ece


def load_eval_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else _EVAL_DIR / "labeled_eval_set.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def run_offline_eval(cases: list[dict[str, Any]] | None = None) -> EvalReport:
    """See module docstring option 1 — evaluates the deterministic
    KB-matching heuristic, not the LLM. Zero network calls."""
    cases = cases if cases is not None else load_eval_cases()

    records = []
    for case in cases:
        kb = load_kb(case["domain"])
        ranked = rank_candidates(case["symptoms"], kb.diseases)
        top = ranked[0] if ranked else {"name": "unknown", "score": 0.0}
        records.append(PredictionRecord(true_label=case["true_disease"], predicted_label=top["name"], confidence=top["score"]))

    return evaluate_predictions(records)

"""
Loads the versioned per-domain knowledge base and hands back both the disease
table and the metadata block (guideline_version, publication_date,
retrieval_timestamp, document_source) — the refinement you asked for:
"version your medical knowledge base."

Every place that touches KB content (Planner, RAG, rule engines) should pull
metadata through here rather than reading the JSON directly, so a future
"which guideline version produced this answer?" audit query has exactly one
place to look.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_KB_DIR = Path(__file__).parent


@dataclass(frozen=True)
class KBMetadata:
    domain: str
    guideline_version: str
    guideline_source: str
    publication_date: str
    retrieval_timestamp: str
    document_source: str
    note: str = ""


@dataclass(frozen=True)
class KnowledgeBase:
    metadata: KBMetadata
    diseases: list[dict[str, Any]]
    symptom_vocabulary: list[str]

    def disease_by_name(self, name: str) -> dict[str, Any] | None:
        return next((d for d in self.diseases if d["name"] == name), None)


@lru_cache(maxsize=None)
def load_kb(domain: str) -> KnowledgeBase:
    """domain: 'eye' or 'general'. Cached — the file is only read once per process."""
    path = _KB_DIR / f"{domain}_kb.json"
    if not path.exists():
        raise ValueError(f"No knowledge base file for domain={domain!r} (looked at {path})")
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = KBMetadata(**raw["metadata"])
    return KnowledgeBase(metadata=meta, diseases=raw["diseases"], symptom_vocabulary=raw["symptom_vocabulary"])


def all_domains() -> list[str]:
    return sorted(p.stem.replace("_kb", "") for p in _KB_DIR.glob("*_kb.json"))

"""
Builds the compiled LangGraph StateGraph. This is the one module in the
project that genuinely requires `langgraph` installed to actually RUN —
everything it wires together (nodes/*, confidence.py, rules/*) is
independently unit-tested without it, per the project's "business logic vs.
orchestration" split (see README "How this was tested without a GPU or
LangGraph installed").

The graph's topology (which nodes exist, which edges connect them) is
expressed as plain data — `NODE_FACTORY`, `EDGES`, `CONDITIONAL_EDGES` below
— rather than being buried inside a sequence of `builder.add_*` calls. Two
things fall out of that for free:

  1. `validate_wiring()` can statically check the topology (every edge target
     is a real node, every node is reachable from START, END is reachable)
     using nothing but stdlib — no LangGraph import required. This is run as
     a test (`tests/test_graph_wiring.py`) in an environment that can't
     install LangGraph, and it's exactly the kind of check that catches a
     typo'd node name before you ever get a runtime traceback.
  2. `build_graph()` becomes "walk the topology and call the real
     `StateGraph` methods" instead of being the single source of truth
     itself — so the thing being tested and the thing being run are
     guaranteed to be the same data.

Three deliberate corrections vs. the architecture doc's literal mermaid
diagram (documented here rather than silently — a "recheck" is supposed to
surface exactly this kind of thing):

1. DOMAIN SUBGRAPHS ARE NOT PHYSICALLY DUPLICATED. The doc's diagram shows
   parallel PLN1/PLN2, EX1/EX2, etc. for eye vs. general. Functionally,
   every node already reads `state["domain"]` and picks the right KB/prompt
   at call time (see nodes/planner.py, knowledge_base/kb_loader.py) — so
   duplicating the graph nodes themselves would be dead weight with no
   behavioral difference. One graph, domain-parameterized nodes.

2. THE ESCALATION FAN-OUT ORDER IS CHANGED TO AVOID A STATE-WRITE CONFLICT.
   The doc's diagram has Safety Validator fan out to Calibration, Emergency
   Classification, AND Department Recommendation unconditionally, with a
   separate edge to Human Escalation. But `recommended_department` would
   then be written by BOTH department_recommendation_node (normal path) and
   escalation_node (escalation path) if they ever land in the same
   superstep — LangGraph raises on unresolvable concurrent writes to a
   non-reducer key. Fix: Calibration and Emergency Classification (which
   don't need each other) fan out first and join at a gate; the escalation
   decision is evaluated there, and ONLY THEN does the graph branch into the
   (now mutually exclusive) Department Recommendation vs. Escalation paths.
   Bonus: Human Escalation can now surface emergency_level in its message
   ("this also looks urgent") — the original ordering couldn't support that.

3. HARD RED FLAGS NOW ACTUALLY SHORT-CIRCUIT THE FOLLOW-UP LOOP. The doc's
   §9 says a hard red-flag combination (chest pain + breathlessness, stroke
   pattern, etc.) should bypass the rest of the loop. The first version of
   this file didn't actually implement that — `check_red_flags()` was only
   ever called inside Safety Validator, which sits AFTER the full bounded
   follow-up loop already ran to completion. A person typing "chest pain and
   I can't breathe" would have gone through up to 3 more follow-up questions
   before anything treated it as urgent. Fixed: `_route_after_confidence_engine`
   now checks `check_red_flags()` first and skips straight to the
   RAG/Diagnosis path (still producing a real prediction + department, just
   without spending the loop's question budget) whenever a hard combination
   is already known to be present.
"""
from __future__ import annotations

from config import Settings, load_settings
from models.model_router import ModelRouter
from nodes.calibration_node import calibration_node
from nodes.confidence_engine_node import confidence_engine_node
from nodes.diagnosis import (
    conditional_rag_decision_node,
    direct_reasoning_node,
    make_diagnosis_node,
    make_rag_retrieval_node,
)
from nodes.emergency_department import department_recommendation_node, emergency_classification_node
from nodes.escalation import escalation_node
from nodes.extraction_normalization import make_extractor_node, make_normalizer_node
from nodes.follow_up import follow_up_interrupt_node, make_follow_up_generate_node
from nodes.orchestrator import evaluate_escalation_triggers, is_state_malformed, should_continue_follow_up
from nodes.planner import planner_node
from nodes.router import make_router_node
from nodes.safety_validator import make_safety_validator_node
from nodes.synthesis import make_synthesis_node
from rules.rule_engine import check_red_flags
from state import State

# Sentinels for the topology data below — resolved to LangGraph's real
# START/END only inside build_graph(), so this module (and validate_wiring)
# never needs langgraph imported to be checked.
_START = "__graph_start__"
_END = "__graph_end__"


def _route_after_router(state: State) -> str:
    return "escalation" if is_state_malformed(state) else "extractor"


def _route_after_confidence_engine(state: State) -> str:
    # Correction #3 — see module docstring. A hard red-flag combination
    # bypasses the rest of the bounded follow-up loop entirely.
    if check_red_flags(state.get("extracted_symptoms", {})).triggered:
        return "conditional_rag_decision"

    continue_loop = should_continue_follow_up(
        confidence_score=state.get("confidence_score", 0.0),
        threshold=state.get("confidence_threshold", 0.75),
        follow_up_count=state.get("follow_up_count", 0),
        max_follow_ups=state.get("max_follow_ups", 3),
    )
    return "follow_up_generate" if continue_loop else "conditional_rag_decision"


def _route_after_conditional_rag_decision(state: State) -> str:
    return "rag_retrieval" if state.get("rag_required") else "direct_reasoning"


def _route_after_escalation_gate(state: State) -> str:
    return "escalation" if evaluate_escalation_triggers(state) else "department_recommendation"


def _escalation_gate_node(state: State) -> dict:
    """No-op join point — see module docstring, correction #2. Exists purely
    so LangGraph waits for both Calibration and Emergency Classification
    before the escalation decision is evaluated."""
    return {}


# ---------------------------------------------------------------------------
# Topology as data. `deps` is a tiny namespace carrying whatever a node
# factory needs (currently just model_router and settings) — kept generic so
# adding a new dependency later doesn't mean touching every factory lambda.
# ---------------------------------------------------------------------------


class _Deps:
    def __init__(self, model_router: ModelRouter, settings: Settings):
        self.model_router = model_router
        self.settings = settings


NODE_FACTORY: dict[str, "callable"] = {
    "router": lambda d: make_router_node(d.model_router),
    "planner": lambda d: planner_node,
    "extractor": lambda d: make_extractor_node(d.model_router),
    "normalizer": lambda d: make_normalizer_node(d.model_router),
    "confidence_engine": lambda d: confidence_engine_node,
    "follow_up_generate": lambda d: make_follow_up_generate_node(d.model_router, d.settings),
    "follow_up_interrupt": lambda d: follow_up_interrupt_node,
    "conditional_rag_decision": lambda d: conditional_rag_decision_node,
    "rag_retrieval": lambda d: make_rag_retrieval_node(d.settings),
    "direct_reasoning": lambda d: direct_reasoning_node,
    "diagnosis": lambda d: make_diagnosis_node(d.model_router),
    "safety_validator": lambda d: make_safety_validator_node(d.model_router),
    "calibration": lambda d: calibration_node,
    "emergency_classification": lambda d: emergency_classification_node,
    "escalation_gate": lambda d: _escalation_gate_node,
    "department_recommendation": lambda d: department_recommendation_node,
    "escalation": lambda d: escalation_node,
    "synthesis": lambda d: make_synthesis_node(d.model_router),
}

# Plain (source, target) edges — unconditional.
EDGES: list[tuple[str, str]] = [
    (_START, "router"),
    ("extractor", "normalizer"),
    ("normalizer", "planner"),
    ("planner", "confidence_engine"),
    ("follow_up_generate", "follow_up_interrupt"),
    ("follow_up_interrupt", "extractor"),  # loop back, per §3.4 / mermaid diagram
    ("rag_retrieval", "diagnosis"),
    ("direct_reasoning", "diagnosis"),
    ("diagnosis", "safety_validator"),
    # Parallel fan-out #1 (correction #2)
    ("safety_validator", "calibration"),
    ("safety_validator", "emergency_classification"),
    ("calibration", "escalation_gate"),
    ("emergency_classification", "escalation_gate"),
    ("department_recommendation", "synthesis"),
    ("escalation", "synthesis"),
    ("synthesis", _END),
]

# (source, routing_fn, {label: target}) — routing_fn's return value selects
# which target to use; every value in the mapping must be a valid node (or _END).
CONDITIONAL_EDGES: list[tuple[str, "callable", dict[str, str]]] = [
    ("router", _route_after_router, {"extractor": "extractor", "escalation": "escalation"}),
    (
        "confidence_engine",
        _route_after_confidence_engine,
        {"follow_up_generate": "follow_up_generate", "conditional_rag_decision": "conditional_rag_decision"},
    ),
    (
        "conditional_rag_decision",
        _route_after_conditional_rag_decision,
        {"rag_retrieval": "rag_retrieval", "direct_reasoning": "direct_reasoning"},
    ),
    (
        "escalation_gate",
        _route_after_escalation_gate,
        {"escalation": "escalation", "department_recommendation": "department_recommendation"},
    ),
]


def validate_wiring() -> list[str]:
    """
    Pure-stdlib static check of the topology above: every edge/conditional-edge
    endpoint must be a real registered node (or START/END), and every node
    must be reachable from START. Returns a list of problem descriptions —
    empty means the topology is internally consistent. No LangGraph import,
    so this runs in any environment, including one that can't install it.
    """
    problems: list[str] = []
    node_names = set(NODE_FACTORY.keys())

    def _check_endpoint(label: str, value: str) -> None:
        if value not in (_START, _END) and value not in node_names:
            problems.append(f"{label} {value!r} is not a registered node (and not START/END)")

    adjacency: dict[str, list[str]] = {}

    for src, dst in EDGES:
        _check_endpoint("Edge source", src)
        _check_endpoint("Edge target", dst)
        adjacency.setdefault(src, []).append(dst)

    for src, _fn, mapping in CONDITIONAL_EDGES:
        _check_endpoint("Conditional edge source", src)
        if src not in node_names:
            continue
        if not mapping:
            problems.append(f"Conditional edge from {src!r} has an empty routing map")
        for label, target in mapping.items():
            _check_endpoint(f"Conditional edge {src!r} -> {label!r} target", target)
            adjacency.setdefault(src, []).append(target)

    # Every node must have at least one incoming edge (or be START itself) —
    # otherwise it's dead code that can never execute.
    all_targets = {dst for _src, dst in EDGES}
    for _src, _fn, mapping in CONDITIONAL_EDGES:
        all_targets |= set(mapping.values())
    for name in node_names:
        if name not in all_targets:
            problems.append(f"Node {name!r} has no incoming edge — it can never be reached")

    # BFS reachability from START.
    visited: set[str] = set()
    frontier = [_START]
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(adjacency.get(current, []))

    unreachable = node_names - visited
    if unreachable:
        problems.append(f"Nodes unreachable from START: {sorted(unreachable)}")
    if _END not in visited:
        problems.append("END is not reachable from START by any path")

    return problems


def build_graph(settings: Settings | None = None, checkpointer=None):
    """
    Returns a compiled LangGraph app, built from the NODE_FACTORY / EDGES /
    CONDITIONAL_EDGES topology above. Import of `langgraph` is local to this
    function so the rest of the module (including validate_wiring) stays
    importable — and checkable — without it.
    """
    from langgraph.graph import END, START, StateGraph

    settings = settings or load_settings()
    model_router = ModelRouter(settings)
    deps = _Deps(model_router, settings)
    sentinel_map = {_START: START, _END: END}

    builder = StateGraph(State)

    for name, factory in NODE_FACTORY.items():
        builder.add_node(name, factory(deps))

    for src, dst in EDGES:
        builder.add_edge(sentinel_map.get(src, src), sentinel_map.get(dst, dst))

    for src, fn, mapping in CONDITIONAL_EDGES:
        resolved_mapping = {label: sentinel_map.get(target, target) for label, target in mapping.items()}
        builder.add_conditional_edges(src, fn, resolved_mapping)

    if checkpointer is None:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # SqliteSaver.from_conn_string(...) is a @contextmanager — calling it
        # directly (without a `with` block) returns a _GeneratorContextManager,
        # not a saver, which is exactly the TypeError you hit. For a
        # long-lived process like cli.py (not a short "with-block" script),
        # construct the sqlite3.Connection ourselves and hand it to
        # SqliteSaver directly instead.
        conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()  # creates the checkpoint tables on first run; safe/idempotent after that

    return builder.compile(checkpointer=checkpointer)

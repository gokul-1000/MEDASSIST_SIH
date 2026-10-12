# Medical Symptom Assessment — LangGraph Prototype

A working, free-model-only, CPU-only-hardware prototype of the architecture
in `medical-symptom-assessment-langgraph-architecture.md`, implementing every
refinement from your review list. This README is also the "recheck" — it's
explicit about what changed from the doc, what's untested, and what a real
deployment would still need.

## Quickstart (zero setup, ~30 seconds)

```bash
cd medassist
python run_tests.py        # 90 tests, pure stdlib, no installs, ~15ms
python cli.py               # interactive session, auto-falls-back to mock mode
```

With no API keys set, `cli.py` runs against `MockProvider` — deterministic,
keyword-based, **not a real medical model**, but enough to walk the entire
graph (routing → follow-up loop → RAG gating → diagnosis → safety → emergency
→ escalation/synthesis) end to end so you can see the control flow work.

## Real setup (free tier, ~5 minutes)

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env
# fill in GROQ_API_KEY from https://console.groq.com  (no credit card)
# fill in GEMINI_API_KEY from https://aistudio.google.com/apikey (no credit card)
python cli.py
```

Both signups are genuinely free, no-card developer tiers as of July 2026.

---

## Project structure

```
medassist/
├── state.py                    State schema (§2 of the doc)
├── config.py                   Model tiers, thresholds, escalation policy — all constants
├── confidence.py                Confidence Engine + Confidence Calibration (two distinct signals)
├── graph.py                     LangGraph StateGraph wiring — the ONLY file that needs langgraph installed
├── cli.py                       Interactive runner
├── observability.py              Per-session + aggregate metrics (§6)
├── knowledge_base/
│   ├── eye_kb.json, general_kb.json   Versioned KB (guideline_version, publication_date, retrieval_timestamp, source)
│   └── kb_loader.py
├── rules/rule_engine.py          Red flags, emergency levels, department mapping (§9, §10) — no LLM
├── models/
│   ├── llm_provider.py           Groq / Gemini / Mock provider implementations
│   ├── model_router.py           Tiered routing + retry/fallback + confidence-aware upgrade
│   └── embeddings_reranker.py    CPU-only embeddings + cross-encoder (with offline mock fallback)
├── rag/rag_pipeline.py           Retrieve → Rerank → Compress, conditional decision, async prefetch
├── nodes/                        One file per node (or tightly-coupled pair), matching §3
├── eval/                         Offline evaluation harness — precision/recall/F1/calibration error
└── tests/                        90 tests, stdlib unittest only, run in ~15ms with zero installs
```

---

## How the refinements map to code

| Your refinement | Where it lives |
|---|---|
| Split Supervisor → Router + Orchestrator | `nodes/router.py` (domain + candidate ranking only) vs. `nodes/orchestrator.py` (loop guard, circuit breaker, escalation policy) — genuinely separate files, separately unit-tested |
| Probabilistic disease router before the Planner | `nodes/router.py` seeds `candidate_diseases` via `confidence.rank_candidates()` immediately after domain classification, before the Planner ever runs |
| Expanded Human Escalation policy | `config.ESCALATION_TRIGGERS` (6 named triggers) + `nodes/orchestrator.py::evaluate_escalation_triggers()` — each trigger is its own testable function (see `tests/test_orchestrator_escalation.py`) |
| Version the medical KB | `knowledge_base/*_kb.json` `metadata` block (guideline_version, publication_date, retrieval_timestamp, document_source), threaded through `kb_loader.py` and into every RAG source citation |
| Evaluation mode | `eval/eval_harness.py` — precision/recall/F1 (macro + per-class) + Expected Calibration Error, separate from the operational metrics in `observability.py` |
| Confidence-aware model routing | `models/model_router.py::ModelRouter.call(escalate_if_low_confidence=...)` — a low self-reported confidence retries against the "large" tier automatically |
| Clarify the two confidence mechanisms | `confidence.py` module docstring + two separate functions (`compute_loop_confidence` vs. `calibrate_confidence`) in two separate graph nodes |
| Document the state lifecycle | See "State lifecycle" section below |

## Round 2: clinical content + closing the verification gap as far as it'll go

You asked to push toward 10, starting with clinical content. Here's exactly
what changed and, as before, what still didn't.

**Knowledge base: 12 diseases → 30.** Eye went from 5 to 13 (added
blepharitis, subconjunctival hemorrhage, scleritis, keratitis, uveitis,
allergic conjunctivitis, chalazion/stye, cataract). General went from 7 to
17 (added sinusitis, pharyngitis, tension headache, GERD, UTI, kidney stone,
pyelonephritis, pulmonary embolism, anxiety/panic attack, anaphylaxis). The
new entries are deliberately not all easy: several intentionally overlap on
required symptoms with an existing entry (migraine/tension headache,
conjunctivitis/allergic conjunctivitis, corneal abrasion/keratitis) the same
way real differentials do, and one cluster — acute coronary syndrome /
pulmonary embolism / panic attack — shares its entire required-symptom set
on purpose, because a self-assessment tool has no business trying to
distinguish those from symptoms alone. That's not a gap; it's the correct
behavior, and it's specifically why `chest_pain + shortness_of_breath` is a
hard rule-engine short-circuit (see below) that fires regardless of which of
the three the ranking heuristic happens to favor.

This is still hand-curated general medical knowledge, not a licensed
guideline feed — the `guideline_version` fields are still intentionally
fabricated-looking placeholders demonstrating the versioning pattern, not
real citations. That disclaimer isn't going away just because the list got
longer; a bigger illustrative KB is still an illustrative KB.

**A real, previously-undetected bug in the scoring heuristic, found by
stress-testing it against the bigger KB.** The original `_symptom_match_score`
gave a small bonus for explicitly-denied *supporting* symptoms, on the
theory that "a clean no narrows things down too." Once diseases with
overlapping supporting/contraindicating lists existed, that turned out to be
backwards — it let a common disease win purely by being told what it *didn't*
have, which isn't real evidence for it. Removed. In its place: a symmetric,
directionally-correct bonus for denied *contraindicating* symptoms (ruling
out a more alarming alternative's warning sign is real evidence for the
milder candidate) — this is also what finally let `subconjunctival_hemorrhage`,
a diagnose-by-exclusion case with zero supporting symptoms of its own, earn
any score at all. See `confidence.py` for the full before/after reasoning.

**The offline eval numbers got worse, on purpose, and that's the point.**
29 labeled cases now, accuracy 0.86 (down from a hand-picked-cases 1.0
before the KB expansion). All 4 remaining misses are genuine thin-margin
overlaps — e.g. "headache" alone defaults to tension_headache over migraine
because there's no differentiating symptom in that specific case. A metric
that only looks good because the test cases were too easy isn't a real
metric; run `python -c "from eval.eval_harness import load_eval_cases, run_offline_eval; ..."`
(see `eval/eval_harness.py` docstring) if you want to see exactly which
cases miss and why — every one is explainable, not a mystery.

**Closed part of the LangGraph-execution gap — as far as it can be closed
without installing LangGraph.** `graph.py` is refactored so its topology
(`NODE_FACTORY`, `EDGES`, `CONDITIONAL_EDGES`) is plain data instead of being
buried inside a sequence of builder calls, and `validate_wiring()` statically
checks it — every edge target is a real node, every node is reachable from
START, END is reachable — using nothing but stdlib. `tests/test_graph_wiring.py`
runs that check, plus calls every `_route_after_*` routing function directly
with crafted states (16 new tests total). This catches real classes of bugs
(typo'd node names, orphaned nodes, unreachable branches) before you'd ever
see a LangGraph traceback — but it is still not the same thing as compiling
and running the actual `StateGraph`. I tried `pip install langgraph` again
this round specifically to check whether that had become possible; it
hasn't (no network in this environment) — same limitation as before, now
re-verified rather than assumed.

**A real correctness gap, found and fixed: hard red flags didn't actually
skip the follow-up loop.** The architecture doc's §9 says a red-flag
combination should bypass the rest of the loop. The first version never
implemented that — `check_red_flags()` was only called inside Safety
Validator, which sits *after* the full 3-question loop already ran. Someone
typing "chest pain and I can't breathe" would have been asked up to 3 more
questions before anything treated it as urgent. Fixed in
`_route_after_confidence_engine` (`graph.py`), with a dedicated integration
test (`test_red_flag_short_circuit_consumes_zero_follow_ups`) proving it now
takes zero follow-ups for that exact case. This is the kind of thing I'd
want a human reviewer to catch in a PR — I'm glad the bigger KB surfaced it
instead of it shipping quietly.

**Test count: 74 → 90.** All passing.



A "recheck" that just re-confirms the diagram isn't a recheck. Three real
issues turned up while actually building this:

**1. A state-write conflict in the doc's own mermaid diagram.** The diagram
has Safety Validator fan out to Calibration, Emergency Classification, *and*
Department Recommendation unconditionally, with a separate edge to Human
Escalation. But Department Recommendation and Escalation both write
`recommended_department` — if they ever land in the same LangGraph superstep,
that's an unresolvable concurrent write. Fixed in `graph.py` by joining
Calibration + Emergency Classification at a gate node first, evaluating the
escalation decision there, and only *then* branching into the (now mutually
exclusive) Department Recommendation vs. Escalation paths. Side benefit: the
Escalation message can now actually reference `emergency_level` ("this also
looks urgent"), which the original ordering couldn't have supported.

**2. A LangGraph `interrupt()` gotcha in the Follow-Up node.** LangGraph
re-executes a node's full function body on every resume from an
`interrupt()` call inside it — only the interrupt call itself replays. Put
the question-generation LLM call in the same node as `interrupt()`, and it
silently re-runs (different cost, possibly a different question) on every
resume. Fixed by splitting into `follow_up_generate` (the LLM call) and
`follow_up_interrupt` (nothing but `interrupt()` + recording the answer —
safe to re-run). This is documented in `nodes/follow_up.py`'s docstring.

**3. A real bug in my own mock provider**, caught by the test suite, not
by inspection: `_extract_json_field`'s scalar-value regex was tried before
its list-value regex, so against a field like `"raw_symptom_phrases": [...]`
it would match just the leading `[` (truthy, wrong) instead of falling
through to the list parser. Every follow-up answer would have silently
failed to normalize into `extracted_symptoms`. Caught because
`test_normalizer_writes_booleans_for_known_vocabulary_only` and the
red-flag end-to-end test both failed with a clear signal (`KeyError`, wrong
emergency level) — this is the actual argument for the test suite, not a
formality: it caught a bug that would have made every follow-up answer a
no-op. Fixed in `models/llm_provider.py`.

I'm not certain there are zero remaining issues of this kind — the honest
answer is that `graph.py` itself (the actual LangGraph wiring) has **not**
been executed even once, for the plain reason stated below.

## Round 3: the first bug LangGraph itself actually caught

This is the milestone the whole "unverified execution" caveat above was
waiting for: running `build_graph()` for real, on real hardware, with
`langgraph` actually installed. It failed immediately with:

```
TypeError: Invalid checkpointer provided. Expected an instance of
`BaseCheckpointSaver`, `True`, `False`, or `None`. Received
_GeneratorContextManager.
```

The cause: `SqliteSaver.from_conn_string(path)` is a `@contextmanager` —
calling it without a `with` block returns the context manager object
itself, not a saver. That pattern is fine for a short script that opens the
checkpointer, runs one thing, and exits inside the `with` block; it doesn't
fit a long-lived process like `cli.py` that needs the checkpointer to stay
alive for the whole interactive session.

Fix (in `graph.py`): construct the `sqlite3.Connection` directly and pass it
to `SqliteSaver(conn)`, which is a plain constructor, not a context manager.
Confirmed against LangChain's current reference docs for `SqliteSaver`
before shipping the fix back — same pattern, not just re-guessing. Also
added an explicit `checkpointer.setup()` call so the checkpoint tables are
created immediately rather than lazily on first use.

`example_mock_session.py` is new this round: a non-interactive script that
runs two full scripted sessions (a normal case, and a hard-red-flag case)
through the REAL `build_graph()` and `app.invoke()`/`Command(resume=...)` —
no typing required, no API key required. If it runs to completion on your
machine, that's the first actual confirmation this graph's wiring works
through LangGraph itself, not just through the offline node-chaining tests.



The sandbox this was built in has no internet access and none of
`langgraph`, `torch`, `sentence-transformers`, `groq`, or
`google-generativeai` installed — which is realistically close to "brand new
laptop, nothing set up yet." Rather than write code I couldn't check, the
project is split so that:

- Everything with actual decision logic (`confidence.py`, `rules/rule_engine.py`,
  `nodes/orchestrator.py`, and every node function) is a **plain Python
  function over dicts** — zero LangGraph, zero network, zero GPU dependency.
  90 unit tests exercise this layer directly, plus one hand-rolled
  end-to-end test (`tests/test_end_to_end_offline.py`) that chains every
  node function in the exact order `graph.py` wires them, using the mock
  provider, to validate the *whole pipeline's control flow* — loop
  termination, RAG gating, red-flag override, escalation routing — without
  LangGraph in the loop.
- `graph.py`'s topology (`NODE_FACTORY`/`EDGES`/`CONDITIONAL_EDGES`) is
  statically checked by `validate_wiring()` and `tests/test_graph_wiring.py`
  — every edge target is a real node, every node is reachable, every
  `_route_after_*` function is tested directly. This catches wiring bugs
  (typos, orphaned nodes) but, importantly, does NOT catch a LangGraph API
  version mismatch or a runtime-only failure — see the next bullet.
- `graph.py` itself is a thin wiring layer: it imports `langgraph` lazily
  (inside `build_graph()`, not at module level), so the module imports and
  byte-compiles cleanly here, but the actual `StateGraph` has never been
  compiled or run. All model IDs and API patterns (`interrupt()`,
  `Command(resume=...)`, `SqliteSaver`, conditional edges, parallel fan-out)
  were checked against LangGraph's current docs during this session, but
  "checked against docs" and "actually run" are different levels of
  confidence, and I want to be direct about which one this is.

**First thing to do on your laptop:** `pip install langgraph langgraph-checkpoint-sqlite`
and run `python cli.py` in mock mode (no API keys needed yet). If
`build_graph()` raises anything, it's almost certainly a LangGraph API
version mismatch (their conditional-edges / interrupt signatures have
shifted across versions) — the fix is usually small, but I can't promise
it's zero without having run it.

## Model selection (verified July 2026 — re-check before trusting later)

| Tier | Model | Source |
|---|---|---|
| Moderation | `openai/gpt-oss-safeguard-20b` (Groq) | Groq's dedicated safety-model lineup |
| Small | `openai/gpt-oss-20b` (Groq, free tier) | Replaced `llama-3.1-8b-instant` after Groq's June 17, 2026 deprecation |
| Medium | `openai/gpt-oss-120b` (Groq, free tier) | Replaced `llama-3.3-70b-versatile` (same deprecation wave) |
| Medium fallback | `qwen/qwen3.6-27b` (Groq) | Currently a *preview* model on Groq (vision-capable) — fine as a same-tier fallback, but preview models can be pulled with less notice than GA ones |
| Large (free-only substitution) | `openai/gpt-oss-120b` → `gemini-2.5-flash` (Google AI Studio free tier) | See "Where free-only costs you something" below — this replaces the doc's original paid Claude Sonnet 5 tier |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, ~22M params, CPU-friendly |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Small cross-encoder, CPU-friendly |

I checked **Med-Gemini / MedGemma** specifically, since you asked for a
medical-domain model: `google/medgemma-27b-it` and `medgemma-4b-it` are real,
open-weight, and free to download from Hugging Face — but they are **not**
available on Groq's free/fast tier, and Hugging Face's own free serverless
inference is explicitly scoped to small (<10B) CPU-bound models now, not
27B-class chat models. Running MedGemma-27B means a paid GPU endpoint;
MedGemma-4B could theoretically run quantized on your i9 via `llama.cpp`,
but slowly enough (low single-digit tokens/sec, no GPU) that it breaks the
architecture's own "hosted API, not local inference" rule for an
interactive multi-turn tool. **Verdict: not usable for free, real-time
inference in this design as of now** — noted rather than silently dropped,
in case Groq, Together, or another free-tier provider picks it up later.

### Where "free-only" visibly costs you something

The original doc used Claude Sonnet 5 (paid) for the hardest ~5-10% of
cases — rare diseases, low confidence after follow-ups + RAG. The free-only
substitution (bigger Groq model → Gemini Flash) is a real quality
downgrade specifically on that slice, not a wash. If this ever stops being
a course prototype, that's the one place I'd spend money first, not on
infrastructure.

## State lifecycle (the refinement about documenting how fields evolve)

```
raw_query ──► domain, candidate_diseases (Router)
                    │
                    ▼
   ┌──── missing_info (Planner) ◄────────────────┐
   │            │                                │
   │            ▼                                │
   │   raw_symptom_text (Extractor)               │
   │            │                                │
   │            ▼                                │
   │   extracted_symptoms (Normalizer, merged)     │
   │            │                                │
   │            ▼                                │
   │   confidence_score, confidence_history        │
   │   (Confidence Engine)                        │
   │            │                                │
   │       [below threshold, n < max]             │
   │            │                                │
   │            ▼                                │
   │   pending_question (Follow-Up Generate)      │
   │            │                                │
   │            ▼                                │
   └── follow_up_history += QA (Follow-Up Interrupt, on resume)
                │
           [threshold met, or n == max]
                ▼
   rag_required (Conditional RAG Decision)
                │
        ┌───────┴────────┐
        ▼                ▼
  rag_context,      (skip — Direct Reasoning)
  rag_similarity
        └───────┬────────┘
                ▼
   predicted_disease, raw_model_confidence (Diagnosis)
                ▼
   safety_flags (Safety Validator — may force emergency_level=3 later)
                ▼
      ┌─────────┴─────────┐
      ▼                   ▼
calibrated_confidence   emergency_level
 (Calibration)          (Emergency Classification)
      └─────────┬─────────┘
                ▼
   [escalation triggers evaluated against everything above]
                │
        ┌───────┴────────┐
        ▼                ▼
  escalated=True    recommended_department
  escalation_reason  (Department Recommendation)
        └───────┬────────┘
                ▼
        final_response (Synthesis)
```

---

## Honest rating (updated)

Where this stands after round 2, same discipline as before — specifics, not
a number pulled to make you feel good:

- **Architecture design: high confidence, slightly higher than before.**
  Two more real issues found and fixed this round (the backwards scoring
  bonus, the missing red-flag short-circuit), on top of the two from round 1.
  Four for four caught by actually stress-testing the system rather than
  re-reading the diagram, which is the whole point of a recheck.
- **Business logic correctness: high confidence, more thoroughly exercised.**
  90 tests (up from 74), now including static wiring validation and direct
  tests of every routing function `graph.py` uses — this is meaningfully
  more coverage of the orchestration layer than round 1 had, even though it
  still isn't LangGraph itself running.
- **Actual LangGraph execution: still unverified, and I want to be precise
  about why "closed the gap as far as it'll go" isn't the same as "closed."**
  `graph.py`'s topology is now statically checked, which catches wiring bugs
  — but it cannot catch a LangGraph API version mismatch, a state-reducer
  misconfiguration, or an actual runtime error that only shows up when the
  real Pregel engine executes a superstep. That category of bug is still
  fully unverified. I re-confirmed this round that installing LangGraph
  isn't possible in this environment (no network) rather than just
  repeating last round's claim.
- **Clinical content: meaningfully broader, still illustrative.** 30 diseases
  instead of 12, with genuine differential-diagnosis overlaps modeled on
  purpose. Still not licensed, still not a replacement for a real guideline
  source — a bigger hand-curated dataset is a better hand-curated dataset,
  not a validated one.
- **RAG corpus: grew proportionally with the KB** (each disease entry
  becomes a retrievable passage), which helps, but it's still synthesized
  from the same 30 hand-written entries rather than a real corpus — the
  retrieval mechanics are more exercised now, the content grounding them
  isn't fundamentally different in kind.

Net movement from round 1: architecture and business-logic confidence went
up for real reasons (more bugs caught, more coverage). The LangGraph-
execution and real-clinical-content ceilings didn't move, because the two
things that would move them — installing LangGraph, and a licensed data
source — aren't available in this environment. I'd rather tell you that
plainly than round the number up.

## Next steps, in the order I'd actually do them

1. `pip install langgraph langgraph-checkpoint-sqlite` and run `cli.py` in
   mock mode — find and fix any LangGraph API mismatches first, before
   spending a single API call on it. `validate_wiring()` passing gives you
   confidence the topology itself is sound; it can't catch a LangGraph
   version mismatch, so this step still matters.
2. Add your Groq key, rerun a few sessions, eyeball whether the follow-up
   questions and diagnoses look sane for 3-4 manual cases per domain —
   specifically try the overlapping pairs (migraine vs. tension headache,
   conjunctivitis vs. allergic conjunctivitis) and confirm the live loop
   actually asks the differentiating question instead of guessing.
3. If this goes beyond a course prototype: replace the illustrative KB with
   real, licensed, versioned clinical content. 30 diseases is broader
   coverage of the architecture's differential-diagnosis logic, not a
   substitute for this step — that distinction doesn't disappear no matter
   how many more diseases get added by hand.
4. Wire `eval/eval_harness.py::evaluate_predictions()` up to real graph
   output (log `predicted_disease` + `calibrated_confidence` per session
   against a held-out labeled set) instead of only the offline heuristic
   path — this is what would tell you whether the LLM + RAG + follow-up
   loop actually resolves the thin-margin cases the offline heuristic can't.
5. Only then worry about the Large-tier free-vs-paid question above.

---

## Hospital frontend (role-based demo)

`app.py` is now a role-based hospital app. The **triage engine is real** (`graph.py`); hospital data (doctors, queues, beds, diagnostics) is **simulated** in `frontend/store.py`.

```
MEDASSIST_MOCK=1 streamlit run app.py     # offline demo, no API keys
streamlit run app.py                      # live LLMs (keys in .env)
```

| Role | Login (demo) | What they get |
|---|---|---|
| Patient | phone `9876500001`, OTP `123456` | AI symptom check → department + crowd-aware doctor → token → live journey, records, next-visit plan |
| Doctor | `dr.sharma` … / `demo123` | Priority queue, AI pre-brief, prescriptions, test orders, referral, next-visit plan |
| Triage nurse | `nurse.manjeet` / `demo123` | Red-flag queue, vitals, priority override, walk-in triage, diagnostics desk |
| Reception | `desk.front` / `demo123` | Walk-in registration + triage, live queue board, reassign, follow-up check-in |
| Medical director | `admin` / `demo123` | Command center, alerts + one-click rebalance, resources, impact scorecard, guideline governance, audit log |

State is shared across browser sessions, so open a patient window and a doctor window to watch one visit move live.
Layout: `frontend/store.py` (mock data) · `orchestrator.py` (crowd/wait/route logic) · `triage_service.py` (graph adapter) · `views/*` (one file per role).

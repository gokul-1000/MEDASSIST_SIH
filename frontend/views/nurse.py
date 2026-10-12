from __future__ import annotations
import time
import streamlit as st
from .. import orchestrator as orch
from ..store import get_store
from ..theme import html, header, card, badge, level_badge, kpi
from . import reception

@st.fragment(run_every="6s")
def _priority(name):
    store = get_store()
    cases = [v for v in store.waiting() if v["level"] == 3 or v["triage"].get("escalated") or (v["triage"].get("confidence") is not None and v["triage"]["confidence"] < .5)]
    c = st.columns(3)
    c[0].markdown(kpi("Needs review", len(cases), "priority / low confidence", "bad" if cases else "good"), unsafe_allow_html=True)
    c[1].markdown(kpi("Red flags today", len([v for v in store.visits.values() if v["level"] == 3]), "safety rules fired"), unsafe_allow_html=True)
    c[2].markdown(kpi("ER doctors free", len([d for d in store.doctors.values() if d["dept"] == "Emergency Medicine" and d["status"] == "available"]), "on shift"), unsafe_allow_html=True)
    html("<br>")
    if not cases: st.success("No cases need your attention right now.")
    for v in cases:
        p = store.patients[v["patient"]]; t = v["triage"]
        card(f'<div class="row"><div><b>{v["token"]} · {p["name"]}</b> · {p["age"]}{p["gender"]}<div class="muted">“{t["complaint"]}”</div></div><div>{level_badge(v["level"])}{badge(t["department_raw"],"or")}</div></div>'
             + "".join(f'<div style="color:#C1121F;font-weight:600">⚠ {f.get("detail", f.get("code"))}</div>' for f in t["flags"]) + (f'<div class="muted">Reason: {t["escalation_reason"]}</div>' if t.get("escalation_reason") else ""), "red" if v["level"] == 3 else "")
        c = st.columns([1, 1, 1, 1, 1, 1.4])
        bp, hr, sp, tp = c[0].text_input("BP", v["vitals"].get("BP", ""), key=f"bp{v['id']}"), c[1].text_input("HR", v["vitals"].get("HR", ""), key=f"hr{v['id']}"), c[2].text_input("SpO2", v["vitals"].get("SpO2", ""), key=f"sp{v['id']}"), c[3].text_input("Temp", v["vitals"].get("Temp", ""), key=f"tp{v['id']}")
        lvl = c[4].selectbox("Priority", [1, 2, 3], index=v["level"] - 1, key=f"lv{v['id']}", format_func=lambda x: {1: "1 · Routine", 2: "2 · Urgent", 3: "3 · Emergency"}[x])
        c[5].markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if c[5].button("Save & confirm", key=f"sv{v['id']}", type="primary", use_container_width=True):
            vit = {k: x for k, x in dict(BP=bp, HR=hr, SpO2=sp, Temp=tp).items() if x}
            if vit: store.add_vitals(v["id"], vit, name)
            if lvl != v["level"]: store.set_priority(v["id"], lvl, name)
            if lvl == 3 and v["dept"] != "Emergency Medicine": store.refer(v["id"], "Emergency Medicine", "nurse-confirmed emergency", name)
            st.toast("Saved. Doctor's pre-brief updated."); st.rerun()

@st.fragment(run_every="6s")
def _diagnostics(name):
    store = get_store(); rows = [(v, o) for v in store.visits.values() for o in v["orders"] if o["status"] == "pending"]
    if not rows: st.info("No pending test orders."); return
    for dept in ("Laboratory", "Radiology"):
        sub = [(v, o) for v, o in rows if o["dept"] == dept]
        st.markdown(f"##### {dept} · {len(sub)} pending · ~{orch.diag_status(store, dept)['wait']} min wait")
        for v, o in sub:
            c = st.columns([2.2, 3, 1])
            c[0].markdown(f'**{o["name"]}** · {v["token"]} · {store.patients[v["patient"]]["name"]}')
            res = c[1].text_input("Result", key=f"res{v['id']}{o['id']}", label_visibility="collapsed", placeholder="Result summary…")
            if c[2].button("Mark done", key=f"md{v['id']}{o['id']}", use_container_width=True): store.mark_test_done(v["id"], o["id"], res or "Reported normal"); st.rerun()

def render(page, user):
    if page == "Priority Queue":
        header("Triage Nurse Desk", "Red-flag and low-confidence cases land here first. Confirm vitals and priority; the doctor sees it instantly.", "TRIAGE NURSE"); _priority(user["name"])
    elif page == "Walk-in Triage":
        header("Walk-in Triage", "Assist a patient through the AI symptom check.", "TRIAGE NURSE"); reception.walkin("Nurse-assisted", "nur")
    elif page == "Diagnostics Desk":
        header("Diagnostics Desk", "Mock lab and radiology desk. Marking a test done sends the patient straight back to the doctor's queue.", "TRIAGE NURSE"); _diagnostics(user["name"])

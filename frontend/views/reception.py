from __future__ import annotations
import time, datetime as dt
import streamlit as st
from .. import orchestrator as orch
from ..store import DEPTS, get_store
from ..theme import html, header, card, badge, level_badge, crowd_badge, kpi
from . import triage_chat

def walkin(source: str, key: str):
    """Staff-assisted registration + triage (kiosk mode) for patients without the app."""
    store = get_store(); sk = f"walkin_{key}"
    pid = st.session_state.get(sk)
    if not pid:
        c = st.columns([1, 1, 1, 1])
        phone = c[0].text_input("Mobile number", key=f"{sk}_ph", placeholder="98765 xxxxx")
        existing = store.find_patient_by_phone(phone.strip()) if phone else None
        if existing: card(f'<b>Found:</b> {existing["name"]} · {existing["age"]}{existing["gender"]} · ABHA {existing["abha"]}', "green")
        name = c[1].text_input("Full name", key=f"{sk}_nm", value=existing["name"] if existing else "")
        age = c[2].number_input("Age", 0, 110, existing["age"] if existing else 35, key=f"{sk}_ag")
        gender = c[3].selectbox("Gender", ["M", "F", "O"], index=["M", "F", "O"].index(existing["gender"]) if existing else 0, key=f"{sk}_gn")
        if st.button("Register and start symptom check", type="primary", disabled=not (name and phone)):
            p = existing or store.create_patient(name, age, gender, phone.strip()); st.session_state[sk] = p["id"]; st.rerun()
        return
    p = store.patients[pid]
    html(f'<div class="row"><div><span class="eyebrow">PATIENT</span> <b>{p["name"]}</b> · {p["age"]}{p["gender"]} · {p["phone"]}</div></div>')
    triage_chat.render(dict(id=pid, age=p["age"], gender=p["gender"]), sk, source)
    if st.button("Finish and register next patient"): st.session_state.pop(sk, None); triage_chat.reset(sk); st.rerun()

@st.fragment(run_every="6s")
def board():
    store = get_store(); vs = store.waiting()
    c = st.columns(4)
    for col, (l, v, s) in zip(c, [("Waiting", len(vs), "triaged patients"), ("In consultation", len([x for x in store.visits.values() if x["status"] == "in_consult"]), "right now"),
                                  ("Awaiting tests", len([x for x in store.visits.values() if x["status"] == "awaiting_tests"]), "lab / radiology"), ("Emergency priority", len([x for x in vs if x["level"] == 3]), "needs nurse")]):
        col.markdown(kpi(l, v, s), unsafe_allow_html=True)
    html("<br>")
    for v in vs:
        p = store.patients[v["patient"]]; d = store.doctors[v["doctor"]]; mins = int((time.time() - v["created"]) / 60)
        card(f'<div class="row"><div><b>{v["token"]}</b> · {p["name"]} <span class="muted">· {v["dept"]} · {d["name"]} · {d["room"]}</span></div><div>{level_badge(v["level"])}{badge(v["source"],"gray")}<span class="tiny"> {mins} min</span></div></div>', "red" if v["level"] == 3 else "")

def _reassign():
    store = get_store(); vs = store.waiting()
    if not vs: return
    st.markdown("##### Reassign a patient to another doctor")
    c = st.columns([2, 2, 1])
    vid = c[0].selectbox("Patient", [v["id"] for v in vs], format_func=lambda i: f'{store.visits[i]["token"]} · {store.patients[store.visits[i]["patient"]]["name"]}')
    v = store.visits[vid]; rows = orch.rank_doctors(store, v["dept"])
    did = c[1].selectbox("Doctor", [r["doctor"]["id"] for r in rows], format_func=lambda i: f'{store.doctors[i]["name"]} · {orch.queue_len(store, store.doctors[i])} waiting')
    if c[2].button("Reassign", use_container_width=True): store.reassign(vid, did, "Front desk"); st.rerun()

def _followups():
    store = get_store(); today = dt.date.today(); rows = []
    for v in store.visits.values():
        n = v["next_visit"]
        if n and v["status"] == "completed" and dt.date.fromisoformat(n["date"]) <= today + dt.timedelta(days=3): rows.append(v)
    if not rows: st.info("No follow-ups due in the next 3 days."); return
    for v in rows:
        n, p = v["next_visit"], store.patients[v["patient"]]
        c = st.columns([4, 1])
        c[0].markdown(f'<div class="card"><b>{p["name"]}</b> · {n["date"]} · {n["dept"]}<div class="muted">{n["purpose"]} · {n["doctor"]}</div></div>', unsafe_allow_html=True)
        done = any(x["patient"] == p["id"] and x["source"] == "Follow-up check-in" and x["status"] != "completed" for x in store.visits.values())
        if c[1].button("Checked in" if done else "Check in", key=f"ci_{v['id']}", disabled=done, type="primary"):
            dest = n["dept"]; doc = orch.recommend_doctor(store, dest)
            tri = dict(complaint=f"Follow-up: {n['purpose']}", domain="general", symptoms_present=[], symptoms_absent=[], followups=[], predicted=None, differential=[], confidence=None, level=1, department_raw=dest, flags=[], escalated=False, escalation_reason=None, sources=[], age=p["age"], gender=p["gender"], attributes={}, final_response=None)
            store.create_visit(p["id"], tri, dest, doc["id"], "Follow-up check-in"); st.rerun()

def render(page, user):
    if page == "Front Desk":
        header("Front Desk", "Register a walk-in and run the AI symptom check on their behalf. Hindi / Punjabi / English.", "RECEPTION"); walkin("Reception kiosk", "rec")
    elif page == "Live Queue":
        header("Live Queue Board", "Every triaged patient, across departments, in priority order.", "RECEPTION"); board(); _reassign()
    elif page == "Follow-up Check-in":
        header("Follow-up Check-in", "Planned next visits from doctors. One tap puts the patient into the right queue.", "RECEPTION"); _followups()

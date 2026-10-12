from __future__ import annotations
import time, datetime as dt
import streamlit as st
from .. import orchestrator as orch
from ..store import DEPTS, get_store
from ..theme import html, header, card, badge, level_badge, crowd_badge, kpi, bar
from . import triage_chat

def _journey(store, v, compact=False):
    steps = orch.route_plan(store, v); out = ""
    for i, s in enumerate(steps):
        cls = s["state"]; mark = "✓" if cls == "done" else str(i + 1)
        out += f'<div class="step"><div class="dot {cls}">{mark}</div><div><div class="t">{s["title"]}</div><div class="d">{s["detail"]}</div></div></div>'
    return out

@st.fragment(run_every="6s")
def _live_journey(pid):
    store = get_store(); v = store.active_visit(pid)
    if not v: st.info("No active visit. Start a symptom check to get guided."); return
    d = store.doctors[v["doctor"]]; w = orch.wait_min(store, d, v["level"])
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        head = {"waiting": f"Please wait near <b>{d['room']}</b>. About <b>{w} min</b> to go.", "in_consult": f"You're with <b>{d['name']}</b> now.",
                "awaiting_tests": "Go for your tests. Results reach your doctor automatically.", "review_waiting": f"Results are in. Go back to <b>{d['room']}</b>.", "completed": "Consultation done. Collect your medicines."}[v["status"]]
        card(f'<div class="row"><div><div class="pill-lbl">WHERE TO GO NOW</div><div style="font-size:1.15rem;margin-top:.3rem">{head}</div></div><div style="text-align:right"><div class="token">{v["token"]}</div>{level_badge(v["level"])}</div></div>', "hl")
        html('<div class="card">' + _journey(store, v) + '</div>')
    with c2:
        for o in v["orders"]:
            ds = orch.diag_status(store, o["dept"])
            card(f'<div class="row"><b>{o["name"]}</b>{badge("Done","green") if o["status"]=="done" else badge("Pending","amber")}</div><div class="muted">{o["dept"]} · {ds["loc"].split("·")[-1].strip()}' + (f' · ~{ds["wait"]} min wait' if o["status"] == "pending" else f'</div><div style="margin-top:.3rem">{o["result"]}') + '</div>')
        if v["next_visit"]: _next_visit_card(v["next_visit"])
        with st.expander("Activity log"):
            for e in reversed(v["timeline"]): st.caption(f"{time.strftime('%H:%M', time.localtime(e['ts']))} · {e['text']}")
        if v["status"] == "completed" and v["rx"] and st.button("I collected my medicines", use_container_width=True): v["rx_done"] = True; st.rerun()

def _next_visit_card(n):
    card(f'<div class="pill-lbl">NEXT VISIT PLAN</div><h4 style="margin:.2rem 0">{n["date"]} · {n["dept"]}</h4><div class="muted">{n.get("doctor","")}</div>'
         f'<div style="margin-top:.5rem"><b>Purpose:</b> {n.get("purpose","")}</div><div class="muted" style="margin-top:.2rem">{n.get("instructions","")}</div>', "green")

def render(page: str, user: dict):
    store = get_store(); pid = user["id"]; p = store.patients[pid]
    ident = dict(id=pid, age=p["age"], gender=p["gender"])
    if page == "Home":
        header(f"Namaste, {p['name'].split()[0]}", "Your care journey at a glance. MedAssist guides you from the front door to the pharmacy.", "PATIENT PORTAL")
        v = store.active_visit(pid)
        if v: _live_journey(pid)
        else:
            c1, c2 = st.columns([1.3, 1], gap="large")
            with c1:
                card('<div class="pill-lbl">FEELING UNWELL?</div><h2 style="margin:.2rem 0 .4rem">Tell MedAssist what you\'re feeling</h2><p class="muted">Speak or type in Hindi, Punjabi or English. In about two minutes you\'ll know which department to visit, which doctor has the shortest wait, and what happens next.</p>', "hl")
                if st.button("Start symptom check", type="primary", use_container_width=True): st.session_state.nav_request = "Symptom Check"; st.rerun()
            with c2:
                hist = store.history(pid)
                nv = next((h["next_visit"] for h in hist if h["next_visit"]), None)
                if nv: _next_visit_card(nv)
                else: card('<div class="pill-lbl">NEXT VISIT</div><p class="muted">Nothing planned yet.</p>')
        html('<br>'); st.markdown("#### Hospital right now")
        cols = st.columns(4)
        for c, dn in zip(cols, ["General Medicine", "Cardiology", "Ophthalmology", "Emergency Medicine"]):
            s = orch.dept_status(store, dn); c.markdown(f'<div class="kpi"><div class="l">{dn}</div><div class="v">~{s["avg_wait"]} min</div><div class="s">{crowd_badge(s["label"])} {s["total"]} waiting</div></div>', unsafe_allow_html=True)
    elif page == "Symptom Check":
        header("AI Symptom Check", "A guided conversation. Your summary goes to the doctor so you never repeat yourself.", "TRIAGE")
        if store.active_visit(pid) and not st.session_state.get("tri_p_visit"): 
            st.warning("You already have an active visit. Follow it in My Journey.")
            if st.button("Go to My Journey", type="primary"): st.session_state.nav_request = "My Journey"; st.rerun()
        else: triage_chat.render(ident, "p", "MedAssist app")
    elif page == "My Journey":
        header("My Journey", "Live route through the hospital: where to go, how long, what's next.", "LIVE")
        _live_journey(pid)
    elif page == "Records":
        header("My Records", f"ABHA {p['abha']} · {p['age']} yrs · {p['gender']}", "HEALTH RECORD")
        hist = store.history(pid)
        if not hist: st.info("No past visits yet.")
        for h in hist:
            d = store.doctors[h["doctor"]]
            rx = "".join(f"<li>{r['name']} · {r['dose']} · {r['days']} days</li>" for r in h["rx"]) or "<li>None</li>"
            tests = "".join(f"<li>{o['name']}: {o['result']}</li>" for o in h["orders"]) or "<li>None</li>"
            card(f'<div class="row"><b>{time.strftime("%d %b %Y", time.localtime(h["created"]))} · {h["dept"]}</b>{badge(h["diagnosis"] or "Visit","or")}</div><div class="muted">{d["name"]}</div><div style="margin:.5rem 0">{h["notes"]}</div>'
                 f'<div class="pill-lbl">PRESCRIPTION</div><ul>{rx}</ul><div class="pill-lbl">REPORTS</div><ul>{tests}</ul>')
    elif page == "Hospital Guide":
        header("Hospital Guide", "Departments, locations and live crowd, so you can plan your walk.", "NAVIGATION")
        cols = st.columns(3)
        for i, dn in enumerate(DEPTS):
            s = orch.dept_status(store, dn)
            with cols[i % 3]: card(f'<div class="row"><b>{dn}</b>{crowd_badge(s["label"])}</div><div class="muted">{DEPTS[dn]["loc"]}</div>{bar(min(100, s["per_doc"]*10), s["tone"])}<span class="tiny">{s["total"]} waiting · ~{s["avg_wait"]} min · {s["active"]} doctors</span>')
        st.markdown("##### Diagnostics & pharmacy")
        cols = st.columns(3)
        for c, n in zip(cols, ["Laboratory", "Radiology", "Pharmacy"]):
            d = orch.diag_status(store, n); c.markdown(f'<div class="kpi"><div class="l">{n}</div><div class="v">~{d["wait"]} min</div><div class="s">{d["queue"]} in queue · {d["loc"].split("·")[-1].strip()}</div></div>', unsafe_allow_html=True)

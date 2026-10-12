from __future__ import annotations
import time, datetime as dt
import streamlit as st
from .. import orchestrator as orch
from ..store import DEPTS, TESTS, get_store
from ..theme import html, header, card, badge, level_badge, kpi

def _sym(l): return ", ".join(s.replace("_", " ") for s in l) or "none"

def prebrief(store, v):
    """The doctor's one-screen summary from the AI triage: no more 'tell me again from the start'."""
    t = v["triage"]; p = store.patients[v["patient"]]
    flags = "".join(f'<div style="color:#C1121F;font-weight:600">⚠ {f.get("detail", f.get("code"))}</div>' for f in t.get("flags", []))
    conf = "Escalated for human review" if t.get("escalated") else (f"{int(t['confidence']*100)}% calibrated" if t.get("confidence") is not None else "n/a")
    vit = " · ".join(f"{k} {x}" for k, x in v["vitals"].items()) or "not recorded yet"
    card(f'<div class="row"><div><div class="pill-lbl">AI PRE-BRIEF · TRIAGE SUMMARY</div><h3 style="margin:.2rem 0">{p["name"]} · {p["age"]}{p["gender"]} · {v["token"]}</h3></div><div>{level_badge(v["level"])}{badge(t["department_raw"],"or")}</div></div>'
         f'<p style="font-size:1.05rem;margin:.4rem 0"><b>Chief complaint:</b> “{t["complaint"]}”</p>{flags}'
         f'<div style="margin-top:.5rem"><b>Present:</b> {_sym(t["symptoms_present"])}<br><b>Absent:</b> {_sym(t["symptoms_absent"])}<br><b>Vitals:</b> {vit}</div>', "hl" if v["level"] < 3 else "red")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Differential (AI-assisted)**")
        for d in t["differential"][:4]: st.progress(min(1.0, d["score"]), text=f"{str(d['name']).replace('_',' ').title()} · {int(d['score']*100)}%")
        st.caption(f"Confidence: {conf}")
    with c2:
        st.markdown("**Follow-up Q&A captured**")
        for f in t["followups"]: st.markdown(f"- {f['q']} → **{f['a']}**")
        if not t["followups"]: st.caption("No follow-ups were needed.")
        if t["sources"]: st.caption("Guideline basis: " + " | ".join(t["sources"][:3]))
    st.caption("AI-assisted decision support only. The clinician decides.")

@st.fragment(run_every="6s")
def _queue(did):
    store = get_store(); d = store.doctors[did]; q = store.doctor_queue(did); cur = store.current(did)
    c = st.columns(4)
    for col, (l, val, s, tn) in zip(c, [("Waiting (MedAssist)", len(q), "triaged patients", ""), ("Walk-in backlog", d["base"], "not yet triaged", "warn" if d["base"] > 8 else ""),
                                       ("Est. wait for next", f"{orch.wait_min(store, d)} min", "at current pace", ""), ("Status", d["status"].title(), d["room"], "good" if d["status"] == "available" else "warn")]):
        col.markdown(kpi(l, val, s, tn), unsafe_allow_html=True)
    html("<br>")
    if cur:
        st.success(f"In consultation: {store.patients[cur['patient']]['name']} ({cur['token']}). Open the **Consultation** tab.")
    elif st.button("Call next patient", type="primary", disabled=not q or d["status"] != "available"): store.call_next(did); st.session_state.nav_request = "Consultation"; st.rerun()
    st.markdown("#### Queue (priority-sorted)")
    if not q: st.info("Queue is clear.")
    for v in q:
        p = store.patients[v["patient"]]; mins = int((time.time() - v["created"]) / 60); tag = badge("Back with results", "or") if v["status"] == "review_waiting" else ""
        card(f'<div class="row"><div><b>{v["token"]}</b> · {p["name"]} · {p["age"]}{p["gender"]} {tag}<div class="muted">“{v["triage"]["complaint"]}”</div></div><div style="text-align:right">{level_badge(v["level"])}<div class="tiny">waiting {mins} min</div></div></div>', "red" if v["level"] == 3 else "")

def _consult(did, name):
    store = get_store(); v = store.current(did)
    if not v: st.info("No patient in consultation. Call the next patient from **My Queue**."); return
    prebrief(store, v)
    st.markdown("#### Consultation")
    notes = st.text_area("Clinical notes", value=v["notes"], key=f"n_{v['id']}", height=110)
    diag = st.text_input("Diagnosis / impression", value=v["diagnosis"], key=f"d_{v['id']}")
    t1, t2, t3, t4 = st.tabs(["Prescription", "Tests & diagnostics", "Refer / transfer", "Next visit plan"])
    with t1:
        rx = st.session_state.setdefault(f"rx_{v['id']}", [])
        c = st.columns([2, 2, 1, 1])
        mn = c[0].text_input("Medicine", key=f"mn_{v['id']}"); ds = c[1].text_input("Dose / frequency", key=f"ds_{v['id']}"); dy = c[2].number_input("Days", 1, 90, 5, key=f"dy_{v['id']}")
        if c[3].button("Add", key=f"add_{v['id']}") and mn: rx.append(dict(name=mn, dose=ds or "as directed", days=dy)); st.rerun()
        for r in rx: st.markdown(f"- **{r['name']}** · {r['dose']} · {r['days']} days")
    with t2:
        pick = st.multiselect("Order tests", list(TESTS), format_func=lambda n: f"{n} → {TESTS[n]}", key=f"tp_{v['id']}")
        if st.button("Send orders", key=f"to_{v['id']}") and pick:
            store.order_tests(v["id"], pick, name); st.rerun()
        for o in v["orders"]:
            ds = orch.diag_status(store, o["dept"]); st.markdown(f"- **{o['name']}** → {o['dept']} · {o['status']} " + (f"(~{ds['wait']} min wait)" if o["status"] == "pending" else f"· {o['result']}"))
    with t3:
        others = [d for d in DEPTS if d != v["dept"]]; to = st.selectbox("Refer to department", others, key=f"rf_{v['id']}"); why = st.text_input("Reason", key=f"rr_{v['id']}")
        r = orch.rank_doctors(store, to)[0]; st.caption(f"MedAssist will route to **{r['doctor']['name']}** ({r['doctor']['room']}) · shortest queue, ~{r['wait']} min")
        if st.button("Refer patient", key=f"rb_{v['id']}"): store.refer(v["id"], to, why or "clinical need", name); st.success("Referred. Patient and next doctor notified."); st.session_state.pop(f"rx_{v['id']}", None); st.rerun()
    with t4:
        nv = v["next_visit"] or {}
        c = st.columns(2)
        date = c[0].date_input("Date", value=dt.date.today() + dt.timedelta(days=7), key=f"nd_{v['id']}")
        ndept = c[1].selectbox("Department", list(DEPTS), index=list(DEPTS).index(v["dept"]) if v["dept"] in DEPTS else 0, key=f"ndp_{v['id']}")
        purpose = st.text_input("Purpose of next visit", value=nv.get("purpose", ""), key=f"np_{v['id']}"); ins = st.text_input("Instructions for the patient", value=nv.get("instructions", ""), key=f"ni_{v['id']}")
        want = st.checkbox("Schedule a follow-up", value=bool(nv) or True, key=f"nw_{v['id']}")
    if st.button("Complete consultation", type="primary", use_container_width=True, key=f"cc_{v['id']}"):
        nxt = dict(date=str(date), dept=ndept, doctor=orch.rank_doctors(store, ndept)[0]["doctor"]["name"], purpose=purpose or "Follow-up", instructions=ins) if want else None
        store.complete_consult(v["id"], notes, diag, st.session_state.get(f"rx_{v['id']}", []), nxt, name); st.session_state.pop(f"rx_{v['id']}", None); st.session_state.nav_request = "My Queue"; st.rerun()

def render(page, user):
    store = get_store(); did = user["id"]; d = store.doctors[did]
    if page == "My Queue":
        header(d["name"], f"{d['dept']} · {d['room']}", "DOCTOR DESK")
        c = st.columns([3, 1])
        with c[1]:
            new = st.selectbox("Status", ["available", "break", "off"], index=["available", "break", "off"].index(d["status"]), label_visibility="collapsed")
            if new != d["status"]: store.set_doctor_status(did, new); st.rerun()
        _queue(did)
    elif page == "Consultation":
        header("Consultation", "AI pre-brief on top; notes, orders, referral and the next-visit plan below.", "DOCTOR DESK"); _consult(did, d["name"])
    elif page == "Follow-ups I planned":
        header("Follow-ups I planned", "What patients will see in their Next Visit plan.", "DOCTOR DESK")
        rows = [v for v in store.visits.values() if v["next_visit"] and v["doctor"] == did or (v["next_visit"] and v["next_visit"].get("doctor") == d["name"])]
        if not rows: st.info("No follow-ups planned yet.")
        for v in rows: n = v["next_visit"]; card(f'<div class="row"><b>{store.patients[v["patient"]]["name"]}</b>{badge(n["date"],"or")}</div><div class="muted">{n["dept"]} · {n["purpose"]}</div><div class="tiny">{n.get("instructions","")}</div>')
    elif page == "Patient Search":
        header("Patient lookup", "Search by name or phone.", "DOCTOR DESK")
        q = st.text_input("Search")
        for p in [p for p in store.patients.values() if q and (q.lower() in p["name"].lower() or q in p["phone"])]:
            hist = store.history(p["id"]); card(f'<b>{p["name"]}</b> · {p["age"]}{p["gender"]} · {p["phone"]}<div class="muted">{len(hist)} past visit(s). Last: {hist[0]["diagnosis"] if hist else "-"}</div>')

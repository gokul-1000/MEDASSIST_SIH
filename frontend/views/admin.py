from __future__ import annotations
import time
import pandas as pd
import streamlit as st
from .. import orchestrator as orch
from ..store import DEPTS, get_store
from ..theme import html, header, card, badge, crowd_badge, kpi, bar

@st.fragment(run_every="5s")
def _command():
    store = get_store(); ov = orch.overview(store); al = orch.alerts(store)
    html('<span class="sim">SIMULATED HOSPITAL DATA · HACKATHON PROTOTYPE</span><br><br>')
    c = st.columns(6)
    for col, (l, v, s, t) in zip(c, [("Patients today", ov["today"], "all channels", ""), ("In system now", ov["in_system"], "MedAssist-tracked", ""), ("Avg wait", f"{ov['avg_wait']}m", "across departments", "warn" if ov["avg_wait"] > 45 else "good"),
                                      ("Red-flag cases", ov["flagged"], "caught by safety rules", "bad" if ov["flagged"] else "good"), ("ICU beds free", ov["icu_free"], "of 20", "warn" if ov["icu_free"] < 4 else "good"), ("Doctors on duty", f'{ov["docs"]}/{ov["docs_total"]}', "available now", "")]):
        col.markdown(kpi(l, v, s, t), unsafe_allow_html=True)
    html("<br>")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("#### Department load")
        cols = st.columns(2)
        for i, dn in enumerate(DEPTS):
            s = orch.dept_status(store, dn)
            with cols[i % 2]: card(f'<div class="row"><b>{dn}</b>{crowd_badge(s["label"])}</div>{bar(min(100, s["per_doc"]*10), s["tone"])}<span class="tiny">{s["total"]} waiting · {s["active"]}/{s["doctors"]} doctors · ~{s["avg_wait"]} min · {s["high"]} urgent</span>')
    with right:
        st.markdown("#### Orchestration alerts")
        if not al: st.success("All departments are balanced.")
        for i, a in enumerate(al):
            card(f'<div class="pill-lbl" style="color:{"#C1121F" if a["kind"]=="red" else "#B26A00"}">{"CONGESTION" if a["rebalance"] else "ALERT"}</div><b>{a["title"]}</b><div class="muted">{a["detail"]}</div><div style="margin-top:.4rem">{a["action"]}</div>', "red" if a["kind"] == "red" else "")
            if a["rebalance"] and st.button("Apply: rebalance now", key=f"rb{i}_{a['dept']}", type="primary", use_container_width=True):
                moves = orch.rebalance(store, a["dept"], "Admin"); st.toast(" · ".join(moves) if moves else "Already balanced."); st.rerun()
        st.markdown("#### Patient inflow today")
        n = len(store.inflow); st.area_chart(pd.DataFrame({"patients": store.inflow}, index=[f"{(8+i)%24}:00" for i in range(n)]), color="#E8742F", height=190)
    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown("#### Beds & ICU")
        for k, b in store.beds.items():
            pct = int(100 * b["occ"] / b["total"]); st.markdown(f'<div class="row"><b>{k}</b><span class="tiny">{b["total"]-b["occ"]} free of {b["total"]}</span></div>{bar(pct, "red" if pct >= 85 else "amber" if pct >= 70 else "green")}', unsafe_allow_html=True)
    with b2:
        st.markdown("#### Diagnostics & pharmacy")
        for n_ in store.diag:
            d = orch.diag_status(store, n_); st.markdown(f'<div class="row"><b>{n_}</b><span class="tiny">{d["queue"]} in queue · ~{d["wait"]} min</span></div>{bar(min(100, d["wait"]*2.2), "red" if d["wait"] >= 30 else "amber" if d["wait"] >= 20 else "green")}', unsafe_allow_html=True)

def _controls():
    store = get_store(); st.markdown("##### Demo controls (simulate a day at the hospital)")
    c = st.columns(4)
    if c[0].button("Advance 10 minutes", use_container_width=True): store.tick(); st.rerun()
    sd = c[1].selectbox("Surge dept", list(DEPTS), index=2, label_visibility="collapsed")
    if c[2].button("Simulate surge +8", use_container_width=True): store.surge(sd, 8); st.rerun()
    if c[3].button("Reset demo data", use_container_width=True): st.cache_resource.clear(); st.rerun()

def _resources():
    store = get_store(); header("Doctors & Resources", "Live staffing and queue per doctor. Rebalancing and on-call counters are one click.", "ADMIN")
    rows = [dict(Doctor=d["name"], Department=d["dept"], Room=d["room"], Status=d["status"].title(), Waiting=orch.queue_len(store, d), **{"Est. wait (min)": orch.wait_min(store, d)}) for d in store.doctors.values()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    c = st.columns([2, 1, 1]); dn = c[0].selectbox("Department", list(DEPTS))
    if c[1].button("Rebalance load", use_container_width=True): st.toast(" · ".join(orch.rebalance(store, dn)) or "Already balanced."); st.rerun()
    if c[2].button("Open on-call counter", use_container_width=True): store.open_oncall(dn); st.rerun()

def _scorecard():
    header("Impact Scorecard", "The KPIs we commit to in the pitch deck, tracked against each hospital's own baseline.", "ADMIN")
    html('<span class="sim">SIMULATED PILOT VALUES</span><br><br>')
    rows = [("Time to reach the right counter", "−40%", 31, "−25% (90-day)"), ("Wrong-department visits", "−35%", 24, "−20% (90-day)"), ("Red-flag recall (safety)", "≥ 98%", 97, "≥ 95% (90-day)"),
            ("Intake in patient's own language", "≥ 95%", 91, "≥ 85% (90-day)"), ("Front-desk 'where do I go?' load", "−50%", 38, "−30% (90-day)"), ("Equity reach (60+, rural, non-English)", "≥ 55%", 52, "≥ 40% (90-day)")]
    for k, tgt, now, pilot in rows:
        c = st.columns([2.2, 1.2, 1.2, 2]); c[0].markdown(f"**{k}**"); c[1].markdown(f'<div class="kpi"><div class="l">Year-1 target</div><div class="v" style="font-size:1.3rem">{tgt}</div></div>', unsafe_allow_html=True)
        c[2].markdown(f'<div class="kpi good"><div class="l">Current</div><div class="v" style="font-size:1.3rem">{now}{"%" if "−" in tgt or "≥" in tgt else ""}</div></div>', unsafe_allow_html=True); c[3].markdown(f'{bar(now)}<span class="tiny">Pilot goal {pilot}</span>', unsafe_allow_html=True)

def _governance():
    header("Clinical Governance", "Guideline knowledge base and offline evaluation of the triage engine.", "ADMIN")
    from knowledge_base.kb_loader import load_kb, all_domains
    from eval.eval_harness import run_offline_eval, load_eval_cases
    t1, t2 = st.tabs(["Guidelines explorer", "Evaluation board"])
    with t1:
        domains = all_domains(); c = st.columns(2); dom = c[0].selectbox("Domain", domains, format_func=lambda x: f"{x.title()} Medicine")
        kb = load_kb(dom); names = [d["name"] for d in kb.diseases]; dn = c[1].selectbox("Condition", names, format_func=lambda x: x.replace("_", " ").title()); d = kb.disease_by_name(dn)
        if d:
            a, b = st.columns([1, 1.5], gap="large")
            a.markdown(f'<div class="card"><h4>{dn.replace("_"," ").title()}</h4><div class="muted">Department: <b>{d.get("department")}</b></div>{badge("RED FLAG","3") if d.get("red_flag") else badge("ROUTINE","1")}<div class="muted" style="margin-top:.5rem">Base rate: {int(d.get("base_rate",0)*100)}%</div></div>', unsafe_allow_html=True)
            j = lambda k: ", ".join(s.replace("_", " ") for s in d.get(k, [])) or "none"
            b.markdown(f"**Required:** {j('required_symptoms')}\n\n**Supporting:** {j('supporting_symptoms')}\n\n**Contraindicating:** {j('contraindicating_symptoms')}")
            st.caption(f"KB {kb.metadata.guideline_version} · published {kb.metadata.publication_date} · {kb.metadata.document_source}")
    with t2:
        rep = run_offline_eval(); cases = load_eval_cases(); m = st.columns(4)
        m[0].metric("Eval cases", rep.n); m[1].metric("Accuracy", f"{int(rep.accuracy*100)}%"); m[2].metric("ECE", f"{rep.calibration_error:.4f}", help="Lower is better"); m[3].metric("Macro F1", f"{rep.f1_macro:.2f}")
        q = st.text_input("Filter by true disease"); fc = [c for c in cases if q.lower() in c["true_disease"].lower()][:10]
        if fc: st.table([{"True disease": c["true_disease"].replace("_", " ").title(), "Domain": c["domain"].title(), "Symptoms": ", ".join(k.replace("_", " ") for k, v in c["symptoms"].items() if v is True)} for c in fc])

def _audit():
    store = get_store(); header("Audit Log", "Every orchestration step, who did it, and when. Traceability by design.", "ADMIN")
    ev = [dict(Time=time.strftime("%H:%M:%S", time.localtime(e["ts"])), Visit=f'{v["token"]}', Patient=store.patients[v["patient"]]["name"], Event=e["text"], By=e["by"], _t=e["ts"]) for v in store.visits.values() for e in v["timeline"]]
    df = pd.DataFrame(sorted(ev, key=lambda x: -x["_t"])[:80]).drop(columns=["_t"]); st.dataframe(df, width="stretch", hide_index=True)

def render(page, user):
    if page == "Command Center":
        header("Hospital Command Center", "Live view of load, capacity and congestion, and the orchestration engine's recommended fixes.", "MEDICAL DIRECTOR"); _command(); _controls()
    elif page == "Doctors & Resources": _resources()
    elif page == "Impact Scorecard": _scorecard()
    elif page == "Clinical Governance": _governance()
    elif page == "Audit Log": _audit()

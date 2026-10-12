"""AI triage chat (the star feature). Real LangGraph engine → clinical summary → crowd-aware doctor → token."""
from __future__ import annotations
import time, uuid
import streamlit as st
from .. import orchestrator as orch, triage_service as ts
from ..store import DEPTS, get_store
from ..theme import html, badge, level_badge, crowd_badge, card, header

PRESETS = ["Runny nose & cough", "Chest pain, can't breathe", "Severe flank pain", "Red itchy eyes"]
PRESET_TXT = {"Runny nose & cough": "I have a runny nose and a cough for two days", "Chest pain, can't breathe": "I have chest pain and I'm having shortness of breath",
              "Severe flank pain": "I have severe pain in my right flank and blood in my urine", "Red itchy eyes": "my eyes are red and itchy"}
GREET = "Hi, I'm MedAssist. Tell me what you're feeling in your own words (Hindi, Punjabi or English all work), when it started, and what makes it better or worse. I'll ask a couple of quick questions, then guide you to the right department and doctor."

def _k(key, name): return f"tri_{key}_{name}"
def reset(key):
    for n in ("msgs", "stage", "result", "tid", "summary", "visit", "complaint"): st.session_state.pop(_k(key, n), None)

def _stream(text):
    for w in text.split(" "):
        yield w + " "; time.sleep(0.012)

def _run(key, payload, identity):
    tid = st.session_state[_k(key, "tid")]
    with st.spinner("MedAssist is thinking…"):
        res = ts.invoke(payload, tid)
    if "__interrupt__" in res:
        text = res["__interrupt__"][0].value["question"]; st.session_state[_k(key, "stage")] = "followup"
    else:
        st.session_state[_k(key, "result")] = res; st.session_state[_k(key, "stage")] = "done"
        st.session_state[_k(key, "summary")] = ts.summarize(res, st.session_state.get(_k(key, "complaint"), ""), identity.get("age"), identity.get("gender"))
        text = "Thanks, I have what I need. Here's your assessment and the best way through the hospital right now."
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        out = st.write_stream(_stream(text))
    st.session_state[_k(key, "msgs")].append(dict(role="assistant", content=out)); st.rerun()

def _result_card(sm):
    lvl, esc = sm["level"], sm["escalated"]
    title = "Needs urgent clinical review" if (esc or lvl == 3) else (sm["predicted"] or "Assessment").replace("_", " ").title()
    conf = sm["confidence"]; conf_html = "" if esc or conf is None else f'<div class="muted" style="margin-top:.6rem">Calibrated confidence · {int(round(conf*100))}%</div><div class="bar"><div style="width:{int(conf*100)}%"></div></div>'
    tone = "red" if lvl == 3 else ""
    card(f'<div class="row"><div>{level_badge(lvl)}{badge(sm["department_raw"], "or")}</div><span class="tiny">Screening result, not a diagnosis</span></div>'
         f'<h2 style="margin:.6rem 0 0">{title}</h2>{conf_html}'
         f'<div class="disc">AI-assisted decision support only. A clinician confirms every pathway. If symptoms worsen, go to Emergency immediately.</div>', tone)
    with st.expander("Why this recommendation? (what your doctor will also see)"):
        st.markdown("**Symptoms understood:** " + (", ".join(s.replace("_", " ") for s in sm["symptoms_present"]) or "none captured"))
        if sm["symptoms_absent"]: st.markdown("**Reported absent:** " + ", ".join(s.replace("_", " ") for s in sm["symptoms_absent"]))
        for d in sm["differential"][:3]: st.markdown(f"- {str(d['name']).replace('_',' ').title()} · match {int(d['score']*100)}%")
        if sm["flags"]: st.error("Safety flags: " + "; ".join(f.get("detail", f.get("code", "")) for f in sm["flags"]))
        if sm["sources"]: st.caption("Grounded in: " + " | ".join(sm["sources"][:4]))

def _book(key, identity, source):
    store = get_store(); sm = st.session_state[_k(key, "summary")]; dest = orch.destination(sm)
    if dest["emergency"]:
        d = orch.recommend_doctor(store, "Emergency Medicine", 3)
        card(f'<div class="pill-lbl">GO NOW</div><h3 style="margin:.2rem 0">Emergency triage desk · {DEPTS["Emergency Medicine"]["loc"]}</h3>'
             f'<p class="muted" style="margin:0">{d["name"]} · {d["room"]} will see you first. {("Specialist on-call: "+dest["specialty"]+".") if dest["specialty"] else ""} Please don\'t wait in the OPD queue.</p>', "red")
        did, dept = d["id"], "Emergency Medicine"
    else:
        dept = dest["dept"]; rows = orch.rank_doctors(store, dept, sm["level"]); ds = orch.dept_status(store, dept)
        html(f'<div class="row"><div><div class="pill-lbl">RECOMMENDED DEPARTMENT</div><h3 style="margin:.1rem 0">{dept}</h3><span class="muted">{DEPTS[dept]["loc"]}</span></div><div>{crowd_badge(ds["label"])}<span class="tiny">{ds["total"]} waiting · {ds["active"]} doctors</span></div></div>')
        opts = {r["doctor"]["id"]: r for r in rows}
        def fmt(i):
            r = opts[i]; tag = "  · RECOMMENDED: shortest wait" if i == rows[0]["doctor"]["id"] else ""
            return f"{r['doctor']['name']} · {r['doctor']['room']} · {r['queue']} waiting · ~{r['wait']} min{tag}" + ("" if r["available"] else " · on break")
        did = st.radio("Choose your doctor (MedAssist checked each doctor's live queue)", list(opts), format_func=fmt, key=_k(key, "doc"))
    if st.button("Confirm and get my token", type="primary", use_container_width=True, key=_k(key, "confirm")):
        v = store.create_visit(identity["id"], sm, dept, did, source=source); st.session_state[_k(key, "visit")] = v["id"]; st.rerun()

def render(identity: dict, key: str = "p", source: str = "MedAssist app"):
    store = get_store()
    st.session_state.setdefault(_k(key, "msgs"), [dict(role="assistant", content=GREET)])
    st.session_state.setdefault(_k(key, "stage"), "start")
    vid = st.session_state.get(_k(key, "visit"))
    if vid:
        v = store.visits[vid]; d = store.doctors[v["doctor"]]
        card(f'<div class="pill-lbl">TOKEN ISSUED</div><div class="token">{v["token"]}</div><b>{v["dept"]}</b> · {d["name"]} · {d["room"]}<br><span class="muted">{DEPTS[v["dept"]]["loc"]} · ~{orch.wait_min(store, d, v["level"])} min wait · your summary is already with the doctor</span>', "green")
        c1, c2 = st.columns(2)
        if c1.button("New symptom check", use_container_width=True, key=_k(key, "new")): reset(key); st.rerun()
        c2.caption("Open **My Journey** (patient) or the queue board (staff) to follow the visit live.")
        return
    for m in st.session_state[_k(key, "msgs")]:
        with st.chat_message(m["role"], avatar=":material/auto_awesome:" if m["role"] == "assistant" else None): st.markdown(m["content"])
    if st.session_state[_k(key, "stage")] == "done":
        _result_card(st.session_state[_k(key, "summary")]); _book(key, identity, source); return
    if len(st.session_state[_k(key, "msgs")]) == 1:
        html('<p class="muted" style="text-align:center;margin:.6rem 0">Try a preset to see the engine in action:</p>')
        cols = st.columns(len(PRESETS))
        for c, p in zip(cols, PRESETS):
            if c.button(p, key=_k(key, "ch_" + p), use_container_width=True): _submit(key, PRESET_TXT[p]); st.rerun()
    txt = st.chat_input("Describe how you're feeling…" if st.session_state[_k(key, "stage")] == "start" else "Your answer…", key=_k(key, "inp"))
    if txt: _submit(key, txt); st.rerun()
    msgs = st.session_state[_k(key, "msgs")]
    if msgs and msgs[-1]["role"] == "user":
        if st.session_state[_k(key, "stage")] == "start":
            st.session_state[_k(key, "tid")] = str(uuid.uuid4()); st.session_state[_k(key, "complaint")] = msgs[-1]["content"]
            _run(key, ts.initial_state(st.session_state[_k(key, "tid")], msgs[-1]["content"], identity.get("age"), identity.get("gender")), identity)
        else:
            _run(key, ts.resume(msgs[-1]["content"]), identity)

def _submit(key, text): st.session_state[_k(key, "msgs")].append(dict(role="user", content=text))

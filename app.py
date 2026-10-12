"""
MedAssist · AI-powered hospital patient & resource orchestration (frontend prototype).
The triage engine (graph.py) is REAL. Hospital data (queues, doctors, beds) is simulated in frontend/store.py.
Run:  MEDASSIST_MOCK=1 streamlit run app.py      (omit MEDASSIST_MOCK to use live LLM keys from .env)
"""
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="MedAssist", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

from frontend import auth, theme, triage_service
from frontend.store import get_store
from frontend.views import patient, doctor, nurse, reception, admin

theme.inject()
NAV = {
    "patient": (patient, ["Home", "Symptom Check", "My Journey", "Records", "Hospital Guide"]),
    "doctor": (doctor, ["My Queue", "Consultation", "Follow-ups I planned", "Patient Search"]),
    "nurse": (nurse, ["Priority Queue", "Walk-in Triage", "Diagnostics Desk"]),
    "reception": (reception, ["Front Desk", "Live Queue", "Follow-up Check-in"]),
    "admin": (admin, ["Command Center", "Doctors & Resources", "Impact Scorecard", "Clinical Governance", "Audit Log"]),
}
ICON = {"Home": "🏠", "Symptom Check": "🩺", "My Journey": "🧭", "Records": "📋", "Hospital Guide": "🗺️", "My Queue": "👥", "Consultation": "📝", "Follow-ups I planned": "📅", "Patient Search": "🔎",
        "Priority Queue": "🚨", "Walk-in Triage": "🚶", "Diagnostics Desk": "🧪", "Front Desk": "🛎️", "Live Queue": "⏱️", "Follow-up Check-in": "✅",
        "Command Center": "📊", "Doctors & Resources": "🏥", "Impact Scorecard": "🎯", "Clinical Governance": "📖", "Audit Log": "🧾"}
ROLE_LABEL = {"patient": "Patient", "doctor": "Doctor", "nurse": "Triage Nurse", "reception": "Reception", "admin": "Medical Director"}

user = st.session_state.get("user")
if not user:
    auth.render(); st.stop()

mod, pages = NAV[user["role"]]
if st.session_state.get("nav") not in pages: st.session_state.nav = pages[0]
if "nav_request" in st.session_state:
    req = st.session_state.pop("nav_request")
    if req in pages: st.session_state.nav = req

with st.sidebar:
    theme.html('<div style="padding:.6rem 0 .2rem"><span style="font-family:Fraunces,serif;font-size:1.7rem;font-weight:700">MedAssist</span><div class="tiny">Guiding every patient · Orchestrating every pathway</div></div>')
    theme.html(f'<div class="card" style="padding:.7rem .9rem;margin:.6rem 0"><div class="pill-lbl">{ROLE_LABEL[user["role"]]}</div><b>{user["name"]}</b></div>')
    st.radio("Navigate", pages, key="nav", format_func=lambda p: f"{ICON.get(p, '•')}  {p}", label_visibility="collapsed")
    theme.html(f'<br><div class="tiny">Triage engine: <b>{triage_service.engine_mode()}</b><br>Hospital data: simulated</div>')
    if st.button("Sign out", use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

mod.render(st.session_state.nav, user)

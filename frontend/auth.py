"""Mock authentication. Patients: phone + OTP 123456. Staff: username + password demo123 (or one-click demo accounts)."""
from __future__ import annotations
import streamlit as st
from .store import get_store
from .theme import html, card, header

OTP, PASSWORD = "123456", "demo123"

def _login(u: dict): st.session_state.user = u; st.session_state.pop("otp_sent", None); st.rerun()

def render():
    store = get_store(); accounts = store.staff_accounts()
    html('<span class="eyebrow">AI CARE ORCHESTRATION</span>')
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        html('<div class="hero">Guiding every patient.<br><span class="a">Orchestrating every pathway.</span></div><p class="muted" style="font-size:1.05rem;margin:1rem 0 1.4rem">MedAssist understands what a patient is feeling in their own language, applies hard safety rules, then routes them to the right department and the right doctor by checking the live crowd. Doctors get the summary before the patient walks in.</p>')
        c = st.columns(4)
        for col, (n, t, d) in zip(c, [("01", "Understand", "Hindi · Punjabi · English"), ("02", "Protect", "Safety rules + guidelines"), ("03", "Orchestrate", "Dept · doctor · tests"), ("04", "Deliver", "Patient + staff views")]):
            col.markdown(f'<div class="card" style="padding:.9rem"><div class="pill-lbl">{n}</div><b>{t}</b><div class="tiny">{d}</div></div>', unsafe_allow_html=True)
        html('<div class="card hl" style="margin-top:.4rem"><b>Demo guide.</b> Patient: phone <code>9876500001</code>, OTP <code>123456</code>. Staff: pick a role on the right (password <code>demo123</code>). Open two browser windows, one as patient and one as doctor, to watch the same visit move live.</div>')
    with right:
        tp, ts_ = st.tabs(["Patient", "Staff"])
        with tp:
            phone = st.text_input("Mobile number", value=st.session_state.get("lp", ""), placeholder="98765 00001", key="lp")
            p = store.find_patient_by_phone(phone.strip()) if phone else None
            if st.button("Send OTP", use_container_width=True, disabled=not phone): st.session_state.otp_sent = True
            if st.session_state.get("otp_sent"):
                st.caption("Demo OTP: 123456")
                code = st.text_input("Enter OTP", max_chars=6, key="lotp")
                if not p:
                    st.info("New here? Create your profile.")
                    c = st.columns(3); nm = c[0].text_input("Name", key="rn"); ag = c[1].number_input("Age", 0, 110, 30, key="ra"); gn = c[2].selectbox("Gender", ["M", "F", "O"], key="rg")
                if st.button("Verify and enter", type="primary", use_container_width=True):
                    if code != OTP: st.error("Incorrect OTP.")
                    else:
                        if not p:
                            if not nm: st.error("Please enter your name."); return
                            p = store.create_patient(nm, ag, gn, phone.strip())
                        _login(dict(role="patient", id=p["id"], name=p["name"]))
            if st.button("Use demo patient (Gurpreet Singh)", use_container_width=True): _login(dict(role="patient", id=store.find_patient_by_phone("9876500001")["id"], name="Gurpreet Singh"))
        with ts_:
            u = st.text_input("Staff ID", placeholder="dr.sharma · nurse.manjeet · desk.front · admin", key="su"); pw = st.text_input("Password", type="password", key="sp")
            if st.button("Sign in", type="primary", use_container_width=True):
                a = accounts.get(u.strip().lower())
                if a and pw == PASSWORD: _login(dict(a))
                else: st.error("Invalid ID or password (demo password: demo123).")
            st.markdown("**One-click demo roles**")
            doc = st.selectbox("Doctor", [k for k, a in accounts.items() if a["role"] == "doctor"], format_func=lambda k: f'{accounts[k]["name"]} · {store.doctors[accounts[k]["id"]]["dept"]}', index=0, label_visibility="collapsed")
            c = st.columns(2)
            if c[0].button("Enter as Doctor", use_container_width=True): _login(dict(accounts[doc]))
            if c[1].button("Enter as Triage Nurse", use_container_width=True): _login(dict(accounts["nurse.manjeet"]))
            if c[0].button("Enter as Reception", use_container_width=True): _login(dict(accounts["desk.front"]))
            if c[1].button("Enter as Medical Director", use_container_width=True): _login(dict(accounts["admin"]))

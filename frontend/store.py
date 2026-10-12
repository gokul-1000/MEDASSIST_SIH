"""
In-memory hospital store (MOCK DATA). One shared singleton per server process, so a patient
on a phone and a doctor on a laptop see the same live state during the demo.
Swap this class for real DB/API calls later; the views only talk to these methods.
"""
from __future__ import annotations
import random, re, threading, time, datetime as dt
import streamlit as st

DEPTS = {
    "Emergency Medicine": dict(loc="Block A · Gate 2, Ground floor", code="E", emergency=True),
    "General Medicine": dict(loc="Block B · OPD Rooms 10-16", code="G"),
    "Cardiology": dict(loc="Block C · Rooms 21-22", code="C"),
    "Neurology": dict(loc="Block C · Rooms 31-32", code="N"),
    "Urology": dict(loc="Block D · Room 41", code="U"),
    "General Surgery": dict(loc="Block D · Rooms 45-46", code="S"),
    "Ophthalmology": dict(loc="Block E · Rooms 51-52", code="O"),
}
DIAG = {  # diagnostics + pharmacy are queues, not consultations
    "Laboratory": dict(loc="Block A · Sample collection, Ground floor", base=9, avg=2.0),
    "Radiology": dict(loc="Block A · Imaging hall, Ground floor", base=14, avg=2.3),
    "Pharmacy": dict(loc="Block A · Counter 3", base=11, avg=1.2),
}
TESTS = {"Blood test (CBC)": "Laboratory", "Blood sugar": "Laboratory", "Lipid profile": "Laboratory", "Urine routine": "Laboratory",
         "ECG": "Radiology", "Chest X-ray": "Radiology", "Ultrasound abdomen": "Radiology", "CT scan": "Radiology", "MRI brain": "Radiology"}

#  id, name, dept, room, avg consult min, background queue, status
DOCTORS = [
    ("er1", "Dr. Anita Verma", "Emergency Medicine", "ER Bay 1", 8, 3, "available"),
    ("er2", "Dr. Rohan Mehta", "Emergency Medicine", "ER Bay 2", 8, 4, "available"),
    ("gm1", "Dr. S. Kapoor", "General Medicine", "Room 12", 7, 9, "available"),
    ("gm2", "Dr. Meena Iyer", "General Medicine", "Room 14", 7, 6, "available"),
    ("gm3", "Dr. Harpreet Gill", "General Medicine", "Room 16", 7, 12, "available"),
    ("gm4", "Dr. Vikram Rao", "General Medicine", "Room 10", 7, 8, "break"),
    ("ca1", "Dr. Arvind Sharma", "Cardiology", "Room 21", 12, 11, "available"),
    ("ca2", "Dr. Neha Bansal", "Cardiology", "Room 22", 12, 10, "available"),
    ("ne1", "Dr. Kavita Nair", "Neurology", "Room 31", 12, 4, "available"),
    ("ne2", "Dr. Imran Qureshi", "Neurology", "Room 32", 12, 3, "available"),
    ("ur1", "Dr. Prakash Joshi", "Urology", "Room 41", 10, 5, "available"),
    ("su1", "Dr. Sandeep Dhillon", "General Surgery", "Room 45", 10, 4, "available"),
    ("su2", "Dr. Lata Menon", "General Surgery", "Room 46", 10, 2, "available"),
    ("op1", "Dr. Simran Kaur", "Ophthalmology", "Room 51", 9, 6, "available"),
    ("op2", "Dr. Anil Khanna", "Ophthalmology", "Room 52", 9, 3, "available"),
]
BEDS = {"ICU": (20, 17), "Emergency bays": (24, 15), "General ward": (180, 141), "Cardiac care": (16, 13)}

T = lambda c, pres, absn, pred, diff, conf, lvl, dept, flags=(), esc=False, fu=(): dict(
    complaint=c, symptoms_present=list(pres), symptoms_absent=list(absn), followups=list(fu), predicted=pred,
    differential=[dict(name=n, score=s) for n, s in diff], confidence=conf, level=lvl, department_raw=dept,
    flags=list(flags), escalated=esc, escalation_reason="Critical red-flag pattern" if esc else None,
    sources=["General Medicine KB v1.0"], domain="general")
TEMPLATES = [
    T("Runny nose and cough for 2 days", ["runny_nose", "cough", "sore_throat"], ["chest_pain", "shortness_of_breath"], "common_cold", [("common_cold", .71), ("influenza", .18), ("allergic_rhinitis", .08)], .78, 1, "General Medicine", fu=[dict(q="Do you have fever?", a="No fever")]),
    T("Chest pain with breathlessness since morning", ["chest_pain", "shortness_of_breath", "sweating"], [], "acute_coronary_syndrome", [("acute_coronary_syndrome", .62), ("unstable_angina", .21)], None, 3, "Cardiology (Emergency)", flags=[dict(code="RED_FLAG_CHEST_PAIN_DYSPNEA", detail="Chest pain + shortness of breath", severity="critical")], esc=True),
    T("Severe right flank pain spreading to groin", ["flank_pain", "nausea", "blood_in_urine"], ["fever"], "kidney_stone", [("kidney_stone", .74), ("uti", .15)], .74, 2, "Urology", fu=[dict(q="Any blood in urine?", a="Yes, since yesterday")]),
    T("Red, itchy and watery eyes for 3 days", ["eye_redness", "itching", "eye_discharge"], ["vision_loss_sudden"], "conjunctivitis", [("conjunctivitis", .81), ("allergic_conjunctivitis", .12)], .81, 1, "Ophthalmology"),
    T("Sudden severe headache with vomiting", ["headache_severe", "vomiting", "photophobia"], ["weakness_one_side"], "migraine", [("migraine", .58), ("tension_headache", .2)], .58, 2, "Neurology", fu=[dict(q="Is this the worst headache of your life?", a="Not sure")]),
    T("Lower right abdominal pain with fever", ["abdominal_pain", "fever", "nausea"], ["vomiting_blood"], "appendicitis", [("appendicitis", .66), ("gastroenteritis", .2)], .66, 2, "General Surgery (Emergency)", flags=[dict(code="ABDOMEN_FEVER", detail="Right iliac fossa pain with fever", severity="warning")]),
    T("Fever and body ache for 3 days", ["fever", "body_ache", "fatigue"], ["rash"], "influenza", [("influenza", .64), ("viral_fever", .22)], .69, 1, "General Medicine"),
    T("Palpitations and chest tightness on climbing stairs", ["palpitations", "chest_tightness", "fatigue"], ["fainting"], "stable_angina", [("stable_angina", .61), ("arrhythmia", .24)], .61, 2, "Cardiology", fu=[dict(q="Does the tightness settle with rest?", a="Yes, in a few minutes")]),
    T("High BP readings at home with morning headache", ["headache", "dizziness", "high_bp_reading"], ["chest_pain"], "hypertension", [("hypertension", .72), ("anxiety", .1)], .72, 1, "Cardiology"),
]
SEED_NAMES = [("Gurpreet Singh", 62, "M", "9876500001"), ("Simran Kaur", 34, "F", "9876500002"), ("Rajesh Kumar", 51, "M", "9876500003"), ("Anjali Sharma", 28, "F", "9876500004"),
    ("Mohan Lal", 67, "M", "9876500005"), ("Pooja Verma", 41, "F", "9876500006"), ("Balwinder Singh", 58, "M", "9876500007"), ("Neha Gupta", 25, "F", "9876500008"),
    ("Harjit Kaur", 70, "F", "9876500009"), ("Amit Chopra", 45, "M", "9876500010"), ("Ravi Teja", 36, "M", "9876500011"), ("Sunita Devi", 55, "F", "9876500012")]

def base_dept(raw: str) -> str:
    return re.sub(r"\s*\(Emergency\)\s*", "", raw or "").strip() or "General Medicine"
def is_emergency_dept(raw: str) -> bool:
    return "(Emergency)" in (raw or "")

class HospitalStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.rng = random.Random(11)
        self.doctors = {d[0]: dict(id=d[0], name=d[1], dept=d[2], room=d[3], avg=d[4], base=d[5], status=d[6]) for d in DOCTORS}
        self.diag = {k: dict(v) for k, v in DIAG.items()}
        self.beds = {k: dict(total=v[0], occ=v[1]) for k, v in BEDS.items()}
        self.inflow = [12, 34, 52, 47, 38, 22]
        self.patients, self.visits, self.alerts_log = {}, {}, []
        self.tok = {d["code"]: 30 for d in DEPTS.values()}
        self._vid = 0
        self._seed()

    # ---------- users ----------
    def staff_accounts(self):
        acc = {f"dr.{d['name'].split()[-1].lower()}": dict(role="doctor", id=d["id"], name=d["name"]) for d in self.doctors.values()}
        acc.update({"nurse.manjeet": dict(role="nurse", id="n1", name="Sr. Manjeet Kaur"), "desk.front": dict(role="reception", id="r1", name="Front Desk · Counter 2"),
                    "admin": dict(role="admin", id="a1", name="Dr. R. Malhotra · Medical Director")})
        return acc
    def find_patient_by_phone(self, phone):
        return next((p for p in self.patients.values() if p["phone"] == phone), None)
    def create_patient(self, name, age, gender, phone):
        with self.lock:
            pid = f"P{len(self.patients) + 1001}"
            self.patients[pid] = dict(id=pid, name=name, age=int(age), gender=gender, phone=phone, abha=f"91-{self.rng.randint(1000,9999)}-{self.rng.randint(1000,9999)}-{self.rng.randint(1000,9999)}")
            return self.patients[pid]

    # ---------- helpers ----------
    def _ev(self, v, text, by="System"):
        v["timeline"].append(dict(ts=time.time(), text=text, by=by))
    def _token(self, dept):
        code = DEPTS[dept]["code"]; self.tok[code] += 1
        return f"{code}-{self.tok[code]:03d}"
    def waiting(self, **flt):
        out = [v for v in self.visits.values() if v["status"] in ("waiting", "review_waiting")]
        for k, val in flt.items():
            out = [v for v in out if v.get(k) == val]
        return sorted(out, key=lambda v: (-v["level"], v["status"] != "review_waiting", v["created"]))
    def doctor_queue(self, did):
        return self.waiting(doctor=did)
    def current(self, did):
        return next((v for v in self.visits.values() if v["doctor"] == did and v["status"] == "in_consult"), None)
    def active_visit(self, pid):
        mine = [v for v in self.visits.values() if v["patient"] == pid and (v["status"] != "completed" or (v["rx"] and not v.get("rx_done") and time.time() - v["created"] < 86400))]
        return sorted(mine, key=lambda v: v["created"])[-1] if mine else None
    def history(self, pid):
        return sorted([v for v in self.visits.values() if v["patient"] == pid and v["status"] == "completed"], key=lambda v: v["created"], reverse=True)

    # ---------- mutations ----------
    def create_visit(self, pid, triage, dept, doctor_id, source="MedAssist app", level=None):
        with self.lock:
            self._vid += 1; vid = f"V{self._vid:04d}"
            lvl = level or triage.get("level", 1)
            v = dict(id=vid, patient=pid, token=self._token(dept), dept=dept, doctor=doctor_id, level=lvl, status="waiting", created=time.time(),
                     triage=triage, vitals={}, notes="", diagnosis="", rx=[], orders=[], next_visit=None, timeline=[], source=source, specialty_after=base_dept(triage.get("department_raw", "")) if dept == "Emergency Medicine" else None)
            self.visits[vid] = v
            self._ev(v, f"Symptom check completed via {source}", "MedAssist")
            self._ev(v, f"Token {v['token']} issued · {dept} · {self.doctors[doctor_id]['name']}", "MedAssist")
            if lvl == 3: self._ev(v, "PRIORITY FLAG raised, nurse notified", "Safety rules")
            self.inflow[-1] += 1
            return v
    def call_next(self, did):
        with self.lock:
            cur = self.current(did)
            if cur: return cur
            q = self.doctor_queue(did)
            if not q: return None
            v = q[0]; v["status"] = "in_consult"; self._ev(v, f"Called in by {self.doctors[did]['name']}", self.doctors[did]["name"]); return v
    def order_tests(self, vid, names, by):
        with self.lock:
            v = self.visits[vid]
            for n in names:
                v["orders"].append(dict(id=f"T{len(v['orders'])+1}", name=n, dept=TESTS[n], status="pending", result=""))
                self._ev(v, f"Ordered {n} → {TESTS[n]}", by)
    def mark_test_done(self, vid, oid, result):
        with self.lock:
            v = self.visits[vid]
            for o in v["orders"]:
                if o["id"] == oid: o["status"], o["result"] = "done", result; self._ev(v, f"{o['name']} resulted", o["dept"])
            if v["status"] == "awaiting_tests" and all(o["status"] == "done" for o in v["orders"]):
                v["status"] = "review_waiting"; self._ev(v, "All results ready, back in doctor queue (priority)", "MedAssist")
    def complete_consult(self, vid, notes, diagnosis, rx, next_visit, by):
        with self.lock:
            v = self.visits[vid]; v.update(notes=notes, diagnosis=diagnosis, rx=rx, next_visit=next_visit)
            pend = [o for o in v["orders"] if o["status"] == "pending"]
            doc = self.doctors[v["doctor"]]; doc["base"] = max(0, doc["base"] - 1)
            if pend:
                v["status"] = "awaiting_tests"; self._ev(v, f"Sent for {', '.join(o['name'] for o in pend)}", by)
            else:
                v["status"] = "completed"; self._ev(v, "Consultation completed", by)
            if rx: self._ev(v, f"{len(rx)} medicine(s) prescribed → Pharmacy", by)
            if next_visit: self._ev(v, f"Next visit planned: {next_visit['date']} · {next_visit['dept']}", by)
    def refer(self, vid, to_dept, reason, by):
        from . import orchestrator as orch
        with self.lock:
            v = self.visits[vid]; best = orch.recommend_doctor(self, to_dept)
            v.update(dept=to_dept, doctor=best["id"], status="waiting", token=self._token(to_dept)); self._ev(v, f"Referred to {to_dept} ({reason}) → {best['name']} · new token {v['token']}", by)
    def set_priority(self, vid, level, by):
        with self.lock:
            v = self.visits[vid]; self._ev(v, f"Priority changed {v['level']} → {level}", by); v["level"] = level
    def add_vitals(self, vid, vitals, by):
        with self.lock:
            v = self.visits[vid]; v["vitals"] = vitals; self._ev(v, "Vitals recorded: " + ", ".join(f"{k} {x}" for k, x in vitals.items()), by)
    def reassign(self, vid, did, by):
        with self.lock:
            v = self.visits[vid]; v["doctor"] = did; self._ev(v, f"Reassigned to {self.doctors[did]['name']}", by)
    def set_doctor_status(self, did, status):
        self.doctors[did]["status"] = status
    def tick(self):
        """Simulate ~10 minutes of hospital life: background queues move, new arrivals appear."""
        with self.lock:
            for d in self.doctors.values():
                if d["status"] == "available": d["base"] = max(0, d["base"] - self.rng.choice([0, 1, 1, 2]) + self.rng.choice([0, 0, 1, 2]))
            for q in self.diag.values(): q["base"] = max(2, q["base"] + self.rng.choice([-2, -1, 0, 1, 2]))
            self.inflow = (self.inflow + [max(8, self.inflow[-1] + self.rng.randint(-12, 14))])[-8:]
    def open_oncall(self, dept):
        with self.lock:
            docs = [d for d in self.doctors.values() if d["dept"] == dept]; oid = f"oc_{DEPTS[dept]['code']}"
            taken = 0
            for d in docs:
                k = min(4, d["base"] // 2); d["base"] -= k; taken += k
            self.doctors[oid] = dict(id=oid, name=f"Dr. On-call ({dept})", dept=dept, room="OPD Annex", avg=docs[0]["avg"], base=taken, status="available")
            return self.doctors[oid]
    def surge(self, dept, n=8):
        with self.lock:
            docs = [d for d in self.doctors.values() if d["dept"] == dept and d["status"] == "available"]
            for i in range(n): docs[i % len(docs)]["base"] += 1
            self.inflow[-1] += n

    # ---------- seed ----------
    def _seed(self):
        for n, a, g, ph in SEED_NAMES: self.create_patient(n, a, g, ph)
        pids = list(self.patients)
        plan = [("ca1", 7, "waiting", 4), ("ca2", 8, "in_consult", 22), ("er1", 1, "waiting", 3), ("gm2", 0, "waiting", 14), ("gm1", 6, "waiting", 31), ("ur1", 2, "waiting", 18), ("op1", 3, "waiting", 9),
                ("ne1", 4, "awaiting_tests", 45), ("su1", 5, "waiting", 12), ("gm3", 6, "review_waiting", 70), ("gm2", 0, "waiting", 25), ("op2", 3, "waiting", 6)]
        for i, (did, ti, status, mins) in enumerate(plan):
            tr = TEMPLATES[ti]; dept = self.doctors[did]["dept"]
            if tr["level"] == 3: dept, did = "Emergency Medicine", "er1"
            p = self.patients[pids[1 + (i % (len(pids) - 1))]]
            v = self.create_visit(p["id"], dict(tr, age=p["age"], gender=p["gender"]), dept, did, source="MedAssist app" if i % 3 else "Reception kiosk")
            v["created"] = time.time() - mins * 60; v["status"] = status
            if status == "awaiting_tests": v["orders"] = [dict(id="T1", name="MRI brain", dept="Radiology", status="pending", result="")]
            if status == "review_waiting": v["orders"] = [dict(id="T1", name="Blood test (CBC)", dept="Laboratory", status="done", result="Hb 12.8 g/dL, WBC 7.2, normal")]
        # demo patient history + planned follow-up
        g = self.patients[pids[0]]
        h = self.create_visit(g["id"], dict(TEMPLATES[0], complaint="Blood pressure check and mild dizziness", department_raw="General Medicine", age=62, gender="M"), "Cardiology", "ca1", source="Reception kiosk")
        h.update(status="completed", created=time.time() - 14 * 86400, diagnosis="Stage-1 hypertension", notes="BP 148/92. Start amlodipine, low-salt diet, review in 2 weeks.",
                 rx=[dict(name="Amlodipine 5 mg", dose="1 tab once daily", days=30)], orders=[dict(id="T1", name="ECG", dept="Radiology", status="done", result="Normal sinus rhythm")],
                 next_visit=dict(date=str(dt.date.today() + dt.timedelta(days=2)), dept="Cardiology", doctor="Dr. Arvind Sharma", purpose="BP review and medication check", instructions="Bring home BP readings. Fasting not required."))
        h["timeline"] = [dict(ts=h["created"], text="Consultation completed", by="Dr. Arvind Sharma")]

@st.cache_resource(show_spinner=False)
def get_store() -> HospitalStore:
    return HospitalStore()

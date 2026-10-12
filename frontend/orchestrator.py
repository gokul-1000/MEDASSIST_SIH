"""Mock orchestration engine: crowd-aware doctor choice, wait estimates, route plans, alerts, rebalancing."""
from __future__ import annotations
import time
from .store import DEPTS, DIAG, base_dept, is_emergency_dept

def crowd_label(per_doc: float) -> str:
    return "Congested" if per_doc >= 9 else "Busy" if per_doc >= 6 else "Moderate" if per_doc >= 3 else "Low"
def crowd_tone(label): return {"Low": "green", "Moderate": "amber", "Busy": "amber", "Congested": "red"}[label]

def queue_len(store, d) -> int:
    return d["base"] + len(store.doctor_queue(d["id"]))
def wait_min(store, d, level=1) -> int:
    return 3 if level == 3 else int(round(queue_len(store, d) * d["avg"]))

def rank_doctors(store, dept, level=1):
    rows = []
    for d in store.doctors.values():
        if d["dept"] != dept: continue
        rows.append(dict(doctor=d, queue=queue_len(store, d), wait=wait_min(store, d, level), available=d["status"] == "available"))
    return sorted(rows, key=lambda r: (not r["available"], r["wait"]))
def recommend_doctor(store, dept, level=1):
    r = rank_doctors(store, dept, level); return r[0]["doctor"]

def dept_status(store, dept):
    docs = [d for d in store.doctors.values() if d["dept"] == dept]
    act = [d for d in docs if d["status"] == "available"] or docs
    total = sum(queue_len(store, d) for d in docs); per = total / max(1, len(act)); lab = crowd_label(per)
    return dict(dept=dept, total=total, doctors=len(docs), active=len(act), per_doc=round(per, 1), label=lab, tone=crowd_tone(lab),
                avg_wait=int(round(sum(wait_min(store, d) for d in act) / max(1, len(act)))), high=len([v for v in store.waiting(dept=dept) if v["level"] >= 2]))
def diag_status(store, name):
    q = store.diag[name]; return dict(name=name, queue=q["base"], wait=int(round(q["base"] * q["avg"])), loc=q["loc"])

def destination(triage):
    """Triage output → where the patient physically goes first."""
    raw = triage.get("department_raw") or "General Medicine"
    if is_emergency_dept(raw) or triage.get("level") == 3:
        return dict(dept="Emergency Medicine", specialty=base_dept(raw), emergency=True)
    d = base_dept(raw)
    return dict(dept=d if d in DEPTS else "General Medicine", specialty=None, emergency=False)

def route_plan(store, v):
    d = store.doctors[v["doctor"]]; loc = DEPTS[v["dept"]]["loc"]; st = v["status"]
    ahead = len([x for x in store.doctor_queue(d["id"]) if x["id"] != v["id"] and (x["level"], -x["created"]) >= (v["level"], -v["created"])]) + d["base"]
    orders = v["orders"]; pend = [o for o in orders if o["status"] == "pending"]
    stage = {"waiting": 1, "in_consult": 2, "awaiting_tests": 3, "review_waiting": 4, "completed": 5}[st]
    def S(i, title, detail):
        return dict(title=title, detail=detail, state="done" if i < stage else "now" if i == stage else "next")
    steps = [S(0, "Token issued & checked in", f"Token {v['token']} · triage summary shared with the doctor"),
             S(1, f"Go to {v['dept']}", f"{loc} · {d['room']} · {ahead} ahead · ~{wait_min(store, d, v['level'])} min" if st == "waiting" else f"{loc} · {d['room']}"),
             S(2, f"Consultation · {d['name']}", "Doctor reviews your symptom summary first, so you don't repeat yourself")]
    if orders:
        det = "; ".join(f"{o['name']} → {o['dept']} (~{diag_status(store, o['dept'])['wait']} min)" if o["status"] == "pending" else f"{o['name']} done" for o in orders)
        steps += [S(3, "Tests", det), S(4, "Review with doctor", "Results go straight to your doctor")]
    if v["rx"]:
        pharm = diag_status(store, "Pharmacy"); done = v.get("rx_done")
        steps.append(dict(title="Collect medicines", detail=f"Pharmacy · {pharm['loc'].split('·')[-1].strip()} · ~{pharm['wait']} min" if not done else "Collected", state="done" if done else ("now" if st == "completed" else "next")))
    steps.append(dict(title="Next visit / discharge", detail=(f"{v['next_visit']['date']} · {v['next_visit']['dept']}" if v["next_visit"] else "Nothing further planned"), state="done" if st == "completed" and (v.get("rx_done") or not v["rx"]) else "next"))
    # when there are no orders, steps 3/4 absent so remap states by status
    if not orders:
        for i, s in enumerate(steps[:3]):
            s["state"] = "done" if i < min(stage, 2) else "now" if i == min(stage, 2) and st != "completed" else ("done" if st == "completed" else "next")
    return steps

def alerts(store):
    out = []
    for dept in DEPTS:
        s = dept_status(store, dept)
        if s["label"] == "Congested":
            alt = rank_doctors(store, dept)
            out.append(dict(kind="red", dept=dept, title=f"Congestion detected in {dept}", detail=f"{s['total']} waiting, {s['active']} doctors available: ~{s['per_doc']} per doctor.",
                            action=f"Rebalance load across {dept} doctors and route suitable non-emergency cases to the least-loaded doctor ({alt[0]['doctor']['name']}).", rebalance=dept))
    for k, b in store.beds.items():
        pct = b["occ"] / b["total"]
        if pct >= .85: out.append(dict(kind="amber", dept=k, title=f"{k} at {int(pct*100)}% occupancy", detail=f"{b['total']-b['occ']} beds free of {b['total']}.", action="Review discharges and transfers.", rebalance=None))
    for n in DIAG:
        d = diag_status(store, n)
        if d["wait"] >= 30: out.append(dict(kind="amber", dept=n, title=f"{n} wait is {d['wait']} min", detail=f"{d['queue']} in queue.", action="Open a second counter or prioritise urgent orders.", rebalance=None))
    crit = [v for v in store.visits.values() if v["level"] == 3 and v["status"] in ("waiting", "review_waiting")]
    if crit: out.insert(0, dict(kind="red", dept="Emergency Medicine", title=f"{len(crit)} emergency-priority patient(s) waiting", detail=", ".join(v["token"] for v in crit), action="Nurse desk to confirm priority.", rebalance=None))
    return out

def rebalance(store, dept, by="Admin"):
    moves = []
    with store.lock:
        for _ in range(6):
            r = rank_doctors(store, dept)
            if len(r) < 2: break
            hi = max((x for x in r if x["available"]), key=lambda x: x["queue"], default=None); lo = min((x for x in r if x["available"]), key=lambda x: x["queue"], default=None)
            if not hi or not lo or hi["queue"] - lo["queue"] <= 2: break
            cand = [v for v in store.doctor_queue(hi["doctor"]["id"]) if v["level"] < 3]
            if cand:
                v = cand[-1]; store.reassign(v["id"], lo["doctor"]["id"], by); moves.append(f"{v['token']} → {lo['doctor']['name']}")
            else:
                hi["doctor"]["base"] -= 1; lo["doctor"]["base"] += 1; moves.append(f"1 walk-in → {lo['doctor']['name']}")
    s = dept_status(store, dept)
    if s["per_doc"] >= 9 and not any(d["id"].startswith("oc_") and d["dept"] == dept for d in store.doctors.values()):
        oc = store.open_oncall(dept); moves.append(f"Opened extra counter: {oc['name']} ({oc['room']}), took over part of the walk-in load")
    return moves

def overview(store):
    vs = list(store.visits.values()); live = [v for v in vs if v["status"] != "completed"]
    waits = [dept_status(store, d)["avg_wait"] for d in DEPTS]
    return dict(today=sum(store.inflow) + len(vs), in_system=len(live), avg_wait=int(sum(waits) / len(waits)), flagged=len([v for v in vs if v["level"] == 3]),
                triaged=len(vs), icu_free=store.beds["ICU"]["total"] - store.beds["ICU"]["occ"], docs=len([d for d in store.doctors.values() if d["status"] == "available"]), docs_total=len(store.doctors))

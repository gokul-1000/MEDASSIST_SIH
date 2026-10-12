"""Warm MedAssist theme (matches the pitch deck) + small HTML component helpers."""
from __future__ import annotations
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#FFFAF5;--panel:#FFFFFF;--ink:#3B2A24;--mu:#7A6A62;--or:#E8742F;--od:#C95A1B;--bd:#F0DCCB;--pe:#FDEBDD;--pe2:#FFF3EA;--gr:#2F7D52;--gl:#E4F0E8;--rd:#C1121F;--rl:#FCE4E4;--am:#B26A00;--al:#FEF3D6;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg)!important;color:var(--ink);}
#MainMenu,footer{visibility:hidden;} header[data-testid="stHeader"]{background:transparent!important;}
.block-container{padding:1.6rem 2.6rem 3rem!important;max-width:1320px;}
h1,h2,h3,h4{font-family:'Fraunces',serif!important;color:var(--ink)!important;letter-spacing:-.01em;}
/* sidebar */
section[data-testid="stSidebar"]{background:#FFF3EA!important;border-right:1px solid var(--bd);}
section[data-testid="stSidebar"] *{color:var(--ink);}
section[data-testid="stSidebar"] div[role="radiogroup"] label{padding:.55rem .8rem;border-radius:10px;margin-bottom:2px;transition:all .15s;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:rgba(232,116,47,.08);}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:var(--pe);border-left:3px solid var(--or);font-weight:600;}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{display:none;}
/* buttons */
.stButton>button,.stFormSubmitButton>button{border-radius:10px!important;font-weight:600!important;padding:.5rem 1.1rem!important;border:1px solid var(--bd)!important;background:#fff!important;color:var(--ink)!important;box-shadow:none!important;transition:all .15s;}
.stButton>button:hover,.stFormSubmitButton>button:hover{border-color:var(--or)!important;color:var(--od)!important;background:var(--pe2)!important;}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"],button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButtonPrimary"]{background:var(--or)!important;color:#fff!important;border-color:var(--or)!important;box-shadow:0 4px 12px rgba(232,116,47,.22)!important;}
.stButton>button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{background:var(--od)!important;color:#fff!important;}
/* inputs */
div[data-baseweb="input"],div[data-baseweb="select"]>div,div[data-baseweb="textarea"]{border-radius:10px!important;background:#fff!important;}
/* tabs */
button[data-baseweb="tab"]{font-weight:600;color:var(--mu);} button[data-baseweb="tab"][aria-selected="true"]{color:var(--od);}
div[data-baseweb="tab-highlight"]{background:var(--or)!important;}
/* chat */
div[data-testid="stChatMessage"]{background:#fff!important;border:1px solid var(--bd)!important;border-radius:16px!important;padding:1rem 1.25rem!important;margin-bottom:.8rem!important;animation:fi .35s ease;}
div[data-testid="stChatInput"] textarea{border-radius:14px!important;border:1px solid var(--bd)!important;}
div[data-testid="stChatInput"] textarea:focus{border-color:var(--or)!important;box-shadow:0 0 0 3px rgba(232,116,47,.15)!important;}
@keyframes fi{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
/* components */
.eyebrow{display:inline-block;background:var(--pe);color:var(--od);font-weight:700;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;padding:.3rem .8rem;border-radius:999px;margin-bottom:.6rem;}
.ph h1{font-size:2rem;margin:0 0 .2rem;} .ph p{color:var(--mu);margin:0 0 1.2rem;font-size:.98rem;}
.card{background:#fff;border:1px solid var(--bd);border-radius:16px;padding:1.2rem 1.3rem;margin-bottom:.9rem;transition:box-shadow .2s,transform .2s;}
.card:hover{box-shadow:0 8px 22px rgba(59,42,36,.05);}
.card.hl{border-color:var(--or);background:var(--pe2);} .card.dark{background:#3B2A24;color:#FFF3EA;border-color:#3B2A24;} .card.dark h1,.card.dark h2,.card.dark h3,.card.dark h4{color:#fff!important;}
.card.red{border-color:#F3B4B4;background:#FFF5F5;} .card.green{border-color:#BFDCCB;background:#F5FBF7;}
.kpi{background:#fff;border:1px solid var(--bd);border-radius:14px;padding:1rem 1.15rem;} .kpi .l{font-size:.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--mu);}
.kpi .v{font-family:'Fraunces',serif;font-size:1.9rem;font-weight:600;color:var(--ink);line-height:1.15;margin-top:.15rem;} .kpi .s{font-size:.78rem;color:var(--mu);margin-top:.1rem;} .kpi.warn .v{color:var(--od);} .kpi.bad .v{color:var(--rd);} .kpi.good .v{color:var(--gr);}
.badge{display:inline-block;padding:.22rem .65rem;border-radius:999px;font-size:.68rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;margin-right:.3rem;}
.b-1,.b-green{background:var(--gl);color:var(--gr);} .b-2,.b-amber{background:var(--al);color:var(--am);} .b-3,.b-red{background:var(--rl);color:var(--rd);} .b-or{background:var(--pe);color:var(--od);} .b-gray{background:#F1ECE7;color:var(--mu);}
.bar{height:8px;border-radius:99px;background:#F4EBE2;overflow:hidden;margin:.35rem 0;} .bar>div{height:100%;border-radius:99px;background:var(--or);} .bar.red>div{background:var(--rd);} .bar.amber>div{background:#E0A100;} .bar.green>div{background:#4FA374;}
.muted{color:var(--mu);font-size:.85rem;} .tiny{color:var(--mu);font-size:.74rem;} .row{display:flex;gap:.8rem;align-items:center;justify-content:space-between;}
.token{font-family:'Fraunces',serif;font-size:2.6rem;font-weight:700;color:var(--od);letter-spacing:.04em;}
.step{display:flex;gap:.8rem;align-items:flex-start;padding:.55rem 0;} .dot{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:700;flex:none;background:#F1ECE7;color:var(--mu);}
.dot.done{background:var(--gr);color:#fff;} .dot.now{background:var(--or);color:#fff;box-shadow:0 0 0 5px rgba(232,116,47,.18);} .step .t{font-weight:600;} .step .d{font-size:.82rem;color:var(--mu);}
.pill-lbl{font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--or);}
.disc{margin-top:1rem;padding-top:.8rem;border-top:1px solid var(--bd);color:var(--mu);font-size:.78rem;line-height:1.5;}
.hero{font-family:'Fraunces',serif;font-weight:600;font-size:3.1rem;line-height:1.1;letter-spacing:-.02em;} .hero .a{color:var(--or);}
.sim{display:inline-block;background:var(--or);color:#fff;font-size:.66rem;font-weight:700;letter-spacing:.08em;padding:.2rem .6rem;border-radius:99px;}
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--bd);border-radius:14px;padding:.8rem 1rem;}
</style>
"""

def inject():
    st.markdown(CSS, unsafe_allow_html=True)

def html(s: str):
    st.markdown(s, unsafe_allow_html=True)

def header(title: str, sub: str = "", eyebrow: str = ""):
    e = f'<span class="eyebrow">{eyebrow}</span>' if eyebrow else ""
    html(f'<div class="ph">{e}<h1>{title}</h1><p>{sub}</p></div>')

def badge(text: str, kind: str = "gray") -> str:
    return f'<span class="badge b-{kind}">{text}</span>'

LEVEL = {1: ("Routine", "1"), 2: ("Urgent", "2"), 3: ("Emergency", "3")}
def level_badge(level) -> str:
    t, k = LEVEL.get(level, ("Routine", "1"))
    return badge(t, k)

CROWD_KIND = {"Low": "green", "Moderate": "amber", "Busy": "amber", "Congested": "red"}
def crowd_badge(label: str) -> str:
    return badge(label, CROWD_KIND.get(label, "gray"))

def kpi(label: str, value, sub: str = "", tone: str = "") -> str:
    return f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'

def bar(pct: float, tone: str = "") -> str:
    return f'<div class="bar {tone}"><div style="width:{max(2, min(100, pct))}%"></div></div>'

def card(inner: str, cls: str = ""):
    html(f'<div class="card {cls}">{inner}</div>')

"""Deterministic symptom phrase coverage for the fixed KB vocabulary.

The LLM extractor/normalizer is still the broad language layer. This lexicon
acts as a safety net for common phrasings and for terse follow-up answers
where the graph already knows the exact symptom being asked about.
"""
from __future__ import annotations

import re

SYMPTOM_SYNONYMS: dict[str, list[str]] = {
    # Eye
    "eye_redness": ["red eye", "red eyes", "redness", "bloodshot"],
    "eye_pain": ["eye hurts", "eye pain", "pain in my eye", "painful eye", "aching eye"],
    "severe_eye_pain": ["severe eye pain", "intense eye pain", "excruciating eye pain"],
    "eye_discharge": ["eye discharge", "discharge", "gunky eye", "crusty eye"],
    "eye_discharge_purulent": ["thick discharge", "yellow discharge", "green discharge", "pus"],
    "itching": ["itchy", "itching"],
    "eye_dryness": ["dry eye", "dry eyes"],
    "eye_irritation": ["eye irritation", "irritated eye", "burning eye"],
    "blurred_vision": ["blurry", "blurred vision", "blurry vision", "vision is blurred"],
    "vision_loss_sudden": ["sudden vision loss", "lost vision suddenly", "went blind"],
    "vision_loss_gradual": ["vision getting worse", "gradual vision loss", "slowly losing vision"],
    "halos_around_lights": ["halos", "rings around lights", "halo around lights"],
    "foreign_body_sensation": ["something in my eye", "stuck in my eye", "scratchy eye", "foreign body"],
    "tearing": ["watery eye", "watering", "tearing", "tears"],
    "light_sensitivity": ["light sensitivity", "sensitive to light", "bright light hurts", "photophobia"],
    "floaters": ["floaters", "spots in vision"],
    "flashes_of_light": ["flashes of light", "seeing flashes"],
    "curtain_shadow_vision": ["curtain", "shadow in vision", "curtain-like shadow"],
    "eyelid_crusting": ["eyelid crusting", "crusted eyelid", "crusty eyelid"],
    "gritty_sensation": ["gritty", "sandy", "gritty sensation"],
    "eyelid_lump": ["eyelid lump", "lump on eyelid", "bump on eyelid"],
    "eyelid_swelling": ["eyelid swelling", "swollen eyelid"],
    "contact_lens_use": ["contact lenses", "contacts", "wear lenses"],
    "glare_sensitivity": ["glare", "glare sensitivity"],
    # General
    "runny_nose": ["runny nose", "stuffy nose", "blocked nose"],
    "sore_throat": ["sore throat", "throat hurts"],
    "cough": ["cough", "coughing"],
    "mild_fever": ["mild fever", "low grade fever"],
    "fever": ["fever", "temperature", "high temperature"],
    "sneezing": ["sneezing", "sneeze"],
    "body_ache": ["body ache", "body aches", "muscle aches", "aches all over"],
    "fatigue": ["fatigue", "tired", "exhausted", "weak"],
    "headache": ["headache", "head hurts", "head pain"],
    "chills": ["chills", "shivering"],
    "nausea": ["nausea", "nauseous", "feel sick", "queasy"],
    "visual_aura": ["visual aura", "zigzags", "shimmering lights", "blind spots"],
    "throbbing_pain": ["throbbing", "pulsing", "pulsating", "pounding"],
    "sudden_worst_headache_of_life": ["worst headache of my life", "worst headache", "thunderclap"],
    "neurological_deficit": ["weakness one side", "numbness one side", "trouble speaking", "confusion"],
    "diarrhea": ["diarrhea", "loose stools"],
    "vomiting": ["vomiting", "vomited", "throwing up"],
    "abdominal_pain": ["stomach pain", "abdominal pain", "belly pain", "belly hurts"],
    "blood_in_stool": ["blood in stool", "bloody stool", "blood in my stool"],
    "severe_dehydration": ["barely urinating", "can't keep fluids down", "very dehydrated"],
    "chest_pain": ["chest pain", "chest pressure", "chest hurts"],
    "shortness_of_breath": ["shortness of breath", "can't breathe", "breathless", "trouble breathing"],
    "sweating": ["sweating", "sweaty", "cold sweat"],
    "left_arm_pain": ["left arm pain", "pain down my arm", "jaw pain", "shoulder pain"],
    "right_lower_quadrant_pain": ["lower right", "right lower belly", "right lower abdomen"],
    "facial_droop": ["facial droop", "face drooping", "drooping face"],
    "slurred_speech": ["slurred speech", "can't speak clearly"],
    "arm_weakness": ["arm weakness", "weak arm"],
    "facial_pain_pressure": ["facial pain", "sinus pressure", "face pressure"],
    "nasal_congestion": ["nasal congestion", "congested", "blocked nose"],
    "purulent_nasal_discharge": ["thick nasal discharge", "yellow mucus", "green mucus"],
    "reduced_smell": ["reduced smell", "can't smell", "loss of smell"],
    "swollen_lymph_nodes": ["swollen glands", "swollen lymph nodes", "tender glands"],
    "neck_tightness": ["neck tightness", "tight neck", "tight shoulders"],
    "stress": ["stress", "stressed", "under pressure"],
    "heartburn": ["heartburn", "burning chest", "burning after meals"],
    "regurgitation": ["regurgitation", "sour fluid", "bitter fluid", "acid comes up"],
    "dysuria": [
        "burn when i pee",
        "burning urination",
        "burning when i urinate",
        "burns when i urinate",
        "burning when urinating",
        "pain when urinating",
        "dysuria",
    ],
    "urinary_frequency": ["urinating more often", "peeing often", "frequent urination"],
    "urinary_urgency": ["urgent need to urinate", "urinary urgency"],
    "flank_pain": ["flank pain", "side pain", "pain in my side", "back below the ribs"],
    "hematuria": ["blood in urine", "pink urine", "red urine"],
    "radiating_groin_pain": ["radiates to groin", "travels to groin", "shoots to groin"],
    "leg_swelling": ["leg swelling", "swollen leg", "one leg swollen"],
    "rapid_heart_rate": ["heart racing", "rapid heart", "fast heartbeat"],
    "dizziness": ["dizzy", "lightheaded", "light-headed"],
    "sense_of_impending_doom": ["impending doom", "something terrible"],
    "hives_or_rash": ["hives", "rash", "new rash"],
    "facial_swelling": ["facial swelling", "swollen lips", "swollen tongue", "face swelling"],
}

_NO_PATTERNS = (
    r"\bno\b",
    r"\bnot\b",
    r"\bnone\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bdon't\b",
    r"\bdo not\b",
    r"\bdidn't\b",
    r"\bdid not\b",
    r"\bnot that i",
)

_YES_PATTERNS = (
    r"\byes\b",
    r"\byep\b",
    r"\byeah\b",
    r"\bi do\b",
    r"\bi have\b",
    r"\bthere is\b",
    r"\bnoticed\b",
)


def answer_polarity(text: str) -> bool | None:
    lowered = text.strip().lower()
    if not lowered:
        return None
    if any(re.search(pattern, lowered) for pattern in _NO_PATTERNS):
        return False
    if any(re.search(pattern, lowered) for pattern in _YES_PATTERNS):
        return True
    return None


def symptoms_in_text(text: str, vocabulary: set[str]) -> dict[str, bool]:
    lowered = text.lower()
    matches: dict[str, bool] = {}
    for symptom, phrases in SYMPTOM_SYNONYMS.items():
        if symptom not in vocabulary:
            continue
        if symptom in lowered:
            matches[symptom] = _polarity_near_phrase(lowered, symptom.replace("_", " "))
            continue
        for phrase in phrases:
            if phrase in lowered:
                matches[symptom] = _polarity_near_phrase(lowered, phrase)
                break
    return matches


def symptom_phrases_in_text(text: str, vocabulary: set[str]) -> list[str]:
    return [symptom for symptom, present in symptoms_in_text(text, vocabulary).items() if present]


def follow_up_target_mapping(asked_for: str | None, answer: str, vocabulary: set[str]) -> dict[str, bool]:
    if not asked_for or asked_for not in vocabulary:
        return {}
    polarity = answer_polarity(answer)
    return {asked_for: polarity} if polarity is not None else {}


def _polarity_near_phrase(text: str, phrase: str) -> bool:
    idx = text.find(phrase)
    if idx < 0:
        return True
    prefix = text[max(0, idx - 35):idx]
    return not any(re.search(pattern, prefix) for pattern in _NO_PATTERNS)

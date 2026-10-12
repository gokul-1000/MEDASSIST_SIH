"""Curated follow-up wording for known KB symptom targets.

The Planner decides what to ask about. This module keeps wording stable and
domain-specific so raw labels like ``foreign_body_sensation`` do not become
ambiguous questions such as "something stuck in your throat".
"""
from __future__ import annotations

QUESTION_BANK = {
 
    # ---------------------------------------------------------------
    # EYE
    # ---------------------------------------------------------------
    "eye_redness": [
        {"priority": 1, "question": "Is the white part of your eye red or bloodshot?", "context": "default", "severity": "low"},
    ],
    "eye_pain": [
        {"priority": 1, "question": "Do you have pain in the eye itself?", "context": "default", "severity": "low"},
        {"priority": 2, "question": "Is the eye pain severe or intense?", "context": "if_eye_pain", "severity": "high"},
    ],
    "eye_swelling": [
        {"priority": 1, "question": "Is your eyelid or the area around the eye swollen?", "context": "default", "severity": "low"},
        {"priority": 2, "question": "Does it feel like the swelling is burning, and is it around the eye or inside it?", "context": "if_eye_swelling", "severity": "moderate"},
    ],
    "eye_discharge": [
        {"priority": 1, "question": "Is there any discharge, pus, or crusting from the eye?", "context": "default", "severity": "low"},
        {"priority": 2, "question": "Is the eye discharge thick, yellow, or green?", "context": "if_eye_discharge", "severity": "moderate"},
    ],
    "eye_itching": [
        {"priority": 1, "question": "Are your eyes itchy?", "context": "default", "severity": "low"},
    ],
    "eye_dryness": [
        {"priority": 1, "question": "Do your eyes feel dry?", "context": "default", "severity": "low"},
    ],
    "eye_irritation": [
        {"priority": 1, "question": "Do your eyes feel irritated or burning?", "context": "default", "severity": "low"},
    ],
    "blurred_vision": [
        {"priority": 1, "question": "Is your vision blurry?", "context": "default", "severity": "moderate"},
    ],
    "vision_loss_sudden": [
        {"priority": 1, "question": "Have you had sudden vision loss or a curtain-like shadow in your vision?", "context": "default", "severity": "high"},
    ],
    "vision_loss_gradual": [
        {"priority": 1, "question": "Has your vision been getting worse gradually over time?", "context": "default", "severity": "moderate"},
    ],
    "halos_around_lights": [
        {"priority": 1, "question": "Do you see halos or rings around lights?", "context": "if_blurred_vision", "severity": "moderate"},
    ],
    "foreign_body_sensation": [
        {"priority": 1, "question": "Does it feel like something is stuck in your eye or scratching the surface?", "context": "default", "severity": "low"},
    ],
    "tearing": [
        {"priority": 1, "question": "Is the eye watering or tearing more than usual?", "context": "default", "severity": "low"},
    ],
    "light_sensitivity": [
        {"priority": 1, "question": "Does bright light make the eye pain or discomfort worse?", "context": "if_eye_pain", "severity": "moderate"},
    ],
    "floaters": [
        {"priority": 1, "question": "Have you noticed new floaters or spots in your vision?", "context": "default", "severity": "moderate"},
    ],
    "flashes_of_light": [
        {"priority": 1, "question": "Have you seen flashes of light?", "context": "if_floaters", "severity": "high"},
    ],
    "curtain_shadow_vision": [
        {"priority": 1, "question": "Do you see a curtain-like shadow over part of your vision?", "context": "default", "severity": "high"},
    ],
    "eyelid_crusting": [
        {"priority": 1, "question": "Are your eyelids crusted, especially after sleep?", "context": "default", "severity": "low"},
    ],
    "gritty_sensation": [
        {"priority": 1, "question": "Does the eye feel gritty or sandy?", "context": "default", "severity": "low"},
    ],
    "eyelid_lump": [
        {"priority": 1, "question": "Do you have a lump on your eyelid?", "context": "default", "severity": "low"},
    ],
    "contact_lens_use": [
        {"priority": 1, "question": "Do you wear contact lenses?", "context": "if_eye_pain", "severity": "low"},
    ],
    "glare_sensitivity": [
        {"priority": 1, "question": "Are you more bothered by glare than usual?", "context": "if_blurred_vision", "severity": "low"},
    ],
 
    # ---------------------------------------------------------------
    # GENERAL / UPPER RESPIRATORY
    # ---------------------------------------------------------------
    "common_cold": [
        {"priority": 1, "question": "Do you have common cold symptoms, like a runny nose, sneezing, or a mild sore throat?", "context": "default", "severity": "low"},
        {"priority": 2, "question": "How many days have you had these cold symptoms?", "context": "if_common_cold", "severity": "low"},
        {"priority": 3, "question": "Have your symptoms gotten worse instead of better after about a week?", "context": "if_common_cold", "severity": "moderate"},
    ],
    "runny_nose": [
        {"priority": 1, "question": "Do you have a runny or stuffy nose?", "context": "default", "severity": "low"},
    ],
    "sore_throat": [
        {"priority": 1, "question": "Do you have a sore throat?", "context": "default", "severity": "low"},
    ],
    "cough": [
        {"priority": 1, "question": "Do you have a cough?", "context": "default", "severity": "low"},
    ],
    "fever": [
        {"priority": 1, "question": "Have you measured your temperature, or do you feel feverish?", "context": "default", "severity": "moderate"},
        {"priority": 2, "question": "Do you also have a headache along with the fever?", "context": "if_fever", "severity": "moderate"},
    ],
    "mild_fever": [
        {"priority": 1, "question": "Have you had a mild fever?", "context": "default", "severity": "low"},
    ],
    "sneezing": [
        {"priority": 1, "question": "Are you sneezing?", "context": "default", "severity": "low"},
    ],
    "fatigue": [
        {"priority": 1, "question": "Do you feel unusually tired or weak?", "context": "default", "severity": "low"},
    ],
    "headache": [
        {"priority": 1, "question": "Do you have a headache?", "context": "default", "severity": "low"},
        {"priority": 2, "question": "How long have you been having this headache?", "context": "if_headache", "severity": "moderate"},
        {"priority": 3, "question": "Have you noticed any numbness, tingling, or weakness anywhere in your body?", "context": "if_headache", "severity": "high"},
        {"priority": 4, "question": "Is the headache getting worse over time?", "context": "if_headache", "severity": "moderate"},
    ],
    "chills": [
        {"priority": 1, "question": "Have you had chills?", "context": "if_fever", "severity": "low"},
    ],
    "visual_aura": [
        {"priority": 1, "question": "Have you noticed visual aura, such as zigzags, shimmering lights, or blind spots?", "context": "if_headache", "severity": "moderate"},
    ],
    "throbbing_pain": [
        {"priority": 1, "question": "Is the pain throbbing or pulsing?", "context": "if_headache", "severity": "low"},
    ],
    "sudden_worst_headache_of_life": [
        {"priority": 1, "question": "Did the headache start suddenly as the worst headache of your life?", "context": "if_headache", "severity": "high"},
    ],
    "neurological_deficit": [
        {"priority": 1, "question": "Have you had new weakness, numbness, trouble speaking, or facial drooping?", "context": "default", "severity": "high"},
    ],
 
    # ---------------------------------------------------------------
    # GASTROINTESTINAL
    # ---------------------------------------------------------------
    "diarrhea": [
        {"priority": 1, "question": "Have you had diarrhea or loose stools?", "context": "default", "severity": "low"},
    ],
    "vomiting": [
        {"priority": 1, "question": "Have you vomited?", "context": "default", "severity": "low"},
    ],
    "abdominal_pain": [
        {"priority": 1, "question": "Do you have abdominal or belly pain?", "context": "default", "severity": "moderate"},
        {"priority": 2, "question": "Is the belly pain mainly in the lower right side?", "context": "if_abdominal_pain", "severity": "high"},
    ],
    "blood_in_stool": [
        {"priority": 1, "question": "Have you seen blood in your stool?", "context": "if_diarrhea", "severity": "high"},
    ],
    "severe_dehydration": [
        {"priority": 1, "question": "Are you very thirsty, dizzy, barely urinating, or unable to keep fluids down?", "context": "if_diarrhea", "severity": "high"},
    ],
    "heartburn": [
        {"priority": 1, "question": "Do you have burning discomfort behind your breastbone or after meals?", "context": "default", "severity": "low"},
        {"priority": 2, "question": "Do you get sour or bitter fluid coming back up into your throat or mouth?", "context": "if_heartburn", "severity": "low"},
    ],
 
    # ---------------------------------------------------------------
    # CARDIO / RESPIRATORY (chest)
    # ---------------------------------------------------------------
    "chest_pain": [
        {"priority": 1, "question": "Do you have chest pain or pressure?", "context": "default", "severity": "high"},
        {"priority": 2, "question": "Does the pain spread to your left arm, shoulder, jaw, or back?", "context": "if_chest_pain", "severity": "high"},
        {"priority": 3, "question": "Are you sweating more than usual along with the chest pain?", "context": "if_chest_pain", "severity": "high"},
        {"priority": 4, "question": "Do you have a sudden intense feeling that something terrible is about to happen?", "context": "if_chest_pain", "severity": "high"},
    ],
    "shortness_of_breath": [
        {"priority": 1, "question": "Are you short of breath?", "context": "default", "severity": "high"},
    ],
    "rapid_heart_rate": [
        {"priority": 1, "question": "Does your heart feel like it is racing?", "context": "default", "severity": "moderate"},
    ],
    "dizziness": [
        {"priority": 1, "question": "Do you feel dizzy or lightheaded?", "context": "default", "severity": "moderate"},
    ],
    "leg_swelling": [
        {"priority": 1, "question": "Do you have swelling in one leg?", "context": "default", "severity": "moderate"},
    ],
 
    # ---------------------------------------------------------------
    # NEURO / STROKE RED FLAGS
    # ---------------------------------------------------------------
    "facial_droop": [
        {"priority": 1, "question": "Has one side of your face drooped?", "context": "default", "severity": "high"},
    ],
    "slurred_speech": [
        {"priority": 1, "question": "Is your speech slurred or hard to understand?", "context": "default", "severity": "high"},
    ],
    "arm_weakness": [
        {"priority": 1, "question": "Do you have new weakness in one arm?", "context": "default", "severity": "high"},
    ],
 
    # ---------------------------------------------------------------
    # SINUS
    # ---------------------------------------------------------------
    "facial_pain_pressure": [
        {"priority": 1, "question": "Do you feel pressure or pain in your face or around your sinuses?", "context": "default", "severity": "low"},
    ],
    "nasal_congestion": [
        {"priority": 1, "question": "Is your nose congested or blocked?", "context": "default", "severity": "low"},
        {"priority": 2, "question": "Is your nasal discharge thick, yellow, or green?", "context": "if_nasal_congestion", "severity": "low"},
    ],
    "reduced_smell": [
        {"priority": 1, "question": "Is your sense of smell reduced?", "context": "default", "severity": "low"},
    ],
 
    # ---------------------------------------------------------------
    # URINARY
    # ---------------------------------------------------------------
    "dysuria": [
        {"priority": 1, "question": "Does it burn or hurt when you urinate?", "context": "default", "severity": "low"},
    ],
    "urinary_frequency": [
        {"priority": 1, "question": "Are you urinating more often than usual?", "context": "default", "severity": "low"},
    ],
    "urinary_urgency": [
        {"priority": 1, "question": "Do you feel an urgent need to urinate?", "context": "default", "severity": "low"},
    ],
    "flank_pain": [
        {"priority": 1, "question": "Do you have pain in your side or back below the ribs?", "context": "default", "severity": "moderate"},
        {"priority": 2, "question": "Have you noticed blood or pink color in your urine?", "context": "if_flank_pain", "severity": "high"},
        {"priority": 3, "question": "Does the side or back pain travel down toward your groin?", "context": "if_flank_pain", "severity": "moderate"},
    ],
 
    # ---------------------------------------------------------------
    # MISC
    # ---------------------------------------------------------------
    "neck_tightness": [
        {"priority": 1, "question": "Do you feel tightness in your neck or shoulders?", "context": "default", "severity": "low"},
    ],
    "stress": [
        {"priority": 1, "question": "Have you been under more stress than usual?", "context": "default", "severity": "low"},
    ],
    "hives_or_rash": [
        {"priority": 1, "question": "Do you have hives or a new widespread rash?", "context": "default", "severity": "moderate"},
    ],
 
    # ---------------------------------------------------------------
    # GENERIC (reusable across body regions instead of one-off keys)
    # ---------------------------------------------------------------
    "symptom_duration": [
        {"priority": 99, "question": "Since when have you been experiencing this?", "context": "default", "severity": "low"},
    ],
}


def question_for_symptom(
    symptom: str,
    extracted_symptoms: dict[str, object] | None = None,
) -> str | None:
    """
    Return the best question wording for `symptom` given what is already known.

    Selection rules:
    1. A context of ``"default"`` always matches (fallback phrasing).
    2. A context like ``"if_eye_pain"`` matches when ``extracted_symptoms``
       contains ``eye_pain=True`` — indicating a more specific follow-up.
    3. Specific (non-default) context matches are preferred over default ones.
       Within each group, lower priority number wins.
    4. If nothing in the bank matches, return None → caller falls back to LLM.
    """
    entries = QUESTION_BANK.get(symptom)
    if not entries:
        return None

    known_true: set[str] = set()
    if extracted_symptoms:
        known_true = {k for k, v in extracted_symptoms.items() if v is True}

    def _context_satisfied(ctx: str) -> bool:
        if ctx == "default":
            return True
        if ctx.startswith("if_"):
            return ctx[3:] in known_true
        return False

    specific = sorted(
        [e for e in entries if e["context"] != "default" and _context_satisfied(e["context"])],
        key=lambda e: e["priority"],
    )
    if specific:
        return specific[0]["question"]

    # Fall back to default phrasing (lowest priority number among default entries)
    defaults = sorted(
        [e for e in entries if e["context"] == "default"],
        key=lambda e: e["priority"],
    )
    if defaults:
        return defaults[0]["question"]

    # Last resort: lowest priority entry regardless of context
    return min(entries, key=lambda e: e["priority"])["question"]


def get_question_severity(symptom: str) -> str:
    """Return the highest severity level defined for this symptom across all
    its question variants. Used by the planner to weight safety scores.
    Possible values: 'low', 'moderate', 'high'. Defaults to 'low'."""
    entries = QUESTION_BANK.get(symptom)
    if not entries:
        return "low"
    order = {"low": 0, "moderate": 1, "high": 2}
    return max(entries, key=lambda e: order.get(e["severity"], 0))["severity"]

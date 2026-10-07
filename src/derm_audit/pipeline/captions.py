"""Caption generation and safeguards (manuscript section 4.7).

Two captions are produced per image. The template grammar cannot add information
absent from the record, so it is always safe. The language model is text-only,
never sees the image, and receives only fields tagged S, C or E.

Four safeguards limit unsupported content:
  1. input restriction   - only verified fields reach the model
  2. instruction constraints - the prompt forbids unsupported statements
  3. automatic checking  - a rule-based checker rejects violations
  4. fallback            - one regeneration, then the template caption
"""


from __future__ import annotations

import re
from dataclasses import dataclass

from .reports import NOT_RECORDED

LM_INPUT_TAGS = {"S", "C", "E"}
NEGATION_WINDOW = 4
NEGATION_CUES = {"no", "not", "without", "absent", "denies", "negative", "free"}

FORBIDDEN_PATTERNS = [
    (r"\b(treat|treatment|therapy|prescrib\w*|manage\w*ment|apply|dose|dosage|mg\b)", "treatment"),
    (r"\b(prognos\w+|outcome|survival|recur\w*|cure[sd]?\b|resolve[sd]?\b)", "prognosis"),
    (r"\b(\d{1,3})[- ]year[- ]old\b", "age"),
    (r"\b(male|female|man|woman|boy|girl)\b", "sex"),
    (r"\b(caucasian|african|asian|hispanic|latino|white|black)\s+(patient|man|woman|skin)\b", "ethnicity"),
    (r"\b(biops\w+ (?:confirm|prov)\w*|histolog\w+)\b", "unsupported_certainty"),
]

MORPHOLOGY_TERMS = {
    "macule", "papule", "plaque", "nodule", "vesicle", "bulla", "pustule", "wheal",
    "scale", "crust", "erosion", "ulcer", "fissure", "lichenification", "atrophy",
    "annular", "linear", "targetoid", "erythematous", "hyperpigmented",
    "hypopigmented", "violaceous", "scaly", "weeping",
}

PROMPT_TEMPLATE = """You write one factual caption for a dermatology image.

You are given a structured record. Use ONLY the facts in it.

Rules:
1. State no diagnosis other than the record's disease_label.
2. State no treatment, prognosis, age, sex or ethnicity unless it is in the record.
3. Use no morphological or colour descriptor that is not in the record.
4. Name no body site other than the record's anatomical_site.
5. Write one or two sentences. Return JSON: {{"caption": "..."}}

Record:
{record}
"""

__all__ = ["restrict_to_verified_fields", "build_prompt", "template_caption",
           "CaptionCheck", "check_caption", "generate_caption"]


def restrict_to_verified_fields(report: dict) -> dict:
    """Safeguard 1: pass only S/C/E fields, dropping anything not recorded."""
    out = {}
    for name, entry in report.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("provenance") not in LM_INPUT_TAGS:
            continue
        value = entry.get("value")
        if value in (None, "", NOT_RECORDED):
            continue
        if name.startswith("caption"):
            continue
        out[name] = value
    return out


def build_prompt(report: dict) -> str:
    """Safeguard 2: the constrained instruction prompt."""
    import json

    return PROMPT_TEMPLATE.format(
        record=json.dumps(restrict_to_verified_fields(report), indent=2, default=str)
    )


def template_caption(report: dict) -> str:
    """Fixed-grammar caption. Cannot add information absent from the record."""
    def get(key):
        return report.get(key, {}).get("value")

    label = get("disease_label") or "a skin condition"
    site = get("anatomical_site")
    morph = get("lesion_morphology")
    certainty = get("diagnosis_certainty")

    parts = [f"A clinical photograph showing {str(label).lower()}"]
    if site and site != NOT_RECORDED:
        parts.append(f" on the {str(site).lower()}")
    parts.append(".")
    if morph and morph != NOT_RECORDED:
        parts.append(f" The recorded morphology is {str(morph).lower()}.")
    if certainty and certainty != NOT_RECORDED:
        parts.append(f" Diagnosis certainty: {str(certainty).lower()}.")
    return "".join(parts)


def _is_negated(text: str, start: int) -> bool:
    """True when a negation cue falls within the preceding window of words."""
    before = text[:start].split()
    return any(w.strip(".,;:").lower() in NEGATION_CUES for w in before[-NEGATION_WINDOW:])


@dataclass
class CaptionCheck:
    ok: bool
    violations: list


def check_caption(caption: str, report: dict, synonyms: dict | None = None) -> CaptionCheck:
    """Safeguard 3: reject captions that go beyond the record.

    A caption is rejected if it names a different body site, diagnosis or skin
    tone, if it uses a descriptor absent from the record, or if it contains a
    forbidden term. Negated mentions are allowed via a simple negation window.
    """
    synonyms = synonyms or {}
    text = caption.lower()
    violations = []

    def get(key):
        return report.get(key, {}).get("value")

    def allowed_forms(value):
        if not value or value == NOT_RECORDED:
            return set()
        base = str(value).lower()
        forms = {base} | set(synonyms.get(base, []))
        forms |= {w for w in re.split(r"[\s/,-]+", base) if len(w) > 3}
        return forms

    for m in re.finditer(r"|".join(p for p, _ in FORBIDDEN_PATTERNS), text):
        if _is_negated(text, m.start()):
            continue
        for pattern, kind in FORBIDDEN_PATTERNS:
            if re.fullmatch(pattern, m.group(0)) or re.search(pattern, m.group(0)):
                violations.append(f"forbidden_{kind}: {m.group(0)!r}")
                break

    site_forms = allowed_forms(get("anatomical_site"))
    for site in ("face", "scalp", "trunk", "back", "chest", "abdomen", "arm", "forearm",
                 "hand", "leg", "thigh", "foot", "neck", "groin", "buttock"):
        if re.search(rf"\b{site}\b", text) and site_forms and site not in site_forms:
            if not _is_negated(text, text.find(site)):
                violations.append(f"unsupported_site: {site!r}")

    morph_forms = allowed_forms(get("lesion_morphology"))
    for term in MORPHOLOGY_TERMS:
        if re.search(rf"\b{term}\w*\b", text) and term not in morph_forms:
            if not _is_negated(text, text.find(term)):
                violations.append(f"unsupported_descriptor: {term!r}")

    label_forms = allowed_forms(get("disease_label"))
    if label_forms and not any(f in text for f in label_forms):
        violations.append("diagnosis_not_stated_or_changed")

    for tone in ("fair", "pale", "dark-skinned", "light-skinned", "brown skin", "black skin"):
        if tone in text:
            violations.append(f"unsupported_skin_tone: {tone!r}")

    return CaptionCheck(ok=not violations, violations=sorted(set(violations)))


def generate_caption(report: dict, lm_call, synonyms: dict | None = None,
                     max_attempts: int = 2) -> dict:
    """Run the full guarded pipeline (safeguards 1-4).

    `lm_call(prompt) -> str` is any text-only model call. On repeated failure the
    template caption is used and the record is flagged, so every image always has
    a safe caption.
    """
    prompt = build_prompt(report)
    attempts = []
    for attempt in range(1, max_attempts + 1):
        try:
            caption = str(lm_call(prompt)).strip()
        except Exception as exc:  # the model is optional infrastructure
            attempts.append({"attempt": attempt, "error": repr(exc)})
            break
        check = check_caption(caption, report, synonyms)
        attempts.append({"attempt": attempt, "caption": caption,
                         "violations": check.violations})
        if check.ok:
            return {"caption": caption, "route": "lm" if attempt == 1 else "lm_regenerated",
                    "flagged": False, "attempts": attempts}
        prompt = (build_prompt(report)
                  + "\nThe previous attempt was rejected for: "
                  + "; ".join(check.violations) + "\nRewrite it without those problems.")

    return {"caption": template_caption(report), "route": "template_fallback",
            "flagged": True, "attempts": attempts}

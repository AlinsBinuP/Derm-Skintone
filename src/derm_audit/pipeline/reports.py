"""Provenance-tagged structured reports (manuscript section 4.6).

Every field carries exactly one of four provenance tags:

  S  source-provided      copied from the original dataset without change
  C  computed             derived deterministically by the pipeline
  E  expert-validated     confirmed or corrected by the dermatologist
  L  language-model       free-text caption written by the language model

The language model writes captions only. It never writes or modifies a
structured field, and `validate_report` enforces that invariant. Fields absent
from the source are recorded as "not recorded" rather than inferred.
"""


from __future__ import annotations

import json
from pathlib import Path

NOT_RECORDED = "not recorded"

# field -> provenance tags that are permitted for it (manuscript table 8)
FIELD_PROVENANCE = {
    "case_id": {"C"},
    "source_dataset": {"S"},
    "source_label": {"S"},
    "disease_label": {"C", "E"},
    "diagnosis_certainty": {"S"},
    "fitzpatrick_source": {"S"},
    "ita_value": {"C"},
    "ita_confidence": {"C"},
    "analysis_band": {"C", "E"},
    "anatomical_site": {"S", "E"},
    "age_band": {"S"},
    "sex": {"S"},
    "lesion_morphology": {"S", "E"},
    "image_quality_flags": {"C"},
    "caption_template": {"C"},
    "caption_lm": {"L"},
    "validation_status": {"E"},
}

# Structured fields may never be written by the language model.
LM_PERMITTED_FIELDS = {f for f, tags in FIELD_PROVENANCE.items() if "L" in tags}

JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Dermatology image structured report",
    "type": "object",
    "required": ["case_id", "source_dataset", "disease_label", "analysis_band"],
    "additionalProperties": False,
    "properties": {
        name: {
            "type": "object",
            "required": ["value", "provenance"],
            "additionalProperties": False,
            "properties": {
                "value": {},
                "provenance": {"type": "string", "enum": sorted(tags)},
            },
        }
        for name, tags in FIELD_PROVENANCE.items()
    },
}

__all__ = ["JSON_SCHEMA", "FIELD_PROVENANCE", "NOT_RECORDED", "field",
           "build_report", "validate_report", "write_schema", "coverage_by_provenance"]


def field(value, provenance: str) -> dict:
    """Wrap a value with its provenance tag."""
    if provenance not in {"S", "C", "E", "L"}:
        raise ValueError(f"unknown provenance tag {provenance!r}")
    return {"value": NOT_RECORDED if value is None else value, "provenance": provenance}


def build_report(case_id, source_dataset, source_label, disease_label, analysis_band,
                 ita=None, ita_confidence=None, fitzpatrick_source=None,
                 anatomical_site=None, age_band=None, sex=None,
                 lesion_morphology=None, diagnosis_certainty=None,
                 caption_template=None, caption_lm=None, validation_status=None,
                 disease_label_provenance="C", band_provenance="C") -> dict:
    """Assemble one structured report with correct provenance on every field.

    `lesion_morphology` is only ever source-provided or expert-validated: when a
    source does not record it, it stays "not recorded" rather than being inferred.
    """
    report = {
        "case_id": field(case_id, "C"),
        "source_dataset": field(source_dataset, "S"),
        "source_label": field(source_label, "S"),
        "disease_label": field(disease_label, disease_label_provenance),
        "analysis_band": field(analysis_band, band_provenance),
        "ita_value": field(ita, "C"),
        "ita_confidence": field(ita_confidence, "C"),
        "fitzpatrick_source": field(fitzpatrick_source, "S"),
        "anatomical_site": field(anatomical_site, "S" if anatomical_site else "S"),
        "age_band": field(age_band, "S"),
        "sex": field(sex, "S"),
        "lesion_morphology": field(lesion_morphology, "S"),
        "diagnosis_certainty": field(diagnosis_certainty, "S"),
        "caption_template": field(caption_template, "C"),
        "caption_lm": field(caption_lm, "L"),
        "validation_status": field(validation_status, "E"),
    }
    return report


def validate_report(report: dict) -> list:
    """Return a list of provenance violations; empty means the report is valid."""
    problems = []
    for name, entry in report.items():
        if name not in FIELD_PROVENANCE:
            problems.append(f"unknown field: {name}")
            continue
        if not isinstance(entry, dict) or "provenance" not in entry:
            problems.append(f"{name}: missing provenance tag")
            continue
        tag = entry["provenance"]
        if tag not in FIELD_PROVENANCE[name]:
            problems.append(
                f"{name}: provenance {tag!r} not permitted "
                f"(allowed: {sorted(FIELD_PROVENANCE[name])})"
            )
        if tag == "L" and name not in LM_PERMITTED_FIELDS:
            problems.append(f"{name}: language model may not write a structured field")
    for required in ("case_id", "disease_label", "analysis_band"):
        if required not in report:
            problems.append(f"missing required field: {required}")
    return problems


def write_schema(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(JSON_SCHEMA, indent=2), encoding="utf-8")
    return path


def coverage_by_provenance(reports) -> dict:
    """Manuscript supplementary table S4: field coverage by provenance tag."""
    total = len(reports)
    out = {}
    for name in FIELD_PROVENANCE:
        present = sum(
            1 for r in reports
            if name in r and r[name].get("value") not in (None, NOT_RECORDED, "")
        )
        tags = {r[name]["provenance"] for r in reports if name in r}
        out[name] = {"present": present,
                     "percent": round(present / total * 100, 1) if total else 0.0,
                     "tags": sorted(tags)}
    return out

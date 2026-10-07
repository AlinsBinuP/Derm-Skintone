"""The language model must never write a structured field, and captions are checked."""
import pytest

from derm_audit.pipeline.captions import (check_caption, generate_caption,
                                          restrict_to_verified_fields, template_caption)
from derm_audit.pipeline.reports import NOT_RECORDED, build_report, validate_report


@pytest.fixture
def report():
    return build_report("c1", "fitzpatrick17k", "tinea corporis", "Tinea corporis",
                        "darker", anatomical_site="Forearm")


def test_valid_report_has_no_provenance_violations(report):
    assert validate_report(report) == []


def test_language_model_may_not_write_a_structured_field(report):
    tampered = dict(report)
    tampered["lesion_morphology"] = {"value": "Plaque", "provenance": "L"}
    problems = validate_report(tampered)
    assert any("language model may not write" in p for p in problems)


def test_absent_fields_are_recorded_not_inferred(report):
    assert report["lesion_morphology"]["value"] == NOT_RECORDED
    assert report["lesion_morphology"]["provenance"] == "S"


def test_only_verified_fields_reach_the_model(report):
    passed = restrict_to_verified_fields(report)
    assert "caption_lm" not in passed
    assert NOT_RECORDED not in passed.values()


def test_template_caption_adds_nothing(report):
    caption = template_caption(report)
    assert "tinea corporis" in caption.lower()
    assert "forearm" in caption.lower()
    assert check_caption(caption, report).ok


@pytest.mark.parametrize("caption,expected_violation", [
    ("Tinea corporis on the face.", "unsupported_site"),
    ("Tinea corporis in a 45-year-old patient.", "forbidden_age"),
    ("Tinea corporis; treat with terbinafine.", "forbidden_treatment"),
    ("An erythematous plaque of tinea corporis on the forearm.", "unsupported_descriptor"),
    ("Psoriasis on the forearm.", "diagnosis_not_stated_or_changed"),
])
def test_checker_rejects_unsupported_content(report, caption, expected_violation):
    result = check_caption(caption, report)
    assert not result.ok
    assert any(expected_violation in v for v in result.violations)


def test_negated_mentions_are_allowed(report):
    caption = "Tinea corporis on the forearm, with no scale or crust present."
    assert check_caption(caption, report).ok


def test_one_regeneration_then_template_fallback(report):
    attempts = iter(["Psoriasis on the face.",
                     "A clinical photograph showing tinea corporis on the forearm."])
    ok = generate_caption(report, lambda _: next(attempts))
    assert ok["route"] == "lm_regenerated" and not ok["flagged"]

    always_bad = generate_caption(report, lambda _: "Melanoma on the scalp, treat urgently.")
    assert always_bad["route"] == "template_fallback"
    assert always_bad["flagged"] is True
    assert check_caption(always_bad["caption"], report).ok


def test_model_failure_falls_back_safely(report):
    def explode(_):
        raise RuntimeError("model unavailable")
    out = generate_caption(report, explode)
    assert out["route"] == "template_fallback" and out["flagged"]

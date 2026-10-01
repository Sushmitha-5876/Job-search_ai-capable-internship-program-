"""
Tests for src/ats_scoring.py
"""

from src.ats_scoring import (
    check_resume_formatting,
    calculate_keyword_overlap,
    calculate_ats_score,
    get_ats_verdict,
)


GOOD_RESUME = """
Jane Doe
jane.doe@example.com | +91 98765 43210

Summary
Backend developer focused on Python and cloud services.

Experience
Software Engineer, Acme Corp, 2022-Present
Built REST APIs using Python, Django, and PostgreSQL. Deployed on AWS
using Docker and Kubernetes. Wrote unit tests and used Git for version
control. Collaborated with a small team using Agile practices daily.

Education
B.E. Computer Science, XYZ University, 2022

Skills
Python, Django, PostgreSQL, AWS, Docker, Kubernetes, Git, REST APIs
"""

THIN_RESUME = "Jane Doe. Software developer."


def test_empty_resume_text_scores_zero():
    result = check_resume_formatting("")
    assert result["score"] == 0
    assert result["checks"][0]["passed"] is False


def test_good_resume_scores_highly_without_ocr():
    result = check_resume_formatting(GOOD_RESUME, used_ocr=False)
    assert result["score"] >= 80
    # every check should be present and in a stable order
    labels = [c["label"] for c in result["checks"]]
    assert labels[0] == "Text extracts cleanly (no OCR needed)"
    assert any("Experience" in label for label in labels)


def test_ocr_fallback_costs_the_ocr_points_but_nothing_else():
    without_ocr = check_resume_formatting(GOOD_RESUME, used_ocr=False)
    with_ocr = check_resume_formatting(GOOD_RESUME, used_ocr=True)
    assert with_ocr["score"] == without_ocr["score"] - 40
    ocr_check = with_ocr["checks"][0]
    assert ocr_check["passed"] is False
    assert "OCR" in ocr_check["detail"]


def test_missing_sections_are_flagged_individually():
    result = check_resume_formatting(THIN_RESUME, used_ocr=False)
    section_checks = {c["label"]: c["passed"] for c in result["checks"] if "section" in c["label"] or "Has" in c["label"]}
    assert all(passed is False for passed in section_checks.values())


def test_contact_info_detection():
    with_contact = check_resume_formatting(GOOD_RESUME)
    without_contact = check_resume_formatting(THIN_RESUME)
    email_check_with = next(c for c in with_contact["checks"] if "Email" in c["label"])
    email_check_without = next(c for c in without_contact["checks"] if "Email" in c["label"])
    assert email_check_with["passed"] is True
    assert email_check_without["passed"] is False


def test_very_short_resume_fails_length_check():
    result = check_resume_formatting(THIN_RESUME)
    length_check = next(c for c in result["checks"] if "length" in c["label"].lower())
    assert length_check["passed"] is False


def test_keyword_overlap_returns_none_score_with_no_job_description():
    result = calculate_keyword_overlap(GOOD_RESUME, "")
    assert result["score"] is None
    assert result["matched_keywords"] == []
    assert result["missing_keywords"] == []


def test_keyword_overlap_finds_real_matches():
    jd = "Looking for a Python developer experienced with Django, AWS, and Docker."
    result = calculate_keyword_overlap(GOOD_RESUME, jd)
    assert result["score"] is not None
    assert result["score"] > 50
    assert "python" in [k.lower() for k in result["matched_keywords"]]


def test_keyword_overlap_flags_genuinely_missing_terms():
    jd = "Looking for a Rust developer experienced with Kafka and GraphQL."
    result = calculate_keyword_overlap(GOOD_RESUME, jd)
    assert result["score"] is not None
    assert result["score"] < 60
    assert len(result["missing_keywords"]) > 0


def test_get_ats_verdict_thresholds():
    assert get_ats_verdict(None) == "Unknown"
    assert get_ats_verdict(90) == "ATS-friendly"
    assert get_ats_verdict(65) == "Needs minor fixes"
    assert get_ats_verdict(45) == "At risk of being filtered out"
    assert get_ats_verdict(10) == "High risk of being missed by ATS software"


def test_calculate_ats_score_without_job_description_uses_formatting_only():
    result = calculate_ats_score(GOOD_RESUME, used_ocr=False, job_description="")
    assert result["keyword_score"] is None
    assert result["overall_score"] == result["formatting_score"]
    assert result["verdict"] == get_ats_verdict(result["overall_score"])
    assert isinstance(result["summary"], list) and len(result["summary"]) > 0


def test_calculate_ats_score_blends_formatting_and_keywords():
    jd = "Looking for a Python developer experienced with Django, AWS, and Docker."
    result = calculate_ats_score(GOOD_RESUME, used_ocr=False, job_description=jd)
    assert result["keyword_score"] is not None
    expected = round(0.5 * result["formatting_score"] + 0.5 * result["keyword_score"])
    assert result["overall_score"] == expected


def test_calculate_ats_score_on_empty_resume_is_zero_and_has_summary():
    result = calculate_ats_score("", used_ocr=False, job_description="")
    assert result["overall_score"] == 0
    assert result["verdict"] == get_ats_verdict(0)
    assert len(result["summary"]) > 0
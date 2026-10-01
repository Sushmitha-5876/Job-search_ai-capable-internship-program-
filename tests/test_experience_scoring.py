"""
Tests for src/experience_scoring.py

Covers: extracting candidate years from resume text (explicit
statement vs. date ranges vs. undeterminable), estimating a job's
required years from its title, and the gap-scoring rules — including
the exact "fresher scored too high for a Senior role" bug this module
was built to fix.
"""

from src.experience_scoring import (
    extract_candidate_years,
    estimate_required_years,
    score_experience_gap,
)


# ----------------------------------------------------------------
# extract_candidate_years
# ----------------------------------------------------------------

def test_extract_years_from_explicit_statement():
    entries = ["5+ years of experience building backend systems"]
    assert extract_candidate_years(entries) == 5.0


def test_extract_years_from_date_range():
    entries = ["Software Engineer, Acme Corp, Jan 2020 - Present"]
    years = extract_candidate_years(entries)
    assert years is not None
    assert years > 4.0  # depends on "today", but well over 4 years by now


def test_extract_years_returns_none_when_undeterminable():
    entries = ["Built several personal projects using Python"]
    assert extract_candidate_years(entries) is None


def test_extract_years_returns_none_for_empty_input():
    assert extract_candidate_years([]) is None
    assert extract_candidate_years(None) is None


def test_extract_years_ignores_implausible_values():
    # "24/7" support should not be parsed as "7 years".
    entries = ["Provided 24/7 support for production systems"]
    assert extract_candidate_years(entries) is None


# ----------------------------------------------------------------
# estimate_required_years
# ----------------------------------------------------------------

def test_estimate_required_years_senior_title():
    assert estimate_required_years("Senior Software Engineer") == 5.0


def test_estimate_required_years_junior_title():
    assert estimate_required_years("Junior Developer") == 1.0


def test_estimate_required_years_defaults_when_no_keyword():
    assert estimate_required_years("Software Engineer") == 2.0


def test_estimate_required_years_prefers_explicit_description_statement():
    # The description says 5, even though the title alone would guess
    # "Software Engineer" -> 2.0. The employer's stated number should win.
    description = "We require 5+ years of relevant experience with Python."
    assert estimate_required_years("Software Engineer", description) == 5.0


def test_estimate_required_years_range_uses_lower_bound():
    description = "3-5 years of experience in backend development."
    assert estimate_required_years("Software Engineer", description) == 3.0


def test_estimate_required_years_ignores_unrelated_numbers_in_description():
    # "24/7 support" and "5 team members" should NOT be mistaken for
    # a years-of-experience requirement - no "years experience" phrase
    # is present, so this must fall back to the title heuristic.
    description = "Join our 5 team members providing 24/7 support to clients."
    assert estimate_required_years("Senior Software Engineer", description) == 5.0  # from title, not the "5"


def test_estimate_required_years_falls_back_when_description_empty():
    assert estimate_required_years("Senior Software Engineer", "") == 5.0
    assert estimate_required_years("Senior Software Engineer", None) == 5.0


# ----------------------------------------------------------------
# score_experience_gap
# ----------------------------------------------------------------

def test_score_experience_gap_returns_none_without_candidate_years():
    assert score_experience_gap(None, 5) is None


def test_score_experience_gap_perfect_match():
    assert score_experience_gap(5, 5) == 100


def test_score_experience_gap_small_gap_scores_high():
    assert score_experience_gap(4, 5) == 80


def test_score_experience_gap_large_gap_scores_low():
    # Use an OVER-experienced candidate for a large gap, so this
    # doesn't accidentally hit the "fresher vs senior role" special
    # case tested separately below.
    assert score_experience_gap(15, 5) == 20


def test_fresher_scored_low_for_senior_role():
    # This is the exact bug flagged during development: a fresher
    # (0-1 years) should score very low against a Senior (5+ years)
    # role, not land in "Possible Match" territory.
    score = score_experience_gap(0.5, 5)
    assert score == 5


def test_score_is_continuous_not_bucketed():
    # Two candidates with slightly different gaps should get
    # slightly different scores, not the same score just because
    # they used to fall in the same bucket. This is the fix for the
    # old "1.1 years and 1.9 years both score 60" cliff.
    score_close_gap = score_experience_gap(candidate_years=3.9, required_years=5)   # gap 1.1
    score_far_gap = score_experience_gap(candidate_years=3.1, required_years=5)     # gap 1.9

    assert score_close_gap != score_far_gap
    assert score_close_gap > score_far_gap
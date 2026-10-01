"""
Tests for src/salary_scoring.py

Includes tests for _resolve_expected_ctc(), which is the fix for the
original bug: a free-typed "Lakhs" field let a unit mistake (typing
1000000 instead of 10) silently tank every job's salary score.
"""

from src.salary_scoring import _resolve_expected_ctc, calculate_salary_score


# ----------------------------------------------------------------
# _resolve_expected_ctc
# ----------------------------------------------------------------

def test_resolve_no_preference_returns_none():
    assert _resolve_expected_ctc("No preference", 0) is None


def test_resolve_exact_value_wins_over_range():
    assert _resolve_expected_ctc("10-15", 18) == 18


def test_resolve_range_uses_midpoint():
    assert _resolve_expected_ctc("15-20", 0) == 17.5


def test_resolve_open_ended_range_uses_floor():
    assert _resolve_expected_ctc("50+", 0) == 50


def test_resolve_unrecognized_range_returns_none():
    assert _resolve_expected_ctc("garbage-value", 0) is None


# ----------------------------------------------------------------
# calculate_salary_score
# ----------------------------------------------------------------

def test_no_ctc_preference_excludes_category():
    job = {"source": "adzuna", "title": "SDE", "description": "8-12 LPA"}
    assert calculate_salary_score(job, "No preference", 0.0) is None


def test_non_india_source_excludes_category():
    # Salary data from non-India-focused sources isn't reliable enough to score.
    job = {"source": "remoteok", "title": "SDE", "description": "8-12 LPA"}
    assert calculate_salary_score(job, "5-10", 0.0) is None


def test_job_salary_meeting_expectation_scores_100():
    job = {"source": "adzuna", "title": "SDE", "description": "Salary: 15-20 LPA"}
    score = calculate_salary_score(job, "No preference", 15.0)
    assert score == 100


def test_job_salary_far_below_expectation_scores_low():
    job = {"source": "adzuna", "title": "SDE", "description": "Salary: 4-5 LPA"}
    score = calculate_salary_score(job, "No preference", 20.0)
    assert score == 10


def test_no_salary_data_in_job_excludes_category():
    job = {"source": "adzuna", "title": "SDE", "description": "Great team culture!"}
    assert calculate_salary_score(job, "No preference", 15.0) is None
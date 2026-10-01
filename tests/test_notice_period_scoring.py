"""
Tests for src/notice_period_scoring.py
"""

from src.notice_period_scoring import calculate_notice_period_score


def test_returns_none_when_job_does_not_mention_urgency():
    job = {"title": "Backend Developer", "description": "Build APIs using Python and Django."}
    assert calculate_notice_period_score(job, "30 days") is None


def test_immediate_joiner_job_scores_full_for_immediately_available_candidate():
    job = {"title": "Backend Developer", "description": "Immediate joiners preferred."}
    assert calculate_notice_period_score(job, "Immediately available") == 100


def test_immediate_joiner_job_penalizes_long_notice_period():
    job = {"title": "Backend Developer", "description": "We need immediate joiners only."}
    assert calculate_notice_period_score(job, "90 days") == 15


def test_immediate_joiner_job_gives_partial_credit_for_short_notice():
    job = {"title": "Backend Developer", "description": "Urgent hiring for this role."}
    assert calculate_notice_period_score(job, "15 days") == 70


def test_unrecognized_notice_period_value_returns_none():
    job = {"title": "Backend Developer", "description": "Immediate joiners preferred."}
    assert calculate_notice_period_score(job, "Sometime next year") is None
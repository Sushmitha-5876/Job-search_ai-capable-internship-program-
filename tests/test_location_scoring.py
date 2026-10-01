"""
Tests for src/location_scoring.py
"""

from src.location_scoring import calculate_location_score


def test_no_preference_at_all_returns_full_score():
    job = {"location": "Mumbai", "source": "adzuna"}
    assert calculate_location_score(job, preferred_location="", work_mode_preference="No preference") == 100


def test_remote_job_satisfies_remote_preference():
    job = {"location": "Anywhere", "source": "remoteok"}
    assert calculate_location_score(job, work_mode_preference="Remote") == 100


def test_remote_preference_penalizes_onsite_job():
    job = {"location": "Bengaluru", "source": "adzuna"}
    score = calculate_location_score(job, work_mode_preference="Remote")
    assert score < 100


def test_onsite_preference_penalizes_remote_job():
    job = {"location": "Anywhere", "source": "remoteok"}
    score = calculate_location_score(job, work_mode_preference="On-site")
    assert score < 100


def test_matching_location_scores_high():
    job = {"location": "Bengaluru, India", "source": "adzuna"}
    score = calculate_location_score(job, preferred_location="Bengaluru")
    assert score == 100


def test_unrelated_location_scores_low():
    job = {"location": "New York, USA", "source": "adzuna"}
    score = calculate_location_score(job, preferred_location="Bengaluru")
    assert score <= 40


def test_unknown_job_location_is_neutral_not_penalized():
    job = {"location": "", "source": "adzuna"}
    score = calculate_location_score(job, preferred_location="Bengaluru")
    assert score == 50
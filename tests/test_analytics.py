"""
Tests for src/analytics.py
"""

from src.analytics import (
    compute_status_counts,
    aggregate_missing_skills,
    aggregate_suitable_roles,
)


def test_compute_status_counts_basic():
    saved_jobs = [
        {"status": "saved"},
        {"status": "saved"},
        {"status": "applied"},
        {"status": "interview"},
    ]
    counts = compute_status_counts(saved_jobs)
    assert counts["saved"] == 2
    assert counts["applied"] == 1
    assert counts["interview"] == 1
    assert counts["offer"] == 0
    assert counts["rejected"] == 0


def test_compute_status_counts_handles_empty_list():
    counts = compute_status_counts([])
    assert counts == {"saved": 0, "applied": 0, "interview": 0, "offer": 0, "rejected": 0}


def test_compute_status_counts_handles_none():
    counts = compute_status_counts(None)
    assert counts == {"saved": 0, "applied": 0, "interview": 0, "offer": 0, "rejected": 0}


def test_compute_status_counts_ignores_unknown_status():
    counts = compute_status_counts([{"status": "archived"}])
    assert sum(counts.values()) == 0


def test_aggregate_missing_skills_counts_and_ranks():
    scored_jobs = [
        {"missing_skills": ["Docker", "Kubernetes"]},
        {"missing_skills": ["Docker"]},
        {"missing_skills": ["AWS"]},
    ]
    result = aggregate_missing_skills(scored_jobs, top_n=2)
    assert result[0] == ("Docker", 2)
    assert len(result) == 2


def test_aggregate_missing_skills_handles_empty_input():
    assert aggregate_missing_skills([]) == []
    assert aggregate_missing_skills(None) == []


def test_aggregate_suitable_roles_only_counts_strong_matches():
    scored_jobs = [
        {"title": "Backend Developer", "final_score": 87},
        {"title": "Backend Developer", "final_score": 72},
        {"title": "Intern", "final_score": 40},
    ]
    result = aggregate_suitable_roles(scored_jobs)
    assert result[0] == ("Backend Developer", 2)
    role_names = [title for title, _ in result]
    assert "Intern" not in role_names


def test_aggregate_suitable_roles_handles_empty_input():
    assert aggregate_suitable_roles([]) == []
    assert aggregate_suitable_roles(None) == []

def test_format_time_ago_just_now():
    from datetime import datetime
    from src.analytics import format_time_ago

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assert format_time_ago(now_str) == "just now"


def test_format_time_ago_minutes():
    from datetime import datetime, timedelta
    from src.analytics import format_time_ago

    ten_min_ago = (datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
    assert format_time_ago(ten_min_ago) == "10 min ago"


def test_format_time_ago_hours():
    from datetime import datetime, timedelta
    from src.analytics import format_time_ago

    three_hours_ago = (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    assert format_time_ago(three_hours_ago) == "3 hours ago"


def test_format_time_ago_handles_invalid_input():
    from src.analytics import format_time_ago

    assert format_time_ago(None) == "recently"
    assert format_time_ago("garbage") == "recently"
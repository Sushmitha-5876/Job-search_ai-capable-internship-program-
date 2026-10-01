"""
Tests for src/company_score.py

This module resolves the "Company Quality" debate: no hardcoded
tier list, just the candidate's own stated preferences.
"""

from src.company_score import calculate_company_preference_score


def test_no_preference_list_excludes_category():
    job = {"company": "Random Startup"}
    assert calculate_company_preference_score(job, []) is None
    assert calculate_company_preference_score(job, None) is None


def test_matching_company_scores_100():
    job = {"company": "Infosys Ltd"}
    assert calculate_company_preference_score(job, ["Infosys"]) == 100


def test_matching_is_case_insensitive():
    job = {"company": "NVIDIA"}
    assert calculate_company_preference_score(job, ["nvidia"]) == 100


def test_non_matching_company_is_neutral_not_penalized():
    job = {"company": "Random Startup"}
    score = calculate_company_preference_score(job, ["Google", "NVIDIA"])
    assert score == 60  # neutral, not a low/penalized score


def test_missing_company_name_is_neutral():
    job = {"company": ""}
    assert calculate_company_preference_score(job, ["Google"]) == 60
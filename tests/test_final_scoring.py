"""
Tests for src/final_scoring.py

Focuses on combine_category_scores() - the dynamic renormalization
logic that's the fix for the original bug (missing data silently
dragging the average down via a fake neutral 50), and the red-flag
score cap in calculate_final_score().
"""

from src.final_scoring import combine_category_scores, RED_FLAG_SCORE_CAP, calculate_final_score


def test_all_categories_present_gives_weighted_average():
    scores = {
        "skill_match": 100,
        "experience": 100,
        "role_relevance": 100,
        "location": 100,
        "salary": 100,
        "technology_domain": 100,
        "company_preference": 100,
        "notice_period": 100,
    }
    final_score, used, excluded = combine_category_scores(scores)
    assert final_score == 100
    assert excluded == []


def test_missing_category_is_excluded_not_zeroed():
    # Salary, experience, company_preference, and notice_period missing
    # entirely - should NOT drag the average down as if they scored 0,
    # and should NOT be filled with a fake neutral 50 either. (Notice
    # period in particular is expected to be missing on MOST jobs,
    # since most listings never state a notice-period requirement.)
    scores = {
        "skill_match": 100,
        "experience": None,
        "role_relevance": 100,
        "location": 100,
        "salary": None,
        "technology_domain": 100,
        "company_preference": None,
        "notice_period": None,
    }
    final_score, used, excluded = combine_category_scores(scores)
    assert final_score == 100  # remaining categories average to 100, not dragged down
    assert set(excluded) == {"experience", "salary", "company_preference", "notice_period"}


def test_out_of_range_value_is_treated_as_invalid():
    scores = {
        "skill_match": 150,  # invalid - out of 0-100 range
        "experience": 80,
        "role_relevance": 80,
        "location": 80,
        "salary": 80,
        "technology_domain": 80,
        "company_preference": 80,
        "notice_period": 80,
    }
    final_score, used, excluded = combine_category_scores(scores)
    assert "skill_match" in excluded
    assert final_score == 80


def test_returns_none_when_every_category_invalid():
    scores = {key: None for key in [
        "skill_match", "experience", "role_relevance",
        "location", "salary", "technology_domain", "company_preference",
        "notice_period",
    ]}
    final_score, used, excluded = combine_category_scores(scores)
    assert final_score is None
    assert used == []


def test_score_is_clamped_between_1_and_100():
    scores = {
        "skill_match": 0,
        "experience": None,
        "role_relevance": None,
        "location": None,
        "salary": None,
        "technology_domain": None,
        "company_preference": None,
        "notice_period": None,
    }
    final_score, used, excluded = combine_category_scores(scores)
    assert final_score == 1  # clamped up from 0, per max(1, ...) in the implementation


def test_job_with_no_company_name_gets_flagged_and_capped():
    # Reproduces the real bug: a job with a strong skill/role/location
    # match but a genuinely missing company name should NOT silently
    # rank as a "Good Match" with a clean risk status - it should be
    # flagged and capped, per the deterministic missing-company-name
    # red flag added to calculate_final_score().
    job = {
        "title": "Junior Software Developer",
        "company": "",  # <-- genuinely missing, as job_fetcher.py now
                         #     produces for sources with no company data
                         #     (previously defaulted to the literal "N/A")
        "location": "Chennai",
        "description": "Java, SQL, Python role.",
    }
    candidate_profile = {"experience": [], "location": "Chennai"}
    questionnaire_answers = {
        "search_location": "Chennai",
        "work_mode": "Hybrid",
        "preferred_companies": ["Salesforce", "Swiggy", "TCS"],
        "notice_period": "30 days",
    }
    gemini_result = {
        "skill_match_score": 85,
        "role_relevance_score": 90,
        "technology_domain_score": 85,
        "matching_skills": ["Java", "SQL", "Python"],
        "missing_skills": [],
        "red_flags": [],  # Gemini found nothing - this is the realistic case
        "reason": "Strong technical match.",
    }

    result = calculate_final_score(job, candidate_profile, questionnaire_answers, gemini_result)

    assert "No company name listed for this job" in result["red_flags"]
    assert result["score_capped"] is True
    assert result["final_score"] == RED_FLAG_SCORE_CAP


def test_job_with_real_company_name_is_not_flagged():
    job = {
        "title": "Junior Software Developer",
        "company": "TCS",
        "location": "Chennai",
        "description": "Java, SQL, Python role.",
    }
    candidate_profile = {"experience": [], "location": "Chennai"}
    questionnaire_answers = {
        "search_location": "Chennai",
        "work_mode": "Hybrid",
        "preferred_companies": ["Salesforce", "Swiggy", "TCS"],
        "notice_period": "30 days",
    }
    gemini_result = {
        "skill_match_score": 85,
        "role_relevance_score": 90,
        "technology_domain_score": 85,
        "matching_skills": ["Java", "SQL", "Python"],
        "missing_skills": [],
        "red_flags": [],
        "reason": "Strong technical match.",
    }

    result = calculate_final_score(job, candidate_profile, questionnaire_answers, gemini_result)

    assert "No company name listed for this job" not in result["red_flags"]
    assert result["score_capped"] is False
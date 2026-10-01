"""
final_scoring.py
=================
This is where every independent scoring signal built in Steps 1-3
finally comes together into ONE real final score:

    Gemini-judged (from scorer.score_job_with_gemini):
        - skill_match_score        weight 30
        - role_relevance_score     weight 15
        - technology_domain_score  weight 10

    Python-computed, deterministic:
        - experience score (experience_scoring.py)   weight 20
        - location score   (location_scoring.py)     weight 10
        - salary score     (salary_scoring.py)        weight 10
        - company preference (company_score.py)       weight 5
        - notice period    (notice_period_scoring.py) weight 5

The weights above don't need to add up to exactly 100 — see
combine_category_scores() below: it divides by total_weight (the sum
of weights for categories that actually HAVE a valid score for this
job), not by a fixed 100. So the numbers are relative importance, not
percentages of a fixed pie. There's still no "Job Security" category
(no data source currently tells us contract-vs-permanent), and rather
than force a fake weight onto data that doesn't exist, it's simply
not allocated at all.

DYNAMIC WEIGHT RENORMALIZATION (Step 4):
Any category can independently be None — most commonly Experience
(candidate's years couldn't be parsed) or Salary (job has no salary
data, or isn't from an India-focused source). When a category is
None, it is REMOVED from both the weighted sum AND the weight total —
its weight is not redistributed with a fake substitute score, it's
simply excluded, which has the mathematical effect of the remaining
categories automatically accounting for 100% of what's left. This is
the fix for the old bug where missing salary silently dragged down
the average via a fake neutral 50.

RED FLAGS AS A SCORE CAP (Step 5):
If Gemini's red_flags list is non-empty, the final score is capped at
RED_FLAG_SCORE_CAP regardless of how well everything else matched. A
technically well-matched scam listing is worse than a mediocre real
job, not marginally worse — so this is a hard ceiling, not a small
weighted deduction.
"""

from .scorer import get_recommendation
from .experience_scoring import calculate_experience_score
from .location_scoring import calculate_location_score
from .salary_scoring import calculate_salary_score
from .company_score import calculate_company_preference_score
from .notice_period_scoring import calculate_notice_period_score


CATEGORY_WEIGHTS = {
    "skill_match": 30,
    "experience": 20,
    "role_relevance": 15,
    "location": 10,
    "salary": 10,
    "technology_domain": 10,
    "company_preference": 5,
    "notice_period": 5,
}

RED_FLAG_SCORE_CAP = 30


def _valid_score(value):
    """Rejects None, but also anything non-numeric or out of the 0-100 range."""
    return isinstance(value, (int, float)) and 0 <= value <= 100


def combine_category_scores(category_scores):
    """
    Weighted average over whatever categories actually have a valid
    score. Invalid values (None, non-numeric, or out of 0-100 range)
    are excluded entirely (renormalization), not treated as zero or
    as some fake neutral value.

    Args:
        category_scores: dict like {"skill_match": 85, "salary": None, ...}
                          — keys must match CATEGORY_WEIGHTS.

    Returns:
        (final_score: int or None, categories_used: list, categories_excluded: list)
        final_score is None only if EVERY category was invalid (shouldn't
        normally happen, since Gemini's 3 categories are near-always present).
    """
    weighted_sum = 0
    total_weight = 0
    used = []
    excluded = []

    for category, weight in CATEGORY_WEIGHTS.items():
        score = category_scores.get(category)
        if not _valid_score(score):
            excluded.append(category)
            continue
        weighted_sum += score * weight
        total_weight += weight
        used.append(category)

    if total_weight == 0:
        return None, used, excluded

    final_score = round(weighted_sum / total_weight)
    final_score = max(1, min(100, final_score))
    return final_score, used, excluded


def calculate_final_score(job, candidate_profile, questionnaire_answers, gemini_result):
    """
    The main entry point: takes a job, the candidate profile, the
    questionnaire answers (from Step 3's Streamlit form), and the
    already-computed Gemini result (from scorer.score_job_with_gemini),
    and returns the complete final scoring breakdown.

    Args:
        job: job dict
        candidate_profile: dict from resume_parser.extract_candidate_profile()
        questionnaire_answers: dict from st.session_state.questionnaire_data
                                (expects keys: search_location, work_mode,
                                expected_ctc_range, expected_ctc_exact —
                                falls back gracefully if any are missing)
        gemini_result: dict returned by scorer.score_job_with_gemini()

    Returns:
        dict with the final_score, every individual category score,
        which categories were used vs excluded, and red-flag info.
    """
    questionnaire_answers = questionnaire_answers or {}

    experience_score, candidate_years, required_years = calculate_experience_score(
        candidate_profile, job
    )

    location_score = calculate_location_score(
        job,
        preferred_location=questionnaire_answers.get("search_location", ""),
        work_mode_preference=questionnaire_answers.get("work_mode", "No preference"),
    )

    salary_score = calculate_salary_score(
        job,
        expected_ctc_range=questionnaire_answers.get("expected_ctc_range", "No preference"),
        expected_ctc_exact=questionnaire_answers.get("expected_ctc_exact", 0.0),
    )

    company_preference_score = calculate_company_preference_score(
        job,
        preferred_companies=questionnaire_answers.get("preferred_companies", []),
    )

    notice_period_score = calculate_notice_period_score(
        job,
        notice_period=questionnaire_answers.get("notice_period", "Immediately available"),
    )

    category_scores = {
        "skill_match": gemini_result.get("skill_match_score"),
        "experience": experience_score,
        "role_relevance": gemini_result.get("role_relevance_score"),
        "location": location_score,
        "salary": salary_score,
        "technology_domain": gemini_result.get("technology_domain_score"),
        "company_preference": company_preference_score,
        "notice_period": notice_period_score,
    }

    raw_score, categories_used, categories_excluded = combine_category_scores(category_scores)

    # ------------------------------------------------------
    # Deterministic red flag: missing company name.
    #
    # WHY THIS IS HERE (not left to Gemini): the job's "company" field
    # is passed into the Gemini prompt as plain text, but the prompt
    # only gives Gemini EXAMPLES of red flags (payment scams, vague
    # descriptions, unrealistic salary) — it never explicitly instructs
    # Gemini to treat a missing/blank company name as suspicious on its
    # own. That made it unreliable in practice: a job with no company
    # name at all could still come back with an empty red_flags list
    # and a "CLEAR" risk status. This check is a plain Python fact
    # check ("is this field empty?") — exactly the kind of thing that
    # should be deterministic rather than left to an LLM's judgment,
    # per this project's overall design principle (see docstring above
    # this function). This was also explicitly requested during
    # planning as one of the concrete red-flag categories.
    # ------------------------------------------------------
    gemini_red_flags = gemini_result.get("red_flags", []) or []
    company_name = str(job.get("company", "")).strip()
    if not company_name:
        red_flags = gemini_red_flags + ["No company name listed for this job"]
    else:
        red_flags = gemini_red_flags

    score_capped = False
    final_score = raw_score

    if red_flags and raw_score is not None and raw_score > RED_FLAG_SCORE_CAP:
        final_score = RED_FLAG_SCORE_CAP
        score_capped = True

    if final_score is not None:
        recommendation = get_recommendation(final_score)
        if score_capped:
            recommendation = f"⚠️ Red Flag ({recommendation})"
    else:
        recommendation = "Insufficient Data"

    return {
        "final_score": final_score,
        "raw_score_before_cap": raw_score,
        "recommendation": recommendation,
        "category_scores": category_scores,
        "categories_used": categories_used,
        "categories_excluded": categories_excluded,
        "red_flags": red_flags,
        "score_capped": score_capped,
        "candidate_years": candidate_years,
        "required_years": required_years,
    }
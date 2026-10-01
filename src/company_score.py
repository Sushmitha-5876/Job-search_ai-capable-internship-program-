"""
company_score.py
=================
Deterministic (Python-only, no Gemini) company-preference scoring.

This is the resolution to the Company Quality debate from the team
discussion: instead of hardcoding a "which companies are good" tier
list (biased toward brand names, doesn't scale, unfairly tanks
unknown-but-good companies), this scores a job against the
CANDIDATE'S OWN stated preferred companies from the questionnaire.
It's personalization, not a system-wide opinion about company
prestige — so it gives a Nvidia/IBM-style preference exactly when
the candidate actually has one, without baking bias into the
algorithm for everyone else.

Uses fuzzy text matching (RapidFuzz), same tool as location_scoring.py,
since comparing two company names doesn't need an LLM call.
"""

from rapidfuzz import fuzz


MATCH_THRESHOLD = 80  # similarity score (0-100) to count as "same company"
NO_MATCH_SCORE = 60   # neutral, not a penalty — see docstring below


def _normalize(text):
    if not text:
        return ""
    return str(text).strip().lower()


# Placeholder strings some job sources use when they genuinely have no
# company name. Treated identically to an empty string — see the
# missing-company-name handling in calculate_company_preference_score().
_MISSING_COMPANY_PLACEHOLDERS = {"n/a", "na", "not available", "unknown", "none"}


def calculate_company_preference_score(job, preferred_companies=None):
    """
    Args:
        job: job dict (needs 'company')
        preferred_companies: list[str] from the questionnaire,
                              e.g. ["Google", "NVIDIA", "IBM"]

    Returns:
        int score 0-100, or None if the candidate gave no preferred
        companies at all — there's nothing to score against, so this
        category is excluded from the weighted average entirely
        (same pattern as salary_scoring.py and experience_scoring.py
        when their inputs are missing).

        If a preference list WAS given:
        - 100 if the job's company fuzzy-matches one of them
        - 60 (neutral, not a penalty) if it doesn't — not being on
          the candidate's wish list isn't a red flag, so this must
          never drag the score down the way a real mismatch would.
    """
    if not preferred_companies:
        return None

    job_company = _normalize(job.get("company", ""))
    if not job_company or job_company in _MISSING_COMPANY_PLACEHOLDERS:
        return NO_MATCH_SCORE  # can't compare, so stay neutral

    for preferred in preferred_companies:
        preferred_norm = _normalize(preferred)
        if not preferred_norm:
            continue
        similarity = fuzz.partial_ratio(preferred_norm, job_company)
        if similarity >= MATCH_THRESHOLD:
            return 100

    return NO_MATCH_SCORE
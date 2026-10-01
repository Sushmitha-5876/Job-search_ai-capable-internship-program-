"""
location_scoring.py
====================
Deterministic (Python-only, no Gemini) location/work-mode scoring.

Compares a job's location against the candidate's preferred location
and work-mode preference — both collected in the Streamlit
questionnaire (app.py). Uses fuzzy text matching (RapidFuzz), not an
LLM call, since comparing two place names doesn't need reasoning.
"""

from rapidfuzz import fuzz


def normalize_location(text):
    if not text:
        return ""
    return str(text).strip().lower()


def is_remote_job(job):
    """
    True if the job is clearly remote — RemoteOK jobs are remote by
    definition; other sources may say so in the location text or an
    (optional) work-model field.
    """
    if job.get("source") == "remoteok":
        return True

    work_model = normalize_location(job.get("_work_model", ""))
    if work_model in {"remote", "fully remote", "wfh"}:
        return True

    return "remote" in normalize_location(job.get("location", ""))


def calculate_location_score(job, preferred_location="", work_mode_preference="No preference"):
    """
    Args:
        job: job dict (needs 'location'; optionally 'source', '_work_model')
        preferred_location: string from the questionnaire, e.g. "Bengaluru"
        work_mode_preference: "No preference" | "Remote" | "Hybrid" | "On-site"

    Returns:
        int score 0-100. If the candidate gave no location preference at
        all, returns 100 — there's nothing to penalize against, so this
        category shouldn't drag the score down for someone who didn't care.
    """
    preferred_location = normalize_location(preferred_location)
    job_is_remote = is_remote_job(job)

    # A stated work-mode mismatch is a real signal, but not an automatic
    # zero — a strong on-site role can still be worth surfacing even if
    # it's not the candidate's first preference.
    work_mode_penalty = 0
    if work_mode_preference == "Remote" and not job_is_remote:
        work_mode_penalty = 30
    elif work_mode_preference == "On-site" and job_is_remote:
        work_mode_penalty = 20

    if job_is_remote and work_mode_preference in {"No preference", "Remote", "Hybrid"}:
        base_score = 100
    elif not preferred_location:
        base_score = 100
    else:
        job_location = normalize_location(job.get("location", ""))
        if not job_location:
            base_score = 50  # unknown location: neutral, not penalized as if it were a mismatch
        else:
            similarity = fuzz.partial_ratio(preferred_location, job_location)
            if similarity >= 90:
                base_score = 100
            elif similarity >= 70:
                base_score = 70
            elif similarity >= 50:
                base_score = 40
            else:
                base_score = 10

    return max(0, min(100, base_score - work_mode_penalty))
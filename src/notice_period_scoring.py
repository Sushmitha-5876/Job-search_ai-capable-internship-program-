"""
notice_period_scoring.py
=========================
Deterministic (Python-only, no Gemini) notice-period/availability
scoring.

WHY THIS EXISTS: the questionnaire (app.py) already collects the
candidate's notice period, but until now it was saved into
questionnaire_data and then never read by any scoring module —
a dead field. This module closes that gap.

WHY MOST JOBS WILL SCORE None HERE: unlike salary or experience,
most job descriptions never mention a notice-period requirement at
all. This module only produces a real score when the job listing
text ACTUALLY states an urgency signal (e.g. "immediate joiners
preferred", "urgent hiring"). If it doesn't, this returns None —
same "don't fake missing data" pattern used everywhere else
(salary_scoring.py, experience_scoring.py, company_score.py) — so
this category gets excluded from that job's weighted average
rather than silently dragging the score down or up.
"""

import re


# Candidate's selectbox option -> number of days until they can join.
NOTICE_PERIOD_DAYS = {
    "Immediately available": 0,
    "15 days": 15,
    "30 days": 30,
    "60 days": 60,
    "90 days": 90,
}

# Phrases that signal an employer wants someone who can join fast.
# Deliberately conservative — only clear, common phrasings, to avoid
# false positives on unrelated JD text.
_IMMEDIATE_JOINER_PATTERN = re.compile(
    r"\b(?:immediate\s+joiners?|immediately\s+available|"
    r"urgent(?:ly)?\s+(?:hiring|required)|"
    r"candidates?\s+who\s+can\s+join\s+immediately)\b",
    re.IGNORECASE,
)


def _job_wants_immediate_joiner(job):
    """Returns True if the job text explicitly asks for fast joining."""
    text = " ".join(
        str(job.get(field, "") or "")
        for field in ["title", "description"]
    )
    return bool(_IMMEDIATE_JOINER_PATTERN.search(text))


def calculate_notice_period_score(job, notice_period="Immediately available"):
    """
    Args:
        job: job dict (needs 'title'/'description' text to check for
             an urgency signal)
        notice_period: string from the questionnaire, e.g. "30 days"

    Returns:
        int score 0-100, or None if the job listing doesn't state any
        notice-period/urgency requirement — there's nothing to score
        against, so this category is excluded from the weighted
        average entirely (same pattern as salary/experience/company).

        If the job DOES ask for immediate joiners:
        - 100 if the candidate is immediately available
        - 70 if their notice period is short (1-15 days)
        - 40 if it's moderate (16-30 days)
        - 15 if it's long (31+ days) — a real mismatch worth
          reflecting, but not a hard zero, since some employers are
          still flexible in practice.
    """
    if not _job_wants_immediate_joiner(job):
        return None

    candidate_days = NOTICE_PERIOD_DAYS.get(notice_period)
    if candidate_days is None:
        return None  # unrecognized value — treat as no data rather than guess

    if candidate_days == 0:
        return 100
    if candidate_days <= 15:
        return 70
    if candidate_days <= 30:
        return 40
    return 15
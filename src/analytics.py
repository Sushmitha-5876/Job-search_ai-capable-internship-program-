"""
analytics.py
=============
Pure, testable aggregation functions for the Analytics dashboard.

IMPORTANT HONESTY NOTE: these only summarize data that's ACTUALLY
available:
    - Real saved-job statuses from SQLite (via database.get_saved_jobs())
    - The CURRENT session's scored jobs (if a search has been run)

This deliberately does NOT fabricate historical trend data (e.g. "jobs
found per day over the last month", "top companies searched over
time") - that would require a search-history table that doesn't exist
yet. Faking those numbers to visually match a mockup would violate
this whole project's "don't guess at missing data" design principle
(see final_scoring.py's dynamic renormalization for the same idea
applied to scoring). If historical trends are wanted later, the right
fix is to add a search_history table and log each search - not to
invent numbers here.
"""

from collections import Counter


STATUS_KEYS = ["saved", "applied", "interview", "offer", "rejected"]


def format_time_ago(timestamp_str):
    """
    Turns a real SQLite timestamp string ("2026-09-12 11:29:35") into
    a short "X min ago" / "X hours ago" / "X days ago" label - using
    the ACTUAL time it was logged, not a fabricated value.
    """
    from datetime import datetime, timezone

    try:
        logged_at = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return "recently"

    delta = datetime.now() - logged_at
    seconds = delta.total_seconds()

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} min ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(seconds // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def compute_status_counts(saved_jobs):
    """
    Counts saved jobs by status.

    Args:
        saved_jobs: list of dicts from database.get_saved_jobs()

    Returns:
        dict with every status key always present (0 if none), e.g.
        {"saved": 3, "applied": 2, "interview": 1, "offer": 0, "rejected": 0}
    """
    counts = {key: 0 for key in STATUS_KEYS}

    for job in saved_jobs or []:
        status = job.get("status", "saved")
        if status in counts:
            counts[status] += 1

    return counts


def aggregate_missing_skills(scored_jobs, top_n=6):
    """
    Counts how often each skill appears in "missing_skills" across a
    set of already-scored jobs (e.g. the current session's search
    results), most common first.

    Args:
        scored_jobs: list of job dicts, each optionally with a
                     "missing_skills" list (as produced during Gemini
                     scoring - see scorer.py / match_card.py)
        top_n: how many top skills to return

    Returns:
        list of (skill, count) tuples, most frequent first. Empty
        list if scored_jobs is empty or none had missing_skills.
    """
    counter = Counter()

    for job in scored_jobs or []:
        for skill in job.get("missing_skills", []) or []:
            counter[skill] += 1

    return counter.most_common(top_n)


def aggregate_suitable_roles(scored_jobs, top_n=5, score_threshold=70):
    """
    Counts how often each job title appears among scored jobs that
    met or exceeded score_threshold ("suitable" matches), most
    common first.

    Args:
        scored_jobs: list of job dicts with 'title' and 'final_score'
        top_n: how many top roles to return
        score_threshold: minimum final_score to count as "suitable"

    Returns:
        list of (title, count) tuples, most frequent first.
    """
    counter = Counter()

    for job in scored_jobs or []:
        score = job.get("final_score") or 0
        title = job.get("title")
        if score >= score_threshold and title:
            counter[title] += 1

    return counter.most_common(top_n)
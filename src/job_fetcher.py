"""
job_fetcher.py
==============
Fetches, filters, and deduplicates live job listings.

Job sources:
    - RemoteOK (called once per session — its public API ignores query params anyway)
    - Adzuna (called once per generated query — supports real filtering, ~1,000/month free)
    - JobsPipe (called once per session — 100 credits/month free tier)

JSearch was removed: its RapidAPI subscription returned 404s and was
never successfully working, so it added complexity with no real benefit.

The notebook or Streamlit app is responsible for:
    - getting candidate_profile
    - getting search_location
    - displaying results

This module only performs job-search operations.

All HTTP calls go through _send_request(): per-source rate limiting plus
retry/backoff on transient failures (see "POLITE REQUESTS" below).
"""

import os
import re
import threading
import time

import requests

from .skill_matching import find_matching_skills


# ==========================================
# POLITE REQUESTS: RATE LIMITING + RETRY
# ==========================================
#
# Every job-source call goes through _send_request() so the app is a
# well-behaved API client:
#   - a minimum gap between consecutive calls to the SAME source
#     (free tiers are small and easy to trip),
#   - a few retries with exponential backoff on transient failures
#     (timeouts, connection errors, HTTP 429 and 5xx), honouring a
#     numeric Retry-After header when the server sends one.
# Permanent problems (e.g. JobsPipe's 402 "quota exceeded", 4xx auth
# errors) are NOT retried - repeating them would only burn quota.

MIN_REQUEST_INTERVAL_SECONDS = {
    "remoteok": 1.0,
    "adzuna": 0.5,
    "jobspipe": 1.0,
}
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5
MAX_RETRY_AFTER_SECONDS = 10.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_last_request_time = {}
_rate_limit_lock = threading.Lock()


def _pause(seconds):
    """Thin wrapper around time.sleep so tests can skip real waiting."""
    if seconds > 0:
        time.sleep(seconds)


def _wait_for_turn(source):
    """Blocks just long enough to keep MIN_REQUEST_INTERVAL_SECONDS
    between two calls to the same source (thread-safe)."""
    interval = MIN_REQUEST_INTERVAL_SECONDS.get(source, 0)

    with _rate_limit_lock:
        last = _last_request_time.get(source)
        if last is not None and interval:
            remaining = interval - (time.monotonic() - last)
            if remaining > 0:
                _pause(remaining)
        _last_request_time[source] = time.monotonic()


def _retry_delay(response, attempt):
    """Seconds to wait before retry number attempt+1."""
    header = None
    try:
        header = response.headers.get("Retry-After")
    except Exception:
        header = None

    if isinstance(header, str):
        try:
            return min(float(header), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass

    return RETRY_BACKOFF_SECONDS * (2 ** attempt)


def _send_request(source, method, url, **kwargs):
    """
    Sends one HTTP request (method is requests.get / requests.post) with
    rate limiting and retry/backoff. Returns the final response - the
    caller still applies its own status-code handling. If every attempt
    fails with a connection error or timeout, the last exception is
    re-raised so the caller's existing error handling runs unchanged.
    """
    for attempt in range(MAX_RETRIES + 1):
        _wait_for_turn(source)

        try:
            response = method(url, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"{source}: {type(exc).__name__}, retrying (attempt {attempt + 2}/{MAX_RETRIES + 1})")
            _pause(RETRY_BACKOFF_SECONDS * (2 ** attempt))
            continue

        if response.status_code in _RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
            print(
                f"{source}: HTTP {response.status_code}, retrying "
                f"(attempt {attempt + 2}/{MAX_RETRIES + 1})"
            )
            _pause(_retry_delay(response, attempt))
            continue

        return response



# ==========================================
# HELPERS
# ==========================================

def normalize_query_text(value):
    """Convert a value to clean lowercase text."""
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower()
    )


def build_search_queries(candidate_profile):
    """Build job-search queries from the candidate's possible roles."""

    roles = []

    for role in candidate_profile.get("possible_roles", []):
        role = normalize_query_text(role)

        if role and role not in roles:
            roles.append(role)

    standard_roles = [
        "software engineer",
        "software developer",
        "backend developer",
        "frontend developer",
        "full stack developer",
        "python developer",
        "java developer",
        "web developer",
        "application developer",
        "junior software engineer",
    ]

    queries = []

    for role in roles + standard_roles:
        if role and role not in queries:
            queries.append(role)

    return queries[:10]


def get_candidate_skills(candidate_profile):
    """Return technical candidate skills while ignoring soft skills."""

    ignored = {
        "agile",
        "scrum",
        "kanban",
        "communication",
        "leadership",
        "teamwork",
        "problem solving",
        "problem-solving",
        "critical thinking",
        "time management",
        "project management",
    }

    skills = []

    for skill in candidate_profile.get("skills", []):
        skill = normalize_query_text(skill)

        if skill and skill not in ignored:
            skills.append(skill)

    return skills


# ==========================================
# JOB RELEVANCE FILTER
# ==========================================

def is_relevant_job(job, candidate_profile):
    """
    Determine whether a job is relevant to the candidate.

    Returns:
        (accepted, role_matches, skill_matches)
    """

    title = normalize_query_text(
        job.get("title", "")
    )

    description = normalize_query_text(
        job.get("description", "")
    )

    tags = job.get("tags", [])

    if isinstance(tags, list):
        tags_text = " ".join(
            normalize_query_text(tag)
            for tag in tags
        )
    else:
        tags_text = normalize_query_text(tags)

    candidate_roles = [
        normalize_query_text(role)
        for role in candidate_profile.get("possible_roles", [])
        if normalize_query_text(role)
    ]

    candidate_skills = get_candidate_skills(
        candidate_profile
    )

    blocked = {
        "marketing",
        "sales representative",
        "sales executive",
        "recruiter",
        "recruiting",
        "human resources",
        "customer service",
        "customer support",
        "administration",
        "administrator",
        "accountant",
        "accounting",
        "nurse",
        "doctor",
        "teacher",
        "teaching",
        "professor",
        "content writer",
        "copywriter",
        "graphic designer",
        "retail associate",
        "cashier",
        "chef",
        "barista",
        "waiter",
        "housekeeping",
        "security guard",
        "security officer",
        "warehouse worker",
        "driver",
        "lawyer",
    }

    if any(word in title for word in blocked):
        return False, [], []

    role_matches = [
        role
        for role in candidate_roles
        if role in title
    ]

    software_terms = {
        "software engineer",
        "software developer",
        "backend developer",
        "backend engineer",
        "frontend developer",
        "frontend engineer",
        "full stack developer",
        "full-stack developer",
        "web developer",
        "web engineer",
        "java developer",
        "java engineer",
        "python developer",
        "python engineer",
        "application developer",
        "application engineer",
        "mobile developer",
        "devops engineer",
        "cloud engineer",
        "platform engineer",
        "data engineer",
        "machine learning engineer",
        "ml engineer",
        "programmer",
        "automation engineer",
    }

    software_title_match = any(
        term in title
        for term in software_terms
    )

    searchable_text = (
        f"{title} "
        f"{description} "
        f"{tags_text}"
    )

    # Uses fuzzy + synonym matching (src/skill_matching.py) instead of
    # exact substring checks — a candidate with "JavaScript" now
    # correctly matches a job requiring "JS", and vice versa. This was
    # previously the #1 cause of relevant jobs being silently dropped
    # before they ever reached scoring.
    skill_matches = find_matching_skills(candidate_skills, searchable_text)

    accepted = (
        bool(role_matches)
        or software_title_match
        or len(skill_matches) >= 2
    )

    return (
        accepted,
        role_matches,
        skill_matches
    )


# ==========================================
# REMOTEOK
# ==========================================

def fetch_remoteok_jobs(
    query,
    candidate_profile
):
    """Fetch and filter jobs from RemoteOK."""

    query_text = normalize_query_text(query)

    if not query_text:
        return []

    try:
        response = _send_request(
            "remoteok",
            requests.get,
            "https://remoteok.com/api",
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=30,
        )

        print(
            f"RemoteOK HTTP status: "
            f"{response.status_code}"
        )

        response.raise_for_status()
        data = response.json()

    except Exception as exc:
        print(
            f"RemoteOK - {query_text}: "
            f"failed ({exc})"
        )
        return []

    raw_jobs = [
        job
        for job in data
        if isinstance(job, dict)
        and job.get("id")
    ]

    print(
        f"RemoteOK - {query_text}: "
        f"{len(raw_jobs)} jobs received"
    )

    relevant = []

    for job in raw_jobs:

        normalized_job = {
            "title": (
                job.get("position")
                or job.get("title")
                or ""
            ),
            "description": (
                job.get("description")
                or ""
            ),
            "tags": (
                job.get("tags", [])
                if isinstance(
                    job.get("tags"),
                    list
                )
                else []
            ),
        }

        accepted, role_matches, skill_matches = (
            is_relevant_job(
                normalized_job,
                candidate_profile
            )
        )

        if not accepted:
            continue

        relevant.append(
            {
                "source": "remoteok",
                "title": (
                    job.get("position")
                    or job.get("title")
                    or "N/A"
                ),
                "company": (
                    job.get("company")
                    or ""
                ),
                "location": (
                    job.get("location")
                    or "Remote"
                ),
                "url": (
                    job.get("url")
                    or ""
                ),
                "description": (
                    job.get("description")
                    or ""
                ),
                "tags": (
                    job.get("tags", [])
                    if isinstance(
                        job.get("tags"),
                        list
                    )
                    else []
                ),
                "raw": job,
                "_reason": (
                    "role_match"
                    if role_matches
                    else "technical_skills"
                ),
                "_role_matches": role_matches,
                "_technical_matches": skill_matches,
            }
        )

    print(
        f"RemoteOK - {query_text}: "
        f"{len(relevant)} relevant"
    )

    return relevant


# ==========================================
# ADZUNA
# ==========================================

def fetch_adzuna_jobs(
    query,
    location,
    candidate_profile
):
    """Fetch and filter jobs from Adzuna."""

    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        print(
            "Adzuna skipped: "
            "credentials not configured"
        )
        return []

    try:
        response = _send_request(
            "adzuna",
            requests.get,
            "https://api.adzuna.com/v1/api/jobs/in/search/1",
            params={
                "app_id": app_id,
                "app_key": app_key,
                "results_per_page": 20,
                "what": query,
                "where": location or "India",
                "content-type": "application/json",
            },
            timeout=30,
        )

        print(
            f"Adzuna HTTP status: "
            f"{response.status_code}"
        )

        response.raise_for_status()
        data = response.json()

    except Exception as exc:
        print(
            f"Adzuna - {query}: "
            f"failed ({exc})"
        )
        return []

    results = data.get(
        "results",
        []
    )

    relevant = []

    for item in results:

        job = {
            "title": (
                item.get("title")
                or "N/A"
            ),
            "company": (
                item.get("company", {})
                or {}
            ).get(
                "display_name",
                ""
            ),
            "location": (
                item.get("location", {})
                or {}
            ).get(
                "display_name",
                ""
            ),
            "url": (
                item.get("redirect_url")
                or ""
            ),
            "description": (
                item.get("description")
                or ""
            ),
            "tags": [],
        }

        accepted, role_matches, skill_matches = (
            is_relevant_job(
                job,
                candidate_profile
            )
        )

        if not accepted:
            continue

        job["source"] = "adzuna"
        job["raw"] = item

        job["_reason"] = (
            "role_match"
            if role_matches
            else "technical_skills"
        )

        job["_role_matches"] = role_matches
        job["_technical_matches"] = skill_matches

        relevant.append(job)

    print(
        f"Adzuna - {query}: "
        f"{len(results)} received, "
        f"{len(relevant)} relevant"
    )

    return relevant


# ==========================================
# JOBSPIPE
# ==========================================

def fetch_jobspipe_jobs(
    query,
    location,
    candidate_profile
):
    """
    Fetch and filter jobs from JobsPipe.

    JobsPipe uses:
        POST https://api.jobspipe.dev/v1/jobs/search

    Authentication:
        Authorization: Bearer <JOBSPIPE_API_KEY>
    """

    api_key = os.getenv("JOBSPIPE_API_KEY")

    if not api_key:
        print(
            "JobsPipe skipped: "
            "API key not configured"
        )
        return []

    query_text = normalize_query_text(query)

    if not query_text:
        return []

    body = {
        "job_title_or": [query_text],
        "job_country_code_or": ["IN"],
        "limit": 25,
        "include_total_results": True,
    }

    if location:
        body["job_location_or"] = [location]

    try:
        response = _send_request(
            "jobspipe",
            requests.post,
            "https://api.jobspipe.dev/v1/jobs/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        )

        print(
            f"JobsPipe HTTP status: "
            f"{response.status_code}"
        )

        if response.status_code == 402:
            print(
                "JobsPipe: monthly quota exceeded."
            )
            return []

        if response.status_code == 429:
            print(
                "JobsPipe: rate limit reached."
            )
            return []

        response.raise_for_status()
        data = response.json()

    except requests.exceptions.RequestException as exc:
        print(
            f"JobsPipe - {query_text}: "
            f"failed ({exc})"
        )
        return []

    results = data.get(
        "data",
        []
    )

    print(
        f"JobsPipe - {query_text}: "
        f"{len(results)} jobs received"
    )

    relevant = []

    for item in results:

        job = {
            "title": (
                item.get("job_title")
                or item.get("normalized_title")
                or "N/A"
            ),
            "company": (
                item.get("company")
                or ""
            ),
            "location": (
                item.get("location")
                or item.get("short_location")
                or "India"
            ),
            "url": (
                item.get("final_url")
                or item.get("url")
                or item.get("source_url")
                or ""
            ),
            "description": (
                item.get("description")
                or ""
            ),
            "tags": (
                item.get("keyword_slugs", [])
                if isinstance(
                    item.get("keyword_slugs"),
                    list
                )
                else []
            ),
        }

        accepted, role_matches, skill_matches = (
            is_relevant_job(
                job,
                candidate_profile
            )
        )

        if not accepted:
            continue

        job["source"] = "jobspipe"
        job["raw"] = item

        job["_reason"] = (
            "role_match"
            if role_matches
            else "technical_skills"
        )

        job["_role_matches"] = role_matches
        job["_technical_matches"] = skill_matches

        relevant.append(job)

    print(
        f"JobsPipe - {query_text}: "
        f"{len(relevant)} relevant"
    )

    return relevant


# ==========================================
# DEDUPLICATION
# ==========================================

def dedupe_jobs(jobs):
    """Remove duplicate jobs using URL or title + company."""

    seen = set()
    unique = []

    for job in jobs:

        url = normalize_query_text(
            job.get("url")
        )

        title = normalize_query_text(
            job.get("title")
        )

        company = normalize_query_text(
            job.get("company")
        )

        if url:
            key = (
                "url",
                url
            )

        elif title and company:
            key = (
                "title_company",
                title,
                company
            )

        else:
            continue

        if key in seen:
            continue

        seen.add(key)
        unique.append(job)

    return unique


def filter_jobs_missing_company(jobs):
    """
    Drops any job with no identifiable company name.

    WHY THIS EXISTS: previously, jobs with no company name (a source
    simply didn't provide one) still went through pre-ranking and a
    full Gemini scoring call, then showed up in results as "N/A" with
    a normal-looking score - confusing, since most of that score was
    computed from title/description/location text that genuinely had
    nothing to do with the missing company field. Filtering these out
    BEFORE pre-ranking means:
        - no wasted Gemini API call scoring a job you'll never trust
        - the final results list is 100% jobs with a real, named
          employer, leaving room for more comparable, complete listings
        - no need for a runtime red-flag/score-cap workaround, since
          the job never reaches scoring at all

    NOTE: dedupe_jobs() above already silently drops a company-less
    job if it ALSO has no URL (its dedup key needs title+company as a
    fallback when there's no URL). This filter catches the common case
    dedupe misses: a job WITH a valid apply URL but no company name -
    those pass dedup fine today, so this filter is still needed.
    """
    kept = []
    dropped_count = 0

    for job in jobs:
        company = normalize_query_text(job.get("company"))

        if company:
            kept.append(job)
        else:
            dropped_count += 1

    if dropped_count:
        print(
            f"Filtered out {dropped_count} job(s) with no listed "
            f"company name."
        )

    return kept


# ==========================================
# FETCH ALL JOBS
# ==========================================

def fetch_all_jobs(
    candidate_profile,
    search_location="",
    return_breakdown=False
):
    """
    Fetch jobs from all configured sources.

    Sources:
        RemoteOK
        Adzuna
        JobsPipe

    Args:
        return_breakdown: when True, also return a dict of real
            per-source and pipeline-stage counts (see below) instead
            of just the final job list. Defaults to False so every
            existing caller (app.py, the notebook) keeps working
            unchanged.

    Returns:
        - return_breakdown=False (default): list of unique relevant jobs.
        - return_breakdown=True: (jobs, breakdown) tuple, where breakdown
          is a dict with keys "remoteok", "adzuna", "jobspipe" (raw counts
          returned by each source), "before_dedupe", "after_dedupe", and
          "final" (after the no-company-name filter — same as len(jobs)).
    """

    queries = build_search_queries(
        candidate_profile
    )

    location = normalize_query_text(
        search_location
    )

    if not location:
        location = normalize_query_text(
            candidate_profile.get(
                "location",
                ""
            )
        )

    print(
        "\nGenerated search queries:"
    )

    for query in queries:
        print(
            f"- {query}"
        )

    all_jobs = []

    remoteok_returned = 0
    adzuna_returned = 0
    jobspipe_returned = 0

    top_query = queries[0] if queries else "software engineer"

    # ======================================
    # REMOTEOK  (called ONCE per session, not per query)
    # ======================================
    # RemoteOK's public API has no search/query parameter — it always
    # returns the same ~100 latest listings regardless of what you ask
    # for. Looping this 10x (once per generated query) was hitting the
    # exact same static dataset 10 times, which is wasted network calls
    # and was likely what caused the one connection drop we saw in
    # testing. Filtering by candidate_profile happens locally either way.

    remote_jobs = fetch_remoteok_jobs(
        top_query,
        candidate_profile
    )

    remoteok_returned += len(
        remote_jobs
    )

    all_jobs.extend(
        remote_jobs
    )

    # ======================================
    # ADZUNA
    # ======================================
    # Adzuna DOES support real query filtering per search term, and its
    # free tier (~1,000/month) supports looping — so this one stays
    # per-query.

    for query in queries:

        adzuna_jobs = fetch_adzuna_jobs(
            query,
            location,
            candidate_profile
        )

        adzuna_returned += len(
            adzuna_jobs
        )

        all_jobs.extend(
            adzuna_jobs
        )

    # ======================================
    # JOBSPIPE  (called ONCE per session, not per query)
    # ======================================
    # JobsPipe's free tier is 100 credits/month. Looping this over all
    # ~10 search queries would burn 10 credits per single resume search
    # — only ~10 test runs before the monthly quota is gone. Calling it
    # once with the top query keeps this source usable all month.

    jobspipe_jobs = fetch_jobspipe_jobs(
        top_query,
        location,
        candidate_profile
    )

    jobspipe_returned += len(
        jobspipe_jobs
    )

    all_jobs.extend(
        jobspipe_jobs
    )

    # ======================================
    # DEDUPLICATION
    # ======================================

    before_dedupe = len(
        all_jobs
    )

    combined = dedupe_jobs(
        all_jobs
    )

    before_company_filter = len(combined)

    combined = filter_jobs_missing_company(
        combined
    )

    # ======================================
    # SOURCE SUMMARY
    # ======================================

    print(
        "\n=== JOB SOURCE SUMMARY ==="
    )

    print(
        f"RemoteOK: "
        f"requested=1, "
        f"returned={remoteok_returned}"
    )

    print(
        f"Adzuna: "
        f"requested={len(queries)}, "
        f"returned={adzuna_returned}"
    )

    print(
        f"JobsPipe: "
        f"requested=1, "
        f"returned={jobspipe_returned}"
    )

    print(
        f"Total jobs before deduplication: "
        f"{before_dedupe}"
    )

    print(
        f"Total jobs after deduplication: "
        f"{before_company_filter}"
    )

    print(
        f"Total jobs after removing listings with no company name: "
        f"{len(combined)}"
    )

    if not return_breakdown:
        return combined

    breakdown = {
        "remoteok": remoteok_returned,
        "adzuna": adzuna_returned,
        "jobspipe": jobspipe_returned,
        "before_dedupe": before_dedupe,
        "after_dedupe": before_company_filter,
        "final": len(combined),
    }

    return combined, breakdown
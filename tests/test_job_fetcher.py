"""
Tests for src/job_fetcher.py's filter_jobs_missing_company()
"""

from src.job_fetcher import filter_jobs_missing_company


def test_drops_jobs_with_no_company():
    jobs = [
        {"title": "Backend Developer", "company": "TCS"},
        {"title": "Junior Software Developer", "company": ""},
        {"title": "Data Analyst", "company": None},
    ]
    result = filter_jobs_missing_company(jobs)
    assert len(result) == 1
    assert result[0]["company"] == "TCS"


def test_keeps_all_jobs_when_every_one_has_a_company():
    jobs = [
        {"title": "Backend Developer", "company": "TCS"},
        {"title": "Data Analyst", "company": "Infosys"},
    ]
    result = filter_jobs_missing_company(jobs)
    assert len(result) == 2


def test_empty_input_returns_empty_list():
    assert filter_jobs_missing_company([]) == []


def test_whitespace_only_company_is_treated_as_missing():
    jobs = [{"title": "Backend Developer", "company": "   "}]
    result = filter_jobs_missing_company(jobs)
    assert result == []

"""
Tests for fetch_all_jobs()'s return_breakdown option.

These mock the 3 network-calling source functions (fetch_remoteok_jobs,
fetch_adzuna_jobs, fetch_jobspipe_jobs) so no real API calls happen -
we're testing that the BREAKDOWN MATH is correct, not that the live
APIs work (that's an integration concern, verified by running the app).
"""

from unittest.mock import patch
from src.job_fetcher import fetch_all_jobs


def _job(title, company, url):
    return {"title": title, "company": company, "url": url, "location": "Remote", "description": ""}


@patch("src.job_fetcher.fetch_jobspipe_jobs")
@patch("src.job_fetcher.fetch_adzuna_jobs")
@patch("src.job_fetcher.fetch_remoteok_jobs")
def test_return_breakdown_reports_real_per_source_counts(mock_remoteok, mock_adzuna, mock_jobspipe):
    mock_remoteok.return_value = [_job("Backend Dev", "Acme", "http://a.com/1")]
    # Adzuna is called once PER generated query (several queries per
    # search) - return jobs on the first call only, empty after, so
    # the total returned is deterministic regardless of query count.
    mock_adzuna.side_effect = [
        [
            _job("Software Engineer", "Infosys", "http://a.com/2"),
            _job("Data Analyst", "TCS", "http://a.com/3"),
        ]
    ] + [[]] * 50
    mock_jobspipe.return_value = [_job("Java Developer", "Wipro", "http://a.com/4")]

    candidate_profile = {"possible_roles": ["Software Engineer"], "skills": ["Python"], "location": "Bengaluru"}

    jobs, breakdown = fetch_all_jobs(candidate_profile, search_location="Bengaluru", return_breakdown=True)

    assert breakdown["remoteok"] == 1
    assert breakdown["adzuna"] == 2
    assert breakdown["jobspipe"] == 1
    assert breakdown["before_dedupe"] == 4
    assert breakdown["final"] == len(jobs)


@patch("src.job_fetcher.fetch_jobspipe_jobs")
@patch("src.job_fetcher.fetch_adzuna_jobs")
@patch("src.job_fetcher.fetch_remoteok_jobs")
def test_return_breakdown_reflects_deduplication(mock_remoteok, mock_adzuna, mock_jobspipe):
    duplicate = _job("Backend Dev", "Acme", "http://same-url.com/1")
    mock_remoteok.return_value = [duplicate]
    # Same URL returned once by Adzuna too (first call only) - should
    # be deduped against the RemoteOK copy.
    mock_adzuna.side_effect = [[duplicate]] + [[]] * 50
    mock_jobspipe.return_value = []

    candidate_profile = {"possible_roles": ["Backend Dev"], "skills": [], "location": ""}

    jobs, breakdown = fetch_all_jobs(candidate_profile, return_breakdown=True)

    assert breakdown["before_dedupe"] == 2
    assert breakdown["after_dedupe"] == 1
    assert len(jobs) == 1


@patch("src.job_fetcher.fetch_jobspipe_jobs")
@patch("src.job_fetcher.fetch_adzuna_jobs")
@patch("src.job_fetcher.fetch_remoteok_jobs")
def test_default_call_without_return_breakdown_still_returns_plain_list(mock_remoteok, mock_adzuna, mock_jobspipe):
    mock_remoteok.return_value = [_job("Backend Dev", "Acme", "http://a.com/1")]
    mock_adzuna.return_value = []
    mock_jobspipe.return_value = []

    candidate_profile = {"possible_roles": ["Backend Dev"], "skills": [], "location": ""}

    result = fetch_all_jobs(candidate_profile)  # no return_breakdown - old call signature

    assert isinstance(result, list)
    assert len(result) == 1
"""
tests/test_job_fetcher_network.py
==================================
Mocked-network tests for the 3 live job-source fetchers
(fetch_remoteok_jobs, fetch_adzuna_jobs, fetch_jobspipe_jobs).

WHY THIS FILE EXISTS: test_job_fetcher.py already mocks these three
functions as black boxes to test fetch_all_jobs()'s aggregation math.
This file goes one level deeper and mocks requests.get/requests.post
directly, so we can verify each fetcher's own behavior without ever
touching the real network:
    - the request is built correctly (URL, params/body, headers)
    - missing credentials short-circuit before any network call
    - non-2xx / provider-specific error status codes fail soft
      (return [], never raise) instead of crashing the app
    - a raw provider response is correctly normalized into this
      app's internal job dict shape

None of this requires GEMINI_API_KEY, ADZUNA_APP_ID, etc. to be set
in the environment - responses.Mock/requests.Response is faked
entirely in-process.
"""

from unittest.mock import patch, MagicMock

import requests

from src.job_fetcher import (
    fetch_remoteok_jobs,
    fetch_adzuna_jobs,
    fetch_jobspipe_jobs,
)


CANDIDATE_PROFILE = {
    "possible_roles": ["software engineer"],
    "skills": ["python", "sql"],
    "location": "Bengaluru",
}


def _mock_response(status_code=200, json_data=None, raise_for_status_error=None):
    """Builds a fake requests.Response good enough for these fetchers."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data if json_data is not None else {}

    if raise_for_status_error:
        mock_resp.raise_for_status.side_effect = raise_for_status_error
    else:
        mock_resp.raise_for_status.return_value = None

    return mock_resp


# ==========================================================
# REMOTEOK
# ==========================================================

class TestFetchRemoteOKJobs:

    @patch("src.job_fetcher.requests.get")
    def test_successful_response_is_normalized(self, mock_get):
        mock_get.return_value = _mock_response(
            status_code=200,
            json_data=[
                {"id": "1", "position": "Software Engineer", "company": "Acme",
                 "description": "Build things", "tags": ["python"], "url": "http://x/1"},
            ],
        )

        result = fetch_remoteok_jobs("software engineer", CANDIDATE_PROFILE)

        assert len(result) == 1
        assert result[0]["title"] == "Software Engineer"
        assert result[0]["company"] == "Acme"
        assert result[0]["source"] == "remoteok"

    @patch("src.job_fetcher.requests.get")
    def test_request_uses_expected_url_and_headers(self, mock_get):
        mock_get.return_value = _mock_response(status_code=200, json_data=[])

        fetch_remoteok_jobs("software engineer", CANDIDATE_PROFILE)

        args, kwargs = mock_get.call_args
        assert args[0] == "https://remoteok.com/api"
        assert "User-Agent" in kwargs["headers"]

    @patch("src.job_fetcher.requests.get")
    def test_non_2xx_status_fails_soft(self, mock_get):
        mock_get.return_value = _mock_response(
            status_code=503,
            raise_for_status_error=requests.exceptions.HTTPError("503 Server Error"),
        )

        result = fetch_remoteok_jobs("software engineer", CANDIDATE_PROFILE)
        assert result == []

    @patch("src.job_fetcher.requests.get")
    def test_connection_error_fails_soft(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("no network")

        result = fetch_remoteok_jobs("software engineer", CANDIDATE_PROFILE)
        assert result == []

    @patch("src.job_fetcher.requests.get")
    def test_malformed_json_fails_soft(self, mock_get):
        mock_resp = _mock_response(status_code=200)
        mock_resp.json.side_effect = ValueError("not json")
        mock_get.return_value = mock_resp

        result = fetch_remoteok_jobs("software engineer", CANDIDATE_PROFILE)
        assert result == []

    def test_empty_query_returns_empty_without_network_call(self):
        with patch("src.job_fetcher.requests.get") as mock_get:
            result = fetch_remoteok_jobs("", CANDIDATE_PROFILE)
            assert result == []
            mock_get.assert_not_called()


# ==========================================================
# ADZUNA
# ==========================================================

class TestFetchAdzunaJobs:

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.job_fetcher.requests.get")
    def test_missing_credentials_short_circuits_before_network_call(self, mock_get):
        result = fetch_adzuna_jobs("python developer", "Bengaluru", CANDIDATE_PROFILE)

        assert result == []
        mock_get.assert_not_called()

    @patch.dict("os.environ", {"ADZUNA_APP_ID": "id123", "ADZUNA_APP_KEY": "key456"})
    @patch("src.job_fetcher.requests.get")
    def test_request_params_include_credentials_and_query(self, mock_get):
        mock_get.return_value = _mock_response(status_code=200, json_data={"results": []})

        fetch_adzuna_jobs("python developer", "Bengaluru", CANDIDATE_PROFILE)

        args, kwargs = mock_get.call_args
        assert args[0] == "https://api.adzuna.com/v1/api/jobs/in/search/1"
        assert kwargs["params"]["app_id"] == "id123"
        assert kwargs["params"]["app_key"] == "key456"
        assert kwargs["params"]["what"] == "python developer"
        assert kwargs["params"]["where"] == "Bengaluru"

    @patch.dict("os.environ", {"ADZUNA_APP_ID": "id123", "ADZUNA_APP_KEY": "key456"})
    @patch("src.job_fetcher.requests.get")
    def test_missing_location_falls_back_to_india(self, mock_get):
        mock_get.return_value = _mock_response(status_code=200, json_data={"results": []})

        fetch_adzuna_jobs("python developer", "", CANDIDATE_PROFILE)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["where"] == "India"

    @patch.dict("os.environ", {"ADZUNA_APP_ID": "id123", "ADZUNA_APP_KEY": "key456"})
    @patch("src.job_fetcher.requests.get")
    def test_successful_response_is_normalized(self, mock_get):
        mock_get.return_value = _mock_response(
            status_code=200,
            json_data={
                "results": [
                    {
                        "title": "Software Engineer",
                        "company": {"display_name": "Infosys"},
                        "location": {"display_name": "Bengaluru, India"},
                        "redirect_url": "http://x/2",
                        "description": "Build backend services",
                    }
                ]
            },
        )

        result = fetch_adzuna_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        assert len(result) == 1
        assert result[0]["company"] == "Infosys"
        assert result[0]["source"] == "adzuna"

    @patch.dict("os.environ", {"ADZUNA_APP_ID": "id123", "ADZUNA_APP_KEY": "key456"})
    @patch("src.job_fetcher.requests.get")
    def test_server_error_fails_soft(self, mock_get):
        mock_get.return_value = _mock_response(
            status_code=500,
            raise_for_status_error=requests.exceptions.HTTPError("500 Server Error"),
        )

        result = fetch_adzuna_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)
        assert result == []

    @patch.dict("os.environ", {"ADZUNA_APP_ID": "id123", "ADZUNA_APP_KEY": "key456"})
    @patch("src.job_fetcher.requests.get")
    def test_timeout_fails_soft(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        result = fetch_adzuna_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)
        assert result == []


# ==========================================================
# JOBSPIPE
# ==========================================================

class TestFetchJobsPipeJobs:

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.job_fetcher.requests.post")
    def test_missing_api_key_short_circuits_before_network_call(self, mock_post):
        result = fetch_jobspipe_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        assert result == []
        mock_post.assert_not_called()

    @patch.dict("os.environ", {"JOBSPIPE_API_KEY": "jp_key_123"})
    @patch("src.job_fetcher.requests.post")
    def test_request_body_and_auth_header_are_correct(self, mock_post):
        mock_post.return_value = _mock_response(status_code=200, json_data={"data": []})

        fetch_jobspipe_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.jobspipe.dev/v1/jobs/search"
        assert kwargs["headers"]["Authorization"] == "Bearer jp_key_123"
        assert kwargs["json"]["job_title_or"] == ["software engineer"]
        assert kwargs["json"]["job_country_code_or"] == ["IN"]
        assert kwargs["json"]["job_location_or"] == ["Bengaluru"]

    @patch.dict("os.environ", {"JOBSPIPE_API_KEY": "jp_key_123"})
    @patch("src.job_fetcher.requests.post")
    def test_quota_exceeded_402_fails_soft(self, mock_post):
        mock_post.return_value = _mock_response(status_code=402)

        result = fetch_jobspipe_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)
        assert result == []

    @patch.dict("os.environ", {"JOBSPIPE_API_KEY": "jp_key_123"})
    @patch("src.job_fetcher.requests.post")
    def test_rate_limited_429_fails_soft(self, mock_post):
        mock_post.return_value = _mock_response(status_code=429)

        result = fetch_jobspipe_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)
        assert result == []

    @patch.dict("os.environ", {"JOBSPIPE_API_KEY": "jp_key_123"})
    @patch("src.job_fetcher.requests.post")
    def test_successful_response_is_normalized(self, mock_post):
        mock_post.return_value = _mock_response(
            status_code=200,
            json_data={
                "data": [
                    {
                        "job_title": "Software Engineer",
                        "company": "Wipro",
                        "location": "Bengaluru",
                        "final_url": "http://x/3",
                        "description": "Backend role",
                    }
                ]
            },
        )

        result = fetch_jobspipe_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        assert len(result) == 1
        assert result[0]["company"] == "Wipro"
        assert result[0]["source"] == "jobspipe"

    @patch.dict("os.environ", {"JOBSPIPE_API_KEY": "jp_key_123"})
    @patch("src.job_fetcher.requests.post")
    def test_connection_error_fails_soft(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("no network")

        result = fetch_jobspipe_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)
        assert result == []

# ==========================================================
# RATE LIMITING + RETRY/BACKOFF (_send_request / _wait_for_turn)
# ==========================================================

import src.job_fetcher as job_fetcher

_ADZUNA_ENV = {"ADZUNA_APP_ID": "id123", "ADZUNA_APP_KEY": "key456"}
_GOOD_ADZUNA_BODY = {
    "results": [
        {
            "title": "Software Engineer",
            "company": {"display_name": "Infosys"},
            "location": {"display_name": "Bengaluru, India"},
            "redirect_url": "http://x/2",
            "description": "Build backend services",
        }
    ]
}


class TestRetryAndBackoff:

    @patch.dict("os.environ", _ADZUNA_ENV)
    @patch("src.job_fetcher._pause")
    @patch("src.job_fetcher.requests.get")
    def test_transient_503_is_retried_then_succeeds(self, mock_get, mock_pause):
        mock_get.side_effect = [
            _mock_response(status_code=503),
            _mock_response(status_code=200, json_data=_GOOD_ADZUNA_BODY),
        ]

        result = fetch_adzuna_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        assert mock_get.call_count == 2
        assert len(result) == 1
        assert result[0]["company"] == "Infosys"

    @patch.dict("os.environ", _ADZUNA_ENV)
    @patch("src.job_fetcher._pause")
    @patch("src.job_fetcher.requests.get")
    def test_persistent_503_stops_after_max_retries_and_fails_soft(self, mock_get, mock_pause):
        mock_get.return_value = _mock_response(
            status_code=503,
            raise_for_status_error=requests.exceptions.HTTPError("503"),
        )

        result = fetch_adzuna_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        assert result == []
        assert mock_get.call_count == job_fetcher.MAX_RETRIES + 1

    @patch.dict("os.environ", _ADZUNA_ENV)
    @patch("src.job_fetcher._pause")
    @patch("src.job_fetcher.requests.get")
    def test_timeout_is_retried_then_succeeds(self, mock_get, mock_pause):
        mock_get.side_effect = [
            requests.exceptions.Timeout("slow"),
            _mock_response(status_code=200, json_data=_GOOD_ADZUNA_BODY),
        ]

        result = fetch_adzuna_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        assert mock_get.call_count == 2
        assert len(result) == 1

    @patch.dict("os.environ", _ADZUNA_ENV)
    @patch("src.job_fetcher._pause")
    @patch("src.job_fetcher.requests.get")
    def test_client_errors_are_not_retried(self, mock_get, mock_pause):
        mock_get.return_value = _mock_response(
            status_code=401,
            raise_for_status_error=requests.exceptions.HTTPError("401"),
        )

        result = fetch_adzuna_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        assert result == []
        assert mock_get.call_count == 1

    @patch.dict("os.environ", {"JOBSPIPE_API_KEY": "jp_key_123"})
    @patch("src.job_fetcher._pause")
    @patch("src.job_fetcher.requests.post")
    def test_jobspipe_quota_402_is_never_retried(self, mock_post, mock_pause):
        mock_post.return_value = _mock_response(status_code=402)

        result = fetch_jobspipe_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        assert result == []
        assert mock_post.call_count == 1

    @patch.dict("os.environ", {"JOBSPIPE_API_KEY": "jp_key_123"})
    @patch("src.job_fetcher._pause")
    @patch("src.job_fetcher.requests.post")
    def test_jobspipe_429_is_retried_before_giving_up(self, mock_post, mock_pause):
        mock_post.return_value = _mock_response(status_code=429)

        result = fetch_jobspipe_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        assert result == []
        assert mock_post.call_count == job_fetcher.MAX_RETRIES + 1

    @patch.dict("os.environ", _ADZUNA_ENV)
    @patch("src.job_fetcher._pause")
    @patch("src.job_fetcher.requests.get")
    def test_backoff_grows_between_retries(self, mock_get, mock_pause):
        mock_get.return_value = _mock_response(
            status_code=503,
            raise_for_status_error=requests.exceptions.HTTPError("503"),
        )

        fetch_adzuna_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        waits = [call.args[0] for call in mock_pause.call_args_list]
        # Only the retry backoffs (not the min-interval gap) are 1.5s / 3.0s.
        assert job_fetcher.RETRY_BACKOFF_SECONDS in waits
        assert job_fetcher.RETRY_BACKOFF_SECONDS * 2 in waits

    @patch.dict("os.environ", _ADZUNA_ENV)
    @patch("src.job_fetcher._pause")
    @patch("src.job_fetcher.requests.get")
    def test_numeric_retry_after_header_is_honoured_and_capped(self, mock_get, mock_pause):
        slow = _mock_response(status_code=429)
        slow.headers = {"Retry-After": "4"}
        very_slow = _mock_response(status_code=429)
        very_slow.headers = {"Retry-After": "9999"}
        mock_get.side_effect = [
            slow,
            very_slow,
            _mock_response(status_code=200, json_data=_GOOD_ADZUNA_BODY),
        ]

        result = fetch_adzuna_jobs("software engineer", "Bengaluru", CANDIDATE_PROFILE)

        waits = [call.args[0] for call in mock_pause.call_args_list]
        assert 4.0 in waits
        assert job_fetcher.MAX_RETRY_AFTER_SECONDS in waits
        assert len(result) == 1


class TestRateLimiting:

    def test_second_call_to_same_source_waits_out_the_minimum_gap(self):
        with patch("src.job_fetcher._pause") as mock_pause:
            job_fetcher._last_request_time.clear()

            job_fetcher._wait_for_turn("adzuna")     # first call: no wait
            assert mock_pause.call_count == 0

            job_fetcher._wait_for_turn("adzuna")     # immediately again: must wait
            assert mock_pause.call_count == 1
            waited = mock_pause.call_args.args[0]
            assert 0 < waited <= job_fetcher.MIN_REQUEST_INTERVAL_SECONDS["adzuna"]

    def test_different_sources_do_not_wait_on_each_other(self):
        with patch("src.job_fetcher._pause") as mock_pause:
            job_fetcher._last_request_time.clear()

            job_fetcher._wait_for_turn("adzuna")
            job_fetcher._wait_for_turn("jobspipe")
            job_fetcher._wait_for_turn("remoteok")

            assert mock_pause.call_count == 0
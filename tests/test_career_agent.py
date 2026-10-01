"""
Tests for src/career_agent.py

What's tested here:
    - extract_location_from_query() - pure logic, no network needed.
    - job_search_tool / company_research_tool are correctly built as
      LangChain tools (have the right name, are callable, etc.)
    - get_career_agent() successfully WIRES UP an AgentExecutor with
      a fake API key, proving the LangChain plumbing (LLM + tools +
      prompt + agent) is assembled correctly.

What's NOT tested here, and why:
    - Actually calling Gemini or Adzuna. Both require a live network
      call to a real API key/service - that's an integration concern,
      not a unit test, and it would make these tests slow, flaky, and
      dependent on account quotas. Building the agent without
      crashing is what a unit test should verify; whether the live
      API responds correctly is verified by running the app itself.
"""

from src.career_agent import (
    extract_location_from_query,
    extract_answer_text,
    job_search_tool,
    company_research_tool,
    get_career_agent,
)


def test_extract_location_finds_known_city_case_insensitively():
    location, remaining = extract_location_from_query("python jobs in pune")
    assert location == "Pune"


def test_extract_location_common_spelling_not_in_city_list_is_a_known_gap():
    # "Bangalore" (the common spelling) isn't in INDIAN_CITIES, which
    # lists "Bengaluru" instead - so this currently returns no match.
    # Documenting this as a known limitation rather than hiding it.
    location, remaining = extract_location_from_query("python jobs in Bangalore")
    assert location is None


def test_extract_location_exact_city_name_match():
    location, remaining = extract_location_from_query("python jobs in Pune")
    assert location == "Pune"
    assert "pune" not in remaining


def test_extract_location_returns_none_when_no_city_mentioned():
    location, remaining = extract_location_from_query("remote data analyst roles")
    assert location is None
    assert remaining == "remote data analyst roles"


def test_extract_answer_text_passes_through_a_plain_string():
    assert extract_answer_text("Hello, here's your answer.") == (
        "Hello, here's your answer."
    )


def test_extract_answer_text_unwraps_a_single_text_block():
    # This is the exact shape that was breaking the chat: a list of
    # content blocks instead of a plain string, with an internal
    # Gemini 'signature' token that must NOT end up in the answer.
    content = [
        {
            "type": "text",
            "text": "Finwizard Technology Pvt. Ltd. is a fintech company.",
            "extras": {"signature": "El4KXAERTTIPHUrHBK+rHECIMb"},
        }
    ]
    result = extract_answer_text(content)
    assert result == "Finwizard Technology Pvt. Ltd. is a fintech company."
    assert "signature" not in result
    assert "El4KXAERTTIPHUrHBK" not in result


def test_extract_answer_text_joins_multiple_text_blocks():
    content = [
        {"type": "text", "text": "First part."},
        {"type": "text", "text": "Second part."},
    ]
    assert extract_answer_text(content) == "First part.\nSecond part."


def test_extract_answer_text_drops_exact_duplicate_blocks():
    # Reproduces a real observed bug: some langchain_google_genai +
    # Gemini response shapes return the same answer as more than one
    # "text" block (e.g. one from an intermediate tool-calling turn,
    # one from the final turn) - joining both with "\n" rendered a
    # clean paragraph immediately followed by a garbled-looking
    # duplicate underneath it in the chat UI.
    content = [
        {"type": "text", "text": "Microsoft's interview process is rigorous."},
        {"type": "text", "text": "Microsoft's interview process is rigorous."},
    ]
    assert extract_answer_text(content) == "Microsoft's interview process is rigorous."


def test_extract_answer_text_drops_near_duplicate_blocks_with_different_whitespace():
    content = [
        {"type": "text", "text": "Microsoft's interview process is rigorous."},
        {"type": "text", "text": "  Microsoft's   interview process is rigorous.  "},
    ]
    assert extract_answer_text(content) == "Microsoft's interview process is rigorous."


def test_extract_answer_text_falls_back_safely_on_unexpected_shape():
    # Shouldn't crash even if content is something neither a string
    # nor a list of text blocks - just stringify rather than error.
    assert extract_answer_text(None) == "None"
    assert extract_answer_text(42) == "42"


def test_job_search_tool_is_a_langchain_tool():
    assert job_search_tool.name == "job_search_tool"
    assert callable(job_search_tool.func)


def test_company_research_tool_is_a_langchain_tool():
    assert company_research_tool.name == "company_research_tool"
    assert callable(company_research_tool.func)


def test_get_career_agent_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.setattr("src.career_agent.GEMINI_API_KEY", None)
    try:
        get_career_agent()
        assert False, "Expected a RuntimeError when no API key is set"
    except RuntimeError as exc:
        assert "GEMINI_API_KEY" in str(exc)


def test_get_career_agent_builds_successfully_with_a_key(monkeypatch):
    # A fake key is enough to prove the LangChain wiring (LLM + tools +
    # system prompt + agent graph) assembles correctly - constructing
    # these objects doesn't make a network call, only .invoke() would.
    monkeypatch.setattr("src.career_agent.GEMINI_API_KEY", "fake-key-for-testing")

    agent = get_career_agent()

    assert agent is not None
    # create_agent() returns a compiled LangGraph graph with a "tools"
    # node when tools are attached - this confirms both tools were
    # actually wired into the agent, not just imported.
    assert "tools" in agent.nodes


def test_get_career_agent_has_5_tools_without_resume_context(monkeypatch):
    monkeypatch.setattr("src.career_agent.GEMINI_API_KEY", "fake-key-for-testing")
    agent = get_career_agent()
    tool_names = set(agent.nodes["tools"].bound.tools_by_name.keys())
    assert tool_names == {
        "job_search_tool",
        "company_research_tool",
        "save_job_tool",
        "list_saved_jobs_tool",
        "update_application_status_tool",
    }
    assert "match_jobs_to_my_resume_tool" not in tool_names


def test_get_career_agent_adds_resume_tool_when_context_given(monkeypatch):
    monkeypatch.setattr("src.career_agent.GEMINI_API_KEY", "fake-key-for-testing")
    agent = get_career_agent(
        candidate_profile={"possible_roles": ["Software Engineer"], "skills": ["Java"]},
        resume_text="fake resume text",
        questionnaire_data={},
    )
    tool_names = set(agent.nodes["tools"].bound.tools_by_name.keys())
    assert "match_jobs_to_my_resume_tool" in tool_names
    assert len(tool_names) == 6


def test_build_resume_match_tool_calls_the_real_pipeline():
    from unittest.mock import patch, MagicMock
    from src.career_agent import build_resume_match_tool

    candidate_profile = {"possible_roles": ["Software Engineer"], "skills": ["Java"]}
    fake_job = {
        "title": "Backend Engineer", "company": "Acme", "location": "Bengaluru",
        "source": "jobspipe", "skill_match_score": 85, "role_relevance_score": 80,
        "technology_domain_score": 75, "matching_skills": ["Java"], "missing_skills": [],
        "red_flags": [], "reason": "Good fit", "description": "",
    }

    tool = build_resume_match_tool(candidate_profile, "resume text", {"search_location": "Bengaluru"})

    with patch("src.career_agent.fetch_all_jobs", return_value=[fake_job]) as mock_fetch, \
         patch("src.career_agent.pre_rank_jobs", return_value=[fake_job]), \
         patch("src.career_agent.score_jobs_with_gemini", return_value=[dict(fake_job)]), \
         patch("src.career_agent.get_gemini_client", return_value=MagicMock()):
        result = tool.func(location_override="")

    assert mock_fetch.called
    assert "Backend Engineer" in result
    assert "/100" in result


def test_build_resume_match_tool_no_jobs_found_returns_clear_message():
    from unittest.mock import patch
    from src.career_agent import build_resume_match_tool

    tool = build_resume_match_tool({"possible_roles": [], "skills": []}, "resume text", {})

    with patch("src.career_agent.fetch_all_jobs", return_value=[]):
        result = tool.func(location_override="")

    assert "no relevant jobs" in result.lower()


def test_save_job_tool_saves_a_new_job():
    from unittest.mock import patch
    from src.career_agent import save_job_tool

    with patch("src.career_agent.init_db") as mock_init, \
         patch("src.career_agent.save_job", return_value=True) as mock_save:
        result = save_job_tool.func(title="Backend Engineer", company="Acme", url="http://x.com", final_score=85)

    assert mock_init.called
    assert mock_save.called
    assert "Backend Engineer" in result
    assert "Saved" in result


def test_save_job_tool_reports_duplicate_clearly():
    from unittest.mock import patch
    from src.career_agent import save_job_tool

    with patch("src.career_agent.init_db"), patch("src.career_agent.save_job", return_value=False):
        result = save_job_tool.func(title="Backend Engineer", company="Acme")

    assert "already" in result.lower()


def test_list_saved_jobs_tool_handles_empty_list():
    from unittest.mock import patch
    from src.career_agent import list_saved_jobs_tool

    with patch("src.career_agent.init_db"), patch("src.career_agent.get_saved_jobs", return_value=[]):
        result = list_saved_jobs_tool.func()

    assert "haven't saved" in result.lower()


def test_list_saved_jobs_tool_shows_saved_jobs_with_scores():
    from unittest.mock import patch
    from src.career_agent import list_saved_jobs_tool

    fake_jobs = [{"id": 1, "title": "Backend Engineer", "company": "Acme", "final_score": 85, "status": "saved"}]
    with patch("src.career_agent.init_db"), patch("src.career_agent.get_saved_jobs", return_value=fake_jobs):
        result = list_saved_jobs_tool.func()

    assert "Backend Engineer" in result
    assert "85" in result


def test_update_application_status_tool_finds_job_by_partial_case_insensitive_match():
    from unittest.mock import patch
    from src.career_agent import update_application_status_tool

    fake_jobs = [{"id": 1, "title": "Backend Engineer", "company": "Acme", "final_score": 85, "status": "saved"}]
    with patch("src.career_agent.init_db"), \
         patch("src.career_agent.get_saved_jobs", return_value=fake_jobs), \
         patch("src.career_agent.update_job_status") as mock_update:
        result = update_application_status_tool.func(job_title="backend", company="acme", new_status="applied")

    assert mock_update.called
    assert mock_update.call_args[0] == (1, "applied")
    assert "applied" in result.lower()


def test_update_application_status_tool_handles_job_not_found():
    from unittest.mock import patch
    from src.career_agent import update_application_status_tool

    with patch("src.career_agent.init_db"), patch("src.career_agent.get_saved_jobs", return_value=[]):
        result = update_application_status_tool.func(job_title="Nonexistent", company="NoCo", new_status="applied")

    assert "couldn't find" in result.lower()

# ==========================================================
# Speed: shared LLM client, cached agents, candidate context
# ==========================================================

def test_llm_client_is_built_once_and_reused(monkeypatch):
    from unittest.mock import MagicMock
    import src.career_agent as ca

    fake_class = MagicMock(side_effect=lambda **kwargs: object())
    monkeypatch.setattr(ca, "ChatGoogleGenerativeAI", fake_class)
    ca._get_llm.cache_clear()

    first = ca._get_llm("some-model", "key-1", 0.3)
    second = ca._get_llm("some-model", "key-1", 0.3)
    other = ca._get_llm("some-model", "key-2", 0.3)

    assert first is second
    assert other is not first
    assert fake_class.call_count == 2
    ca._get_llm.cache_clear()


def test_llm_client_falls_back_if_max_retries_is_not_supported(monkeypatch):
    import src.career_agent as ca

    class OldClient:
        def __init__(self, model, google_api_key, temperature):
            self.model = model

    monkeypatch.setattr(ca, "ChatGoogleGenerativeAI", OldClient)
    ca._get_llm.cache_clear()
    assert ca._get_llm("m", "k", 0.3).model == "m"
    ca._get_llm.cache_clear()


def test_cached_agent_is_reused_until_candidate_data_changes(monkeypatch):
    import src.career_agent as ca

    monkeypatch.setattr(ca, "GEMINI_API_KEY", "fake-key-for-testing")
    builds = []
    monkeypatch.setattr(
        ca, "get_career_agent",
        lambda *args: builds.append(args) or object(),
    )
    ca.clear_agent_cache()

    profile = {"possible_roles": ["Data Analyst"], "skills": ["SQL"]}
    prefs = {"search_location": "Pune"}

    a1 = ca.get_cached_career_agent(profile, "resume text", prefs)
    a2 = ca.get_cached_career_agent(dict(profile), "resume text", dict(prefs))   # equal data, new objects
    assert a1 is a2
    assert len(builds) == 1

    ca.get_cached_career_agent(profile, "a DIFFERENT resume", prefs)
    ca.get_cached_career_agent(profile, "resume text", {"search_location": "Delhi"})
    ca.get_cached_career_agent(None, None, None)
    assert len(builds) == 4
    ca.clear_agent_cache()


def test_agent_cache_is_bounded(monkeypatch):
    import src.career_agent as ca

    monkeypatch.setattr(ca, "GEMINI_API_KEY", "fake-key-for-testing")
    monkeypatch.setattr(ca, "get_career_agent", lambda *args: object())
    ca.clear_agent_cache()
    for index in range(ca._AGENT_CACHE_MAX + 5):
        ca.get_cached_career_agent({"skills": [str(index)]}, "r", {})
    assert len(ca._agent_cache) == ca._AGENT_CACHE_MAX
    ca.clear_agent_cache()


def test_cached_agent_still_has_every_tool(monkeypatch):
    import src.career_agent as ca

    monkeypatch.setattr(ca, "GEMINI_API_KEY", "fake-key-for-testing")
    ca.clear_agent_cache()
    agent = ca.get_cached_career_agent({"possible_roles": ["QA Engineer"], "skills": ["Selenium"]}, "resume", {})
    names = set(agent.nodes["tools"].bound.tools_by_name.keys())
    assert names == {
        "job_search_tool", "company_research_tool", "save_job_tool",
        "list_saved_jobs_tool", "update_application_status_tool",
        "match_jobs_to_my_resume_tool",
    }
    ca.clear_agent_cache()


def test_cached_agent_without_key_raises_the_same_clear_error(monkeypatch):
    import src.career_agent as ca

    monkeypatch.setattr(ca, "GEMINI_API_KEY", None)
    ca.clear_agent_cache()
    try:
        ca.get_cached_career_agent()
        assert False, "Expected a RuntimeError when no API key is set"
    except RuntimeError as exc:
        assert "GEMINI_API_KEY" in str(exc)


def test_candidate_context_summarises_profile_and_preferences():
    from src.career_agent import build_candidate_context

    context = build_candidate_context(
        {
            "possible_roles": ["Backend Developer"],
            "skills": ["Python", "Django"],
            "experience": ["3 years at Acme building APIs"],
            "education": ["B.Tech CS"],
            "location": "Pune",
        },
        {"work_mode": "Remote", "notice_period": "30 days", "expected_ctc_range": "No preference",
         "preferred_companies": ["Google"]},
    )
    assert "Backend Developer" in context
    assert "Python, Django" in context
    assert "3 years at Acme" in context
    assert "Pune" in context
    assert "work mode: Remote" in context
    assert "No preference" not in context
    assert "Google" in context


def test_candidate_context_is_bounded_and_empty_profile_gives_nothing():
    from src.career_agent import build_candidate_context

    assert build_candidate_context({}, {}) == ""
    assert build_candidate_context(None) == ""
    huge = build_candidate_context({"skills": [f"skill{i}" for i in range(500)],
                                    "experience": ["x" * 5000] * 50})
    assert len(huge) < 4000


def test_system_prompt_carries_the_profile_only_when_a_resume_is_loaded():
    from src.career_agent import build_system_prompt

    with_resume = build_system_prompt(
        candidate_profile={"possible_roles": ["Data Engineer"], "skills": ["Spark"]},
        resume_text="resume", questionnaire_data={},
    )
    assert "Spark" in with_resume
    assert "match_jobs_to_my_resume_tool" in with_resume

    without = build_system_prompt()
    assert "<candidate_profile>" not in without


# ==========================================================
# Streaming
# ==========================================================

class _Msg:
    def __init__(self, type_, content, id_=None):
        self.type, self.content, self.id = type_, content, id_


class _FakeStreamingAgent:
    """Yields what LangGraph yields for stream_mode=['messages', 'values']."""

    def __init__(self, events):
        self.events = events
        self.stream_calls = []

    def stream(self, payload, stream_mode=None):
        self.stream_calls.append((payload, stream_mode))
        yield from self.events

    def invoke(self, payload):
        raise AssertionError("invoke must not be used when streaming works")


def _chunk(text, id_="run-1"):
    return ("messages", (_Msg("AIMessageChunk", text, id_), {"langgraph_node": "model"}))


def test_stream_agent_response_yields_growing_text_then_final():
    from src.career_agent import stream_agent_response

    final_message = _Msg("ai", [{"type": "text", "text": "Hello there!", "extras": {"signature": "SIG"}}])
    agent = _FakeStreamingAgent([
        _chunk("Hello"), _chunk(" there"), _chunk("!"),
        ("values", {"messages": [final_message]}),
    ])

    events = list(stream_agent_response(agent, [{"role": "user", "content": "hi"}]))

    assert events == [
        ("text", "Hello"), ("text", "Hello there"), ("text", "Hello there!"),
        ("final", "Hello there!"),
    ]
    payload, mode = agent.stream_calls[0]
    assert payload == {"messages": [{"role": "user", "content": "hi"}]}
    assert "messages" in mode and "values" in mode


def test_stream_agent_response_shows_only_the_current_model_reply_after_a_tool_call():
    from src.career_agent import stream_agent_response

    tool_result = ("messages", (_Msg("tool", "Found 3 jobs", "tool-1"), {}))
    final_message = _Msg("ai", "Here are 3 jobs.")
    agent = _FakeStreamingAgent([
        _chunk("Let me search.", "run-1"),
        tool_result,
        _chunk("Here are ", "run-2"), _chunk("3 jobs.", "run-2"),
        ("values", {"messages": [final_message]}),
    ])

    events = list(stream_agent_response(agent, []))
    texts = [text for kind, text in events if kind == "text"]

    assert texts == ["Let me search.", "Here are ", "Here are 3 jobs."]
    assert "Found 3 jobs" not in " ".join(texts)          # tool output is not answer text
    assert events[-1] == ("final", "Here are 3 jobs.")


def test_stream_agent_response_final_uses_the_existing_duplicate_block_cleanup():
    from src.career_agent import stream_agent_response

    duplicated = _Msg("ai", [
        {"type": "text", "text": "Microsoft's interview process is rigorous."},
        {"type": "text", "text": "  Microsoft's   interview process is rigorous. "},
    ])
    agent = _FakeStreamingAgent([
        _chunk("Microsoft's interview process is rigorous."),
        ("values", {"messages": [duplicated]}),
    ])
    assert list(stream_agent_response(agent, []))[-1] == ("final", "Microsoft's interview process is rigorous.")


def test_stream_agent_response_falls_back_to_streamed_text_if_final_message_has_none():
    from src.career_agent import stream_agent_response

    agent = _FakeStreamingAgent([
        _chunk("The answer."),
        ("values", {"messages": [_Msg("ai", [{"type": "thinking", "thinking": "..."}])]}),
    ])
    assert list(stream_agent_response(agent, []))[-1] == ("final", "The answer.")


def test_stream_agent_response_handles_block_list_chunks_without_mangling_spaces():
    from src.career_agent import stream_agent_response

    agent = _FakeStreamingAgent([
        _chunk([{"type": "text", "text": "Use "}]),
        _chunk([{"type": "text", "text": "Python."}]),
        ("values", {"messages": [_Msg("ai", "Use Python.")]}),
    ])
    texts = [t for k, t in stream_agent_response(agent, []) if k == "text"]
    assert texts[-1] == "Use Python."


def test_stream_agent_response_uses_invoke_when_streaming_is_off_or_missing(monkeypatch):
    import src.career_agent as ca

    class InvokeOnly:
        def invoke(self, payload):
            return {"messages": [_Msg("ai", [{"type": "text", "text": "Plain answer."}])]}

    assert list(ca.stream_agent_response(InvokeOnly(), [])) == [("final", "Plain answer.")]

    monkeypatch.setattr(ca, "STREAMING_ENABLED", False)

    class Both(_FakeStreamingAgent):
        def invoke(self, payload):
            return {"messages": [_Msg("ai", "From invoke.")]}

    assert list(ca.stream_agent_response(Both([]), [])) == [("final", "From invoke.")]

# ==========================================================
# Per-visitor workspaces: the saved-job tools are session-bound
#
# On a hosted deployment every visitor shares one SQLite file, so an
# agent's saved-job tools may only ever touch the workspace they were
# built for. The LLM never sees or controls the session id.
# ==========================================================

def _tools_bound_to_temp_db(db_path):
    """Patches career_agent's database functions to use a temp database."""
    from functools import partial
    from unittest.mock import patch
    import src.database as db

    db.init_db(db_path)
    return (
        patch("src.career_agent.init_db", lambda: None),
        patch("src.career_agent.save_job", partial(db.save_job, db_path=db_path)),
        patch("src.career_agent.get_saved_jobs", partial(db.get_saved_jobs, db_path=db_path)),
        patch("src.career_agent.update_job_status", partial(db.update_job_status, db_path=db_path)),
    )


def test_saved_job_tools_only_see_and_change_their_own_workspace(tmp_path):
    from src.career_agent import build_saved_jobs_tools

    p1, p2, p3, p4 = _tools_bound_to_temp_db(tmp_path / "t.db")
    with p1, p2, p3, p4:
        alice_save, alice_list, alice_update = build_saved_jobs_tools("alice")
        bob_save, bob_list, bob_update = build_saved_jobs_tools("bob")

        alice_save.func(title="Backend Engineer", company="Acme", url="http://a", final_score=88)

        assert "Backend Engineer" in alice_list.func()
        assert "haven't saved" in bob_list.func().lower()

        # Bob cannot move Alice's job to a new status...
        assert "couldn't find" in bob_update.func(
            job_title="Backend", company="Acme", new_status="rejected"
        ).lower()
        assert "saved" in alice_list.func()          # ...still 'saved', untouched

        # ...but Alice can.
        assert "applied" in alice_update.func(
            job_title="Backend", company="Acme", new_status="applied"
        ).lower()


def test_two_workspaces_can_each_save_the_same_job(tmp_path):
    from src.career_agent import build_saved_jobs_tools

    p1, p2, p3, p4 = _tools_bound_to_temp_db(tmp_path / "t.db")
    with p1, p2, p3, p4:
        alice_save = build_saved_jobs_tools("alice")[0]
        bob_save = build_saved_jobs_tools("bob")[0]

        first = alice_save.func(title="Data Analyst", company="Zoho", url="http://z")
        second = bob_save.func(title="Data Analyst", company="Zoho", url="http://z")
        again = alice_save.func(title="Data Analyst", company="Zoho", url="http://z")

    assert "Saved" in first
    assert "Saved" in second
    assert "already" in again.lower()


def test_the_session_id_is_not_something_the_llm_can_supply():
    from src.career_agent import build_saved_jobs_tools

    for saved_tool in build_saved_jobs_tools("alice"):
        assert "session_id" not in saved_tool.args      # not in the tool's schema


def test_agent_cache_keeps_one_agent_per_workspace(monkeypatch):
    import src.career_agent as ca

    monkeypatch.setattr(ca, "GEMINI_API_KEY", "fake-key-for-testing")
    builds = []
    monkeypatch.setattr(ca, "get_career_agent", lambda *args: builds.append(args) or object())
    ca.clear_agent_cache()

    profile = {"possible_roles": ["Data Analyst"], "skills": ["SQL"]}
    a1 = ca.get_cached_career_agent(profile, "resume", {}, session_id="alice")
    a2 = ca.get_cached_career_agent(profile, "resume", {}, session_id="alice")
    b1 = ca.get_cached_career_agent(profile, "resume", {}, session_id="bob")

    assert a1 is a2                      # same visitor, same data -> reused
    assert b1 is not a1                  # different visitor -> their own agent
    assert len(builds) == 2
    assert [args[3] for args in builds] == ["alice", "bob"]
    ca.clear_agent_cache()


def test_get_career_agent_binds_the_saved_job_tools_to_the_given_workspace(monkeypatch):
    import src.career_agent as ca

    monkeypatch.setattr(ca, "GEMINI_API_KEY", "fake-key-for-testing")
    bound = []
    real_builder = ca.build_saved_jobs_tools
    monkeypatch.setattr(
        ca, "build_saved_jobs_tools",
        lambda session_id=ca.DEFAULT_SESSION_ID: bound.append(session_id) or real_builder(session_id),
    )

    agent = ca.get_career_agent(session_id="visitor-42")

    assert bound == ["visitor-42"]
    names = set(agent.nodes["tools"].bound.tools_by_name.keys())
    assert {"save_job_tool", "list_saved_jobs_tool", "update_application_status_tool"} <= names
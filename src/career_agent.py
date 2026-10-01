"""
career_agent.py
================
A small LangChain agent with two career tools: job search and
company research.

WHY THIS IS SEPARATE FROM THE MAIN PIPELINE: scorer.py, final_scoring.py,
and everything they depend on are already tested and working - this
file does NOT touch, import from, or depend on that scoring logic in
any way that could break it. This is an ADDITIVE feature: a small
conversational "Career Assistant Chat" that sits next to the main
resume-matching flow, not a replacement for it. It exists specifically
to satisfy Track A's "Framework: Standard LangChain with career tools"
requirement.

Two tools:
    - job_search_tool: searches Adzuna for jobs matching a plain-text
      query (e.g. "python developer jobs in Bangalore").
    - company_research_tool: asks Gemini for a short, structured blurb
      about a named company (industry, size, known for, culture notes).

The agent (an LLM + these two tools) decides which tool, if any, a
user's question needs, and returns a natural-language answer.

SPEED: building the LLM client and compiling the agent graph is the
expensive part of a chat turn, so both are built once and reused:
    - _get_llm()                 one shared Gemini chat client
    - get_cached_career_agent()  one agent per distinct candidate
                                 (profile + resume + preferences); it
                                 is only rebuilt when that data changes
    - stream_agent_response()    runs a turn and yields the answer as it
                                 is generated (token streaming)
"""

import functools
import hashlib
import json
import threading
from collections import OrderedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

from .config import GEMINI_API_KEY, MODEL, FALLBACK_MODEL, get_gemini_client
from .gemini_retry import generate_with_retry
from .job_fetcher import fetch_adzuna_jobs, fetch_all_jobs
from .scorer import pre_rank_jobs, score_jobs_with_gemini
from .final_scoring import calculate_final_score
from .database import (
    DEFAULT_SESSION_ID,
    init_db,
    save_job,
    get_saved_jobs,
    update_job_status,
)
from .data.locations import INDIAN_CITIES


def extract_location_from_query(query):
    """
    Finds a known Indian city mentioned in a plain-text query, if any.

    Returns:
        (location, remaining_query) - location is None if no known
        city was found in the text, in which case remaining_query is
        just the original query unchanged.
    """
    query_lower = query.lower()

    for city in INDIAN_CITIES:
        if city.lower() in query_lower:
            remaining = query_lower.replace(city.lower(), "").strip()
            return city, remaining

    return None, query


def extract_answer_text(content):
    """
    Normalizes a LangChain message's .content into a plain string
    ready to show the user.

    WHY THIS EXISTS: .content is USUALLY a plain string, but some
    langchain_google_genai + Gemini model combinations instead return
    a LIST of structured "content blocks", e.g.:

        [{'type': 'text', 'text': 'the actual answer...',
          'extras': {'signature': 'El4KXAE...'}}]

    That 'signature' is an internal Gemini verification token used
    for tool-call integrity checking - it is NOT part of the answer
    and was never meant to be shown to the user. Without this
    function, that entire raw list (signature included) got printed
    straight into the chat, which is the "answer isn't readable" bug
    this fixes.

    Args:
        content: whatever agent.invoke(...)["messages"][-1].content is
                 - either a plain string, or a list of content blocks.

    Returns:
        A plain string: the concatenated 'text' pieces if content is
        a list of blocks, or content unchanged if it's already a
        plain string. Non-text blocks (e.g. tool-call metadata) are
        skipped rather than shown.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_pieces = []
        seen = set()
        for block in content:
            if isinstance(block, str):
                piece = block
            elif isinstance(block, dict) and block.get("type") == "text":
                piece = block.get("text", "")
            else:
                continue

            piece = piece.strip()
            if not piece:
                continue

            # WHY THIS DEDUP EXISTS: some langchain_google_genai + Gemini
            # response shapes return the SAME (or a near-identical,
            # whitespace-differing) answer as more than one "text" block
            # in the same response - e.g. one block from an intermediate
            # tool-calling turn and one from the final turn, both marked
            # type=="text". Joining every block with "\n" then rendered a
            # clean paragraph immediately followed by a garbled duplicate
            # fragment underneath it. Comparing a normalized (lowercased,
            # whitespace-collapsed) version catches near-duplicates that
            # an exact string match would miss.
            normalized = " ".join(piece.lower().split())
            if normalized in seen:
                continue
            seen.add(normalized)

            text_pieces.append(piece)

        if text_pieces:
            return "\n".join(text_pieces)

    # Unexpected shape - fall back to a plain string rather than
    # crashing, but this path shouldn't normally be reached.
    return str(content)


def build_resume_match_tool(candidate_profile, resume_text, questionnaire_data, max_jobs=10):
    """
    Builds a LangChain tool bound to ONE specific candidate's resume
    data via closure. This is what makes the agent capable of actually
    controlling the main resume-based pipeline (fetch -> pre-rank ->
    Gemini scoring -> final weighted score) instead of only having
    access to the lightweight, resume-blind job_search_tool.

    WHY A FACTORY FUNCTION, NOT A PLAIN @tool: LangChain tools are
    plain functions with a fixed signature - they don't take "extra"
    hidden arguments for whichever candidate happens to be using the
    app right now. Building the tool fresh, per-candidate, with their
    specific candidate_profile/resume_text captured in a closure, is
    how the agent gets access to "this user's" resume without needing
    the LLM to somehow pass an entire resume as a tool argument.

    Args:
        candidate_profile: dict from resume_parser.extract_candidate_profile()
        resume_text: the candidate's raw resume text (needed for Gemini's
                     per-job scoring call, same as the main pipeline)
        questionnaire_data: dict of the candidate's stated preferences
                             (location, salary, notice period, etc.) -
                             same dict final_scoring.py already expects
        max_jobs: how many jobs to send through Gemini scoring - kept
                  small by default since this runs inside a chat
                  response, where the user is actively waiting

    Returns:
        A LangChain @tool that runs the REAL pipeline and returns a
        natural-language summary with genuine final scores - not a
        plain job listing like job_search_tool produces.
    """
    questionnaire_data = questionnaire_data or {}

    @tool
    def match_jobs_to_my_resume_tool(location_override: str = "") -> str:
        """
        Runs the candidate's ACTUAL uploaded resume through the full
        job-matching pipeline (search + pre-rank + Gemini scoring +
        the 8-category final score) and returns the top matches with
        their real scores and reasoning. Use this whenever the user
        asks something like "find jobs matching my resume", "what
        jobs fit my profile", or "search for jobs for me" - i.e.
        whenever they want PERSONALIZED results based on their resume,
        not just a generic keyword search.

        Args:
            location_override: optional - if the user mentions a
                specific city in their request (e.g. "in Chennai"),
                pass it here to search that location instead of their
                saved preference.
        """
        search_location = (
            location_override.strip()
            or questionnaire_data.get("search_location", "")
        )

        jobs = fetch_all_jobs(candidate_profile, search_location=search_location)

        if not jobs:
            return (
                "No relevant jobs were found for your resume right now. "
                "Try a different location, or check back later as new "
                "listings come in."
            )

        top_jobs = pre_rank_jobs(jobs, candidate_profile, max_jobs=max_jobs)

        client = get_gemini_client()
        scored_jobs = score_jobs_with_gemini(top_jobs, resume_text, client, MODEL)

        for scored_job in scored_jobs:
            final_result = calculate_final_score(
                scored_job, candidate_profile, questionnaire_data, scored_job
            )
            scored_job.update(final_result)
            scored_job["match_score"] = scored_job["final_score"]

        scored_jobs.sort(key=lambda j: j.get("match_score", 0), reverse=True)

        lines = [
            f"Ran your resume against {len(jobs)} live jobs "
            f"(scored the top {len(scored_jobs)} with Gemini). "
            f"Here are your best matches:"
        ]
        for job in scored_jobs[:5]:
            warning = " ⚠️ RED FLAG" if job.get("score_capped") else ""
            lines.append(
                f"- {job.get('title', 'N/A')} at {job.get('company', 'N/A')} "
                f"— {job.get('match_score', 0)}/100 "
                f"({job.get('recommendation', '')}){warning}"
            )

        return "\n".join(lines)

    return match_jobs_to_my_resume_tool


@tool
def job_search_tool(query: str) -> str:
    """
    Searches for live jobs matching a plain-text query, e.g.
    "python developer jobs in Bangalore" or "remote data analyst roles".
    Returns a short text summary of the top matching jobs found.
    """
    location, remaining_query = extract_location_from_query(query)
    search_terms = remaining_query or query

    # A lightweight, synthetic "candidate profile" so the existing
    # relevance filter (built for the main resume-matching pipeline)
    # has something to match query terms against - no resume needed
    # for this standalone chat tool.
    synthetic_profile = {
        "possible_roles": [search_terms],
        "skills": search_terms.split(),
    }

    jobs = fetch_adzuna_jobs(
        query=search_terms,
        location=location or "India",
        candidate_profile=synthetic_profile,
    )

    if not jobs:
        return (
            f"No jobs found for '{query}'. Try a broader search term "
            "or a different location."
        )

    top_jobs = jobs[:5]
    lines = [f"Found {len(jobs)} jobs, here are the top {len(top_jobs)}:"]
    for job in top_jobs:
        lines.append(
            f"- {job['title']} at {job['company']} ({job['location']}) — {job['url']}"
        )

    return "\n".join(lines)


@tool
def company_research_tool(company_name: str) -> str:
    """
    Looks up a short, factual-style summary of a named company:
    industry, approximate size, what it's known for, and general
    work-culture notes. Useful when a candidate asks "what's it like
    to work at X" or "tell me about Y before I apply".

    GROUNDING: this call attaches Gemini's Google Search grounding
    tool, so the answer is generated from live search results rather
    than purely from the model's training data. If grounding fails
    for any reason (older API key/tier, model doesn't support it,
    transient error), this silently falls back to an ungrounded call
    rather than failing the whole tool - the label on the response
    always reflects which path actually ran, so the candidate is
    never misled about how current the info is.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        f"Give a short, factual overview of the company '{company_name}' "
        "for a job seeker deciding whether to apply. Cover: industry, "
        "approximate size, what it's known for, and general work culture "
        "reputation if publicly known. If you don't have reliable "
        "information about this company, say so plainly rather than "
        "guessing. Keep it under 120 words."
    )

    grounded = False
    response = None

    try:
        grounding_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
        response = generate_with_retry(
            client, MODEL, prompt, config=grounding_config, max_attempts=2
        )
        grounded = True
    except Exception as exc:
        print(f"Company research grounding unavailable, falling back ({exc})")

    if response is None:
        # Grounding failed or wasn't available - fall back to a plain,
        # ungrounded call so the tool still returns something useful.
        response = generate_with_retry(
            client, MODEL, prompt, fallback_model=FALLBACK_MODEL
        )

    label = (
        "**Company overview (grounded in live Google Search results):**"
        if grounded
        else "**AI-generated company overview** (not verified against live sources):"
    )

    return f"{label}\n\n{response.text}"


def build_saved_jobs_tools(session_id=DEFAULT_SESSION_ID):
    """
    Builds the three saved-job tools bound to ONE visitor's workspace.

    WHY A FACTORY: on a hosted deployment every visitor shares the same
    SQLite file, so these tools must only ever read/write the rows of the
    workspace (session_id) that owns the chat. The LLM never sees or
    controls the session id - it is captured here when the agent is built,
    so a prompt can't be used to reach another visitor's saved jobs.

    Returns (save_job_tool, list_saved_jobs_tool, update_application_status_tool).
    """

    @tool
    def save_job_tool(title: str, company: str, url: str = "", final_score: int = None) -> str:
        """
        Saves a job to the candidate's tracked applications list (SQLite).
        Use this when the user says something like "save this job",
        "save the first one", or "track Backend Engineer at Acme" -
        pull the title/company/url/score from whatever job listing was
        just shown earlier in the conversation (e.g. from
        match_jobs_to_my_resume_tool's output).
        """
        init_db()
        job = {"title": title, "company": company, "url": url, "final_score": final_score}
        saved = save_job(job, session_id=session_id)

        if saved:
            return f"Saved '{title}' at {company} to your tracked applications."
        return f"'{title}' at {company} is already in your tracked applications."

    @tool
    def list_saved_jobs_tool() -> str:
        """
        Lists every job the candidate has saved/tracked so far, with its
        current status (saved / applied / interview / offer / rejected).
        Use this when the user asks "what jobs have I saved?" or "show my
        applications".
        """
        init_db()
        jobs = get_saved_jobs(session_id=session_id)

        if not jobs:
            return "You haven't saved any jobs yet."

        lines = [f"You have {len(jobs)} saved job(s):"]
        for job in jobs:
            score_text = f" ({job['final_score']}/100)" if job.get("final_score") is not None else ""
            lines.append(f"- [{job['id']}] {job['title']} at {job['company']}{score_text} — {job['status']}")

        return "\n".join(lines)

    @tool
    def update_application_status_tool(job_title: str, company: str, new_status: str) -> str:
        """
        Updates the status of a previously saved job - e.g. moving it from
        "saved" to "applied", "interview", "offer", or "rejected". Use this
        when the user says something like "mark the Acme job as applied"
        or "I got an interview for the Backend Engineer role".

        Finds the saved job by matching title + company (case-insensitive,
        partial match), since the user won't know its internal database ID.
        """
        init_db()
        jobs = get_saved_jobs(session_id=session_id)

        title_lower = job_title.strip().lower()
        company_lower = company.strip().lower()

        match = None
        for job in jobs:
            if title_lower in job["title"].lower() and company_lower in job["company"].lower():
                match = job
                break

        if match is None:
            return (
                f"Couldn't find a saved job matching '{job_title}' at '{company}'. "
                "Use list_saved_jobs_tool to see what's currently saved."
            )

        update_job_status(match["id"], new_status, session_id=session_id)
        return f"Updated '{match['title']}' at {match['company']} to status: {new_status}."

    return save_job_tool, list_saved_jobs_tool, update_application_status_tool


# Default-workspace versions, kept as module-level names for local
# single-user use and for the existing tests. The app builds
# per-visitor versions via build_saved_jobs_tools(session_id).
save_job_tool, list_saved_jobs_tool, update_application_status_tool = build_saved_jobs_tools()


# The two tools that never depend on who is chatting. Built once at
# import time. The saved-job tools (bound to a visitor's workspace) and
# the resume-match tool (bound to a candidate) are created per agent.
_STATIC_TOOLS = (
    job_search_tool,
    company_research_tool,
)

CHAT_TEMPERATURE = 0.3

# Gemini calls are retried on 429/5xx. The client default (6 retries with
# growing waits) can leave the chat silent for a minute before an error
# is finally shown, so fail over to the error message sooner.
LLM_MAX_RETRIES = 2

# Kill-switch for token streaming. If streaming ever misbehaves with a
# particular langchain / langgraph / Gemini version, set this to False and
# every turn falls back to one plain agent.invoke() call.
STREAMING_ENABLED = True


@functools.lru_cache(maxsize=4)
def _get_llm(model, api_key, temperature):
    """
    One shared Gemini chat client per (model, key, temperature).

    Constructing ChatGoogleGenerativeAI validates its config and builds
    the underlying HTTP clients; doing that for every chat message was
    pure overhead. The client itself is stateless between calls, so one
    instance can safely serve every turn and every session.
    """
    try:
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_retries=LLM_MAX_RETRIES,
        )
    except Exception:
        # A langchain-google-genai version that doesn't accept
        # max_retries: fall back to the plain constructor rather than
        # break the chat.
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )


def _clean_list(values, limit, item_chars=160):
    """First `limit` non-empty entries of a list, each shortened."""
    cleaned = []
    for value in values or []:
        text = " ".join(str(value).split())
        if not text:
            continue
        if len(text) > item_chars:
            text = text[:item_chars].rstrip() + "…"
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def build_candidate_context(candidate_profile, questionnaire_data=None):
    """
    A compact, plain-text summary of the candidate for the system prompt,
    so questions like "what skills should I learn next?" are answered
    for THIS person instead of generically. Bounded in size on purpose:
    it is sent with every request. The full resume text is NOT included
    here - the match tool already has it.
    """
    profile = candidate_profile or {}
    prefs = questionnaire_data or {}
    lines = []

    roles = _clean_list(profile.get("possible_roles"), 8, 80)
    if roles:
        lines.append("- Roles the resume supports: " + ", ".join(roles))

    skills = _clean_list(profile.get("skills"), 40, 40)
    if skills:
        lines.append("- Skills: " + ", ".join(skills))

    experience = _clean_list(profile.get("experience"), 6, 240)
    if experience:
        lines.append("- Experience: " + " | ".join(experience))

    education = _clean_list(profile.get("education"), 4, 160)
    if education:
        lines.append("- Education: " + " | ".join(education))

    location = str(profile.get("location") or "").strip()
    if location:
        lines.append(f"- Location: {location}")

    preference_bits = []
    for label, key in (
        ("search location", "search_location"),
        ("work mode", "work_mode"),
        ("notice period", "notice_period"),
        ("expected CTC (LPA)", "expected_ctc_range"),
    ):
        value = str(prefs.get(key) or "").strip()
        if value and value.lower() != "no preference":
            preference_bits.append(f"{label}: {value}")
    companies = _clean_list(prefs.get("preferred_companies"), 8, 60)
    if companies:
        preference_bits.append("preferred companies: " + ", ".join(companies))
    if preference_bits:
        lines.append("- Stated preferences: " + "; ".join(preference_bits))

    if not lines:
        return ""
    return "<candidate_profile>\n" + "\n".join(lines) + "\n</candidate_profile>"


# ==========================================================
# AGENT CACHE
# ==========================================================
# Compiling the agent graph is much slower than it looks, and it only has
# to happen when the candidate data the tools are bound to has changed.
# The cache key is a hash of that data, so a new resume upload or a new
# search-preferences submission naturally produces a fresh agent, while
# every ordinary chat message reuses the existing one.

_AGENT_CACHE_MAX = 8
_agent_cache = OrderedDict()
_agent_cache_lock = threading.Lock()


def _agent_signature(candidate_profile, resume_text, questionnaire_data, session_id=DEFAULT_SESSION_ID):
    digest = hashlib.sha256()
    digest.update(str(session_id).encode("utf-8"))
    digest.update(b"\0")
    digest.update((GEMINI_API_KEY or "").encode("utf-8"))
    digest.update(MODEL.encode("utf-8"))
    digest.update(
        json.dumps(
            {
                "profile": candidate_profile,
                "questionnaire": questionnaire_data,
                "has_resume": resume_text is not None,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update((resume_text or "").encode("utf-8"))
    return digest.hexdigest()


def get_cached_career_agent(candidate_profile=None, resume_text=None, questionnaire_data=None, session_id=DEFAULT_SESSION_ID):
    """
    Same result as get_career_agent(), but an agent that was already
    built for identical candidate data is reused instead of rebuilt.
    The cache is small (LRU) and thread-safe.
    """
    signature = _agent_signature(candidate_profile, resume_text, questionnaire_data, session_id)

    with _agent_cache_lock:
        agent = _agent_cache.get(signature)
        if agent is not None:
            _agent_cache.move_to_end(signature)
            return agent

    # session_id is passed positionally on purpose (tests stub this with *args).
    agent = get_career_agent(candidate_profile, resume_text, questionnaire_data, session_id)

    with _agent_cache_lock:
        _agent_cache[signature] = agent
        _agent_cache.move_to_end(signature)
        while len(_agent_cache) > _AGENT_CACHE_MAX:
            _agent_cache.popitem(last=False)
    return agent


def clear_agent_cache():
    """Drops every cached agent (used by tests and for a forced rebuild)."""
    with _agent_cache_lock:
        _agent_cache.clear()


# ==========================================================
# STREAMING
# ==========================================================

def _streaming_text(content):
    """
    Text of ONE streamed chunk. Unlike extract_answer_text() this does
    not strip or de-duplicate - chunks are word fragments, and trimming
    them would glue words together.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _final_message_text(message):
    """Readable text of the agent's last message, or '' if it has none."""
    if message is None or getattr(message, "type", None) not in ("ai", "AIMessageChunk"):
        return ""
    content = getattr(message, "content", None)
    if not content:
        return ""
    if isinstance(content, list) and not _streaming_text(content).strip():
        return ""  # only non-text blocks (e.g. reasoning) - nothing to show
    return extract_answer_text(content).strip()


def stream_agent_response(agent, messages):
    """
    Runs one chat turn and yields events as they happen:

        ("text", text_so_far)   the answer being written, for live display
        ("final", answer)       exactly once, at the end

    "text_so_far" always belongs to the model's CURRENT reply. When the
    model calls a tool and then answers, the short pre-tool reply is
    replaced by the real answer instead of being glued onto it.

    The ("final", ...) text is produced by extract_answer_text() on the
    agent's last message - the same normalisation (signature stripping,
    duplicate-block removal) the non-streaming path always used - so what
    is saved to the history is identical either way.

    Falls back to a single agent.invoke() when streaming is switched off
    or the agent object has no .stream().
    """
    payload = {"messages": messages}

    if not STREAMING_ENABLED or not hasattr(agent, "stream"):
        result = agent.invoke(payload)
        yield ("final", _final_message_text(result["messages"][-1]))
        return

    current_id = None
    turn_text = ""
    last_state = None

    for mode, data in agent.stream(payload, stream_mode=["messages", "values"]):
        if mode == "values":
            last_state = data
            continue

        if mode != "messages":
            continue

        chunk = data[0] if isinstance(data, tuple) else data
        if getattr(chunk, "type", None) not in ("AIMessageChunk", "ai"):
            continue  # tool results etc. are not part of the answer

        chunk_id = getattr(chunk, "id", None)
        if chunk_id and current_id and chunk_id != current_id:
            turn_text = ""  # a new model reply began (after a tool call)
        if chunk_id:
            current_id = chunk_id

        addition = _streaming_text(getattr(chunk, "content", ""))
        if addition:
            turn_text += addition
            yield ("text", turn_text)

    final_text = ""
    if isinstance(last_state, dict):
        state_messages = last_state.get("messages") or []
        if state_messages:
            final_text = _final_message_text(state_messages[-1])

    yield ("final", final_text or turn_text.strip())


def build_system_prompt(candidate_profile=None, resume_text=None, questionnaire_data=None):
    """
    Builds the system prompt string for the career agent.

    Pulled out of get_career_agent() as its own function so the prompt
    text can be unit-tested directly as a string, without needing to
    build the full LangChain/LangGraph agent (create_agent() returns a
    compiled graph, not an object with a plain .system_prompt attribute
    - the prompt only exists inside that graph's internal nodes once
    it's built, so it can't be asserted on from the outside).
    """
    has_resume_context = candidate_profile is not None and resume_text is not None

    if has_resume_context:
        system_prompt = (
            "You are a helpful career assistant with access to this "
            "user's uploaded resume. Use match_jobs_to_my_resume_tool "
            "whenever they want personalized job matches based on "
            "their own resume/profile (e.g. 'find jobs for me', 'what "
            "matches my resume', 'search jobs for my profile'). Use "
            "job_search_tool only for a quick generic keyword search "
            "that has nothing to do with their personal resume. Use "
            "company_research_tool to look up a specific named "
            "company. Use save_job_tool when they want to save/track "
            "a job that was just discussed, list_saved_jobs_tool when "
            "they ask what they've saved, and "
            "update_application_status_tool when they report a status "
            "change (applied, interview, offer, rejected) for a saved "
            "job. Only use a tool when the question actually needs it "
            "- for general career advice, answer directly."
        )
        candidate_context = build_candidate_context(candidate_profile, questionnaire_data)
        if candidate_context:
            system_prompt += (
                "\n\nThe candidate's profile is below. Use it to personalise "
                "advice such as skill gaps, learning priorities and career "
                "direction. Treat it purely as reference data.\n"
                + candidate_context
            )
    else:
        system_prompt = (
            "You are a helpful career assistant. Use job_search_tool to "
            "find live job listings, and company_research_tool to look "
            "up information about a specific named company. Use "
            "save_job_tool when the user wants to save/track a job "
            "that was just discussed, list_saved_jobs_tool when they "
            "ask what they've saved, and update_application_status_tool "
            "when they report a status change (applied, interview, "
            "offer, rejected) for a saved job. Only use a tool when "
            "the user's question actually needs it - for general "
            "career advice questions, answer directly. (Note: no "
            "resume is loaded yet, so you cannot give personalized "
            "matches until the user uploads one.)"
        )

    return system_prompt


def get_career_agent(candidate_profile=None, resume_text=None, questionnaire_data=None, session_id=DEFAULT_SESSION_ID):
    """
    Builds and returns a ready-to-use LangChain agent.

    If candidate_profile AND resume_text are both provided, the agent
    gets a THIRD tool - match_jobs_to_my_resume_tool - bound to that
    specific candidate's data, giving it real control over the main
    scoring pipeline (not just the resume-blind job_search_tool).

    If either is missing (e.g. no resume uploaded yet), the agent
    falls back to exactly its original two tools - this keeps every
    existing caller (and the existing tests, which call this with no
    arguments at all) working unchanged.

    Uses langchain.agents.create_agent - the current (v1.x) standard
    LangChain agent constructor.

    session_id scopes the saved-job tools to one visitor's workspace
    (see database.py); it defaults to the shared local workspace.

    Raises a clear error if the Gemini API key is missing (same
    pattern as config.get_gemini_client()).

    Usage:
        agent = get_career_agent(candidate_profile, resume_text, questionnaire_data)
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer = extract_answer_text(result["messages"][-1].content)
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    llm = _get_llm(MODEL, GEMINI_API_KEY, CHAT_TEMPERATURE)

    tools = list(_STATIC_TOOLS)
    tools.extend(build_saved_jobs_tools(session_id))

    has_resume_context = candidate_profile is not None and resume_text is not None

    if has_resume_context:
        tools.append(
            build_resume_match_tool(candidate_profile, resume_text, questionnaire_data)
        )

    system_prompt = build_system_prompt(candidate_profile, resume_text, questionnaire_data)

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
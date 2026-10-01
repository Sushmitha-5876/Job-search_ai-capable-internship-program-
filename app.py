"""Streamlit UI for the AI Job Search Agent.

The existing resume parsing, job fetching, scoring, PDF export, SQLite,
ATS scoring, and LangChain modules are kept intact. This file is the
presentation and orchestration layer.

DESIGN APPROACH: built on Streamlit's own native components (st.title,
st.caption, st.metric, st.container(border=True), st.tabs, st.button)
wherever one exists, rather than hand-rolled HTML/CSS - those render
consistently regardless of Streamlit version, and don't silently go
out of sync with the stylesheet the way a large custom class system
can. Custom CSS (assets/theme.css) is reserved for things with no
native equivalent: global color/type tokens, and match_card.py's score
ring, category bars, and skill tags. Decorative marketing-page chrome
that a functional TOOL doesn't need on every visit (a hero banner,
static "why use us" cards, canned sample content) has been removed
rather than reskinned.
"""

import base64
import html
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import get_gemini_client, MODEL, JOB_ROLE_OPTIONS
from src.resume_parser import extract_resume_text, extract_candidate_profile
from src.job_fetcher import fetch_all_jobs
from src.scorer import pre_rank_jobs, score_jobs_with_gemini
try:
    from src.scorer import LAST_GEMINI_ERROR
except ImportError:  # older scorer.py without error tracking
    LAST_GEMINI_ERROR = {"text": ""}
from src.final_scoring import calculate_final_score
from src.ui_helpers import build_results_table, get_expander_label, format_chat_answer, score_badge_html
from src.match_card import render_match_card
from src.data.locations import LOCATION_OPTIONS, CUSTOM_LOCATION_OPTION, NO_PREFERENCE_LOCATION_OPTION
from src.data.companies import COMPANY_OPTIONS, CUSTOM_COMPANY_OPTION
from src.database import (
    DEFAULT_SESSION_ID,
    new_session_id,
    init_db,
    get_saved_jobs,
    save_job,
    update_job_status,
    delete_saved_job,
    log_search,
    get_recent_searches,
    delete_recent_search,
)
from src.career_agent import get_cached_career_agent, stream_agent_response
from src.career_chat import (
    ATTACHMENT_CACHE_KEY,
    QUEUE_KEY,
    UPLOADER_NONCE_KEY,
    AttachmentError,
    enqueue_turn,
    get_or_parse_attachment,
    init_chat_state,
    reset_chat_state,
    run_turn,
    validate_attachment_upload,
)
from src.pdf_export import generate_pdf_report
from src.analytics import (
    format_time_ago,
    compute_status_counts,
    aggregate_missing_skills,
    aggregate_suitable_roles,
)
from src.skill_gap_roadmap import build_resume_improvement_suggestions
from src.ats_scoring import calculate_ats_score


# Options for the searchable "Job role / keywords" dropdown on the Find
# Jobs tab. Kept local to app.py (rather than in src/config.py) so this
# list can be edited without touching JOB_ROLE_OPTIONS, which is still
# used elsewhere.
ROLE_DROPDOWN_OPTIONS = [
    "Software Engineer",
    "Software Developer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "Web Developer",
    "Mobile App Developer",
    "Android Developer",
    "iOS Developer",
    "Java Developer",
    "Python Developer",
    ".NET Developer",
    "DevOps Engineer",
    "Cloud Engineer",
    "Cloud Architect",
    "Data Analyst",
    "Data Scientist",
    "Data Engineer",
    "Machine Learning Engineer",
    "AI Engineer",
    "AI/ML Engineer",
    "Generative AI Engineer",
    "NLP Engineer",
    "Computer Vision Engineer",
    "MLOps Engineer",
    "Cybersecurity Engineer",
    "Security Analyst",
    "Information Security Engineer",
    "Network Engineer",
    "Database Administrator",
    "Database Developer",
    "QA Engineer",
    "Test Automation Engineer",
    "Embedded Systems Engineer",
    "IoT Engineer",
    "Blockchain Developer",
    "Product Manager",
    "UI/UX Designer",
    "Business Analyst",
    "System Engineer",
    "Solutions Architect",
    "Site Reliability Engineer",
    "Technical Support Engineer",
    "IT Support Specialist",
    "Research Engineer",
    "Software Architect",
    "Technical Product Manager",
    "Project Manager",
    "Scrum Master",
    "Engineering Manager",
]


def get_workspace_id():
    """
    Returns this visitor's private workspace id (used to scope saved jobs
    and search history in the shared SQLite file - see src/database.py).

    The id lives in the page URL (?ws=...), so a browser refresh or a
    bookmarked link returns to the same workspace, while every other
    visitor gets their own random one. Anything missing or malformed is
    replaced with a fresh id.
    """
    if "workspace_id" not in st.session_state:
        candidate = st.query_params.get("ws")
        valid = (
            isinstance(candidate, str)
            and 8 <= len(candidate) <= 32
            and candidate.isalnum()
            and candidate != DEFAULT_SESSION_ID
        )
        workspace_id = candidate if valid else new_session_id()
        st.query_params["ws"] = workspace_id
        st.session_state["workspace_id"] = workspace_id
    return st.session_state["workspace_id"]


init_db()

st.set_page_config(
    page_title="AI Job Search Agent",
    page_icon="🔎",
    layout="wide",
)

WORKSPACE_ID = get_workspace_id()


# ------------------------------------------------------------------
# CSS — global tokens + the few components with no native equivalent
# ------------------------------------------------------------------
css_path = Path(__file__).parent / "assets" / "theme.css"
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
def init_state():
    defaults = {
        "scored_jobs": None,
        "questionnaire_data": None,
        "candidate_profile": None,
        "resume_text": None,
        "chat_history": [],
        "last_search_count": 0,
        "jobs_viewed": 0,
        "ats_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Career Agent chat plumbing: chat_history, the pending-turn queue,
    # attached documents and their parse cache (see src/career_chat.py).
    init_chat_state(st.session_state)


def get_asset_data_url(asset_name):
    asset_path = Path(__file__).parent / "assets" / asset_name
    # hero-robot.png is a raster image, everything else here is SVG text.
    # SVGs are read as text and base64-encoded once; a raster file must be
    # read as bytes and base64-encoded directly - wrapping a PNG inside an
    # SVG <image> and then base64-encoding that SVG on top (double
    # encoding) was inflating a 1.5MB image into a ~2.7MB data URL, which
    # is large enough that some browsers silently fail to load it, and is
    # exactly what was making the Career Agent illustration render blank.
    if asset_path.suffix.lower() == ".svg":
        content = asset_path.read_text(encoding="utf-8")
        encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        return "data:image/svg+xml;base64," + encoded
    mime = "image/png" if asset_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(asset_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64," + encoded


@st.cache_data(show_spinner=False)
def _downscaled_avatar_data_url(asset_name, source_mtime_ns, max_px):
    # source_mtime_ns is only part of the cache key: replacing the PNG in
    # assets/ produces a fresh copy instead of a stale one.
    import io
    from PIL import Image

    image = Image.open(Path(__file__).parent / "assets" / asset_name).convert("RGBA")
    image.thumbnail((max_px, max_px), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_chat_avatar_data_url(asset_name, max_px=216):
    """
    Data URL for a chat-bubble avatar.

    The avatar PNGs in assets/ are ~1250px and 1.3-1.4 MB, but the theme
    draws them at 54px. Embedding the original in every st.chat_message()
    re-sent roughly 2 MB per bubble on every rerun (a 10-message chat is
    ~20 MB), which is a big part of why the chat felt slow. This serves a
    216px copy (4x the display size, so it stays sharp on high-DPI
    screens; same aspect ratio, so object-fit: cover crops it exactly as
    before) that is built once and cached. The files in assets/ are not
    modified. Falls back to the original image if anything goes wrong.
    """
    try:
        source = Path(__file__).parent / "assets" / asset_name
        return _downscaled_avatar_data_url(asset_name, source.stat().st_mtime_ns, max_px)
    except Exception:
        return get_asset_data_url(asset_name)


def build_svg_hero_html(asset_name, alt_text, size=190):
    svg_data_url = get_asset_data_url(asset_name)
    return (
        f"<div class='hero-illustration-frame' style='width:{size}px; height:{size}px;'>"
        f"<img src='{svg_data_url}' alt='{alt_text}' class='hero-illustration-image' />"
        f"</div>"
    )


# ------------------------------------------------------------------
# ICON SYSTEM — small inline Lucide-style SVGs (24x24, stroke-based).
# Used anywhere a custom HTML badge is rendered (metric cards,
# capability cards) so the app uses one consistent, non-emoji icon
# language instead of unicode glyphs/emoji.
# ------------------------------------------------------------------
_LUCIDE_PATHS = {
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "sliders": (
        '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/>'
        '<line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/>'
        '<line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/>'
        '<line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/>'
        '<line x1="16" x2="16" y1="18" y2="22"/>'
    ),
    "compare": (
        '<path d="m16 3 4 4-4 4"/><path d="M20 7H4"/>'
        '<path d="m8 21-4-4 4-4"/><path d="M4 17h16"/>'
    ),
    "bar-chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "building": (
        '<path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"/>'
        '<path d="M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2"/>'
        '<path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2"/>'
        '<path d="M10 6h4"/><path d="M10 10h4"/><path d="M10 14h4"/><path d="M10 18h4"/>'
    ),
    "bookmark": '<path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>',
    "file-check": (
        '<path d="M4 22h14a2 2 0 0 0 2-2V7l-5-5H6a2 2 0 0 0-2 2v4"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="m3 15 2 2 4-4"/>'
    ),
    "zap": (
        '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 '
        '.78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>'
    ),
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "award": (
        '<path d="m15.477 12.89 1.515 8.526a.5.5 0 0 1-.81.47l-3.58-2.687a1 1 0 0 0-1.197 0l-3.586 2.686a.5.5 0 0 '
        '1-.81-.469l1.514-8.526"/><circle cx="12" cy="8" r="6"/>'
    ),
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "file-text": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>'
    ),
    "trending-up": (
        '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>'
    ),
    "clipboard-check": (
        '<rect width="8" height="4" x="8" y="2" rx="1"/>'
        '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
        '<path d="m9 14 2 2 4-4"/>'
    ),
    "paperclip": (
        '<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>'
    ),
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "lock": (
        '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>'
        '<path d="M7 11V7a5 5 0 0 1 10 0v4"/>'
    ),
    "award": (
        '<circle cx="12" cy="8" r="6"/>'
        '<path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>'
    ),
    # Added for the Find Jobs hero's job-type chips and feature strip.
    "briefcase": (
        '<rect width="20" height="14" x="2" y="7" rx="2" ry="2"/>'
        '<path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>'
    ),
    "graduation-cap": (
        '<path d="M21.42 10.922a1 1 0 0 0-.019-1.838L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.832l8.57 3.908a2 2 0 0 0 1.66 0z"/>'
        '<path d="M22 10v6"/><path d="M6 12.5V16a6 3 0 0 0 12 0v-3.5"/>'
    ),
    "home": (
        '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '<polyline points="9 22 9 12 15 12 15 22"/>'
    ),
    "target": (
        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'
    ),
    "message-square": (
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
    ),
    "more-horizontal": (
        '<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>'
    ),
}


def lucide_icon(name, size=18):
    path = _LUCIDE_PATHS.get(name, _LUCIDE_PATHS["search"])
    return (
        f"<svg class='lucide-icon' width='{size}' height='{size}' viewBox='0 0 24 24' "
        f"fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
        f"stroke-linejoin='round'>{path}</svg>"
    )


init_state()


# ------------------------------------------------------------------
# TOP BRAND BAR + TABS
# ------------------------------------------------------------------
def render_topbar():
    # The search bar, notification bell, and account dropdown that
    # used to sit here were pure decoration - not wired to anything,
    # clicking them did nothing. Removed rather than kept as fake UI;
    # the real search lives in the Job Search tab, and there's no
    # login/notification system in this project.
    logo_path = Path(__file__).parent / "assets" / "logo-icon.svg"
    svg_content = logo_path.read_text(encoding="utf-8")
    logo_data_url = "data:image/svg+xml;base64," + base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")

    st.markdown(
        f"""
        <div class="ai-topbar">
            <div class="ai-brand">
                <img class="ai-logo" src="{logo_data_url}" alt="AI Job Search Agent logo" />
                <div>
                    <div class="ai-brand-title">AI Job Search <span class="ai-title-accent">Agent</span></div>
                    <div class="ai-brand-sub">Find · Understand · Decide</div>
                </div>
            </div>
            <div class="ai-topbar-tagline">A smarter way<br>to your next opportunity.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_navbar():
    """
    Two visible tabs only: the main job-search dashboard and the
    career agent. Analytics remains available in code for the
    existing saved-job/session insights, but it is intentionally not
    exposed in the main UI.
    """
    return st.tabs([":material/search: Find Jobs", ":material/smart_toy: Career Agent"])


# ------------------------------------------------------------------
# JOB SEARCH & SCORING
# ------------------------------------------------------------------
def run_search(resume_file, role_keywords, search_location, work_mode, notice_period, ctc_range, expected_ctc_exact, preferred_companies, max_jobs):
    if resume_file is None:
        st.error("Please upload a resume PDF before searching.")
        return

    file_bytes = resume_file.getvalue()
    if len(file_bytes) > 10 * 1024 * 1024:
        st.error("Resume exceeds the 10 MB limit. Please upload a smaller PDF.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        resume_path = tmp.name

    st.session_state.questionnaire_data = {
        "search_location": search_location,
        "work_mode": work_mode,
        "notice_period": notice_period,
        "expected_ctc_range": ctc_range,
        "expected_ctc_exact": expected_ctc_exact,
        "preferred_companies": preferred_companies,
        "max_jobs": max_jobs,
    }

    try:
        with st.status("Running your job search pipeline...", expanded=True) as status:
            st.write("Reading resume...")
            resume_text = extract_resume_text(resume_path)

            st.write("Building candidate profile...")
            client = get_gemini_client()
            candidate_profile = extract_candidate_profile(resume_text, client, MODEL)
            if candidate_profile.get("_source") == "local_fallback":
                st.warning(
                    "Gemini is busy right now, so your profile was built with "
                    "the built-in resume parser. Results still work - try "
                    "again later for AI-quality profile extraction."
                )

            # Manually typed role/keywords take priority over whatever
            # Gemini inferred, since the user explicitly asked for them -
            # this field used to be collected and silently discarded.
            role_keywords = (role_keywords or "").strip()
            if role_keywords:
                manual_roles = [r.strip() for r in role_keywords.split(",") if r.strip()]
                existing_roles = candidate_profile.get("possible_roles", []) or []
                manual_lower = {r.lower() for r in manual_roles}
                candidate_profile["possible_roles"] = manual_roles + [
                    r for r in existing_roles if r.lower() not in manual_lower
                ]
            st.session_state.candidate_profile = candidate_profile
            st.session_state.resume_text = resume_text

            st.write("Searching RemoteOK, Adzuna, and JobsPipe...")
            jobs = fetch_all_jobs(candidate_profile, search_location=search_location)
            if not jobs:
                st.session_state.scored_jobs = []
                st.session_state.last_search_count = 0
                status.update(label="No matching jobs found", state="error")
                st.warning("No relevant jobs were found. Try a broader location or role.")
                return

            st.write(f"Pre-ranking {len(jobs)} jobs...")
            top_jobs = pre_rank_jobs(jobs, candidate_profile, max_jobs=max_jobs)

            st.write(f"Scoring {len(top_jobs)} jobs with Gemini...")
            LAST_GEMINI_ERROR["text"] = ""
            scored_jobs = score_jobs_with_gemini(top_jobs, resume_text, client, MODEL)
            if top_jobs and not scored_jobs:
                st.error(
                    "Gemini could not score any of the jobs, so nothing can be "
                    "ranked. Real error: "
                    + (LAST_GEMINI_ERROR["text"] or "unknown")[:400]
                )

            for job in scored_jobs:
                final_result = calculate_final_score(
                    job,
                    candidate_profile,
                    st.session_state.questionnaire_data,
                    job,
                )
                job.update(final_result)
                job["match_score"] = job["final_score"]

            st.session_state.scored_jobs = scored_jobs
            st.session_state.last_search_count = len(scored_jobs)
            st.session_state.jobs_viewed += len(scored_jobs)

            search_summary_parts = [
                part for part in [
                    candidate_profile.get("possible_roles", [None])[0] if candidate_profile.get("possible_roles") else None,
                    search_location or None,
                    work_mode if work_mode != "No preference" else None,
                ]
                if part
            ]
            log_search(" · ".join(search_summary_parts) or "Job search", session_id=WORKSPACE_ID)

            status.update(label=f"Search complete — {len(scored_jobs)} jobs ranked", state="complete")
    except Exception as exc:
        st.error(f"Something went wrong while searching: {exc}")
    finally:
        try:
            os.remove(resume_path)
        except OSError:
            pass


def run_ats_check(resume_file, job_description):
    """
    Runs the deterministic ATS readiness check (src/ats_scoring.py).
    Deliberately separate from run_search(): no Gemini call, no
    questionnaire preferences needed - just the resume text and,
    optionally, a pasted job description.
    """
    if resume_file is None:
        st.error("Please upload a resume PDF first.")
        return

    file_bytes = resume_file.getvalue()
    if len(file_bytes) > 10 * 1024 * 1024:
        st.error("Resume exceeds the 10 MB limit. Please upload a smaller PDF.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        resume_path = tmp.name

    try:
        resume_text, used_ocr = extract_resume_text(resume_path, return_used_ocr=True)
        st.session_state.ats_result = calculate_ats_score(
            resume_text, used_ocr=used_ocr, job_description=job_description
        )
    except Exception as exc:
        st.error(f"Could not run the ATS check: {exc}")
    finally:
        try:
            os.remove(resume_path)
        except OSError:
            pass


def render_ats_results():
    result = st.session_state.ats_result

    if not result:
        st.info(
            "Your ATS score, strengths, missing keywords, and recommendations will "
            "show up here once you run a check above."
        )
        return

    checks = result.get("checks", [])
    passed = [item["label"] for item in checks if item.get("passed")]
    failed = [item["label"] for item in checks if not item.get("passed")]
    score = result.get("overall_score", 0)
    status = "Ready" if score >= 75 else "Needs review"

    with st.expander(
        f"Check ATS readiness · {score}/100 · {status}",
        expanded=True,
    ):
        c1, c2 = st.columns([1.1, 2.2])
        with c1:
            st.metric("ATS score", f"{score}/100")
            st.caption(result["verdict"])
        with c2:
            progress = max(0, min(100, score)) / 100
            st.progress(progress, text=f"Resume match quality: {score}/100")
            for line in result.get("summary", [])[:3]:
                st.markdown(f"- {line}")

        insight_1, insight_2 = st.columns(2)
        with insight_1:
            st.markdown("**Resume strengths**")
            if passed:
                for item in passed[:5]:
                    st.markdown(f"• {item}")
            else:
                st.caption("No explicit strengths detected in the current ATS check.")
        with insight_2:
            st.markdown("**Improvement suggestions**")
            if failed:
                for item in failed[:5]:
                    st.markdown(f"• {item}")
            else:
                st.caption("No major ATS issues were detected.")

        if result.get("keyword_score") is not None:
            st.markdown(f"**Keyword match with job description: {result['keyword_score']}%**")
            k1, k2 = st.columns(2)
            with k1:
                st.markdown(":material/check_circle: **Found**")
                if result.get("matched_keywords"):
                    st.caption(", ".join(result["matched_keywords"]))
                else:
                    st.caption("No keywords matched from the job description.")
            with k2:
                st.markdown(":material/cancel: **Missing**")
                if result.get("missing_keywords"):
                    st.caption(", ".join(result["missing_keywords"]))
                else:
                    st.caption("No missing keywords detected.")

        with st.expander("Detailed ATS checks"):
            for check in checks:
                icon = ":material/check_circle:" if check["passed"] else ":material/cancel:"
                st.markdown(f"{icon} **{check['label']}** — {check['detail']}")


def render_jobs_empty_state():
    st.info(
        "No jobs matched your current search yet. Try broadening the role keywords, location, or work mode, or use a more general set of preferred companies."
    )


def render_profile_empty_state():
    with st.container(border=True, key="candidate_profile_card"):
        st.markdown(":material/badge: **Your candidate profile**")
        st.caption(
            "Once your resume is uploaded, this card will show what we extracted — role, skills, experience, location, and education."
        )


def _format_profile_work_history(experience):
    if not experience:
        return ""

    if isinstance(experience, str):
        return experience

    if not isinstance(experience, list):
        return str(experience)

    # Each role is its own markdown BLOCK separated by a blank line.
    # Joining every line with a single "\n" made markdown treat the next
    # role's heading as a continuation of the previous role's bullet, so
    # three separate jobs rendered as one run-on paragraph.
    blocks = []
    for entry in experience:
        if isinstance(entry, dict):
            title = entry.get("title") or "Role"
            company = entry.get("company") or "Company not specified"
            dates = entry.get("dates") or entry.get("date") or entry.get("period") or ""

            header = f"**{title}** — {company}"
            if dates:
                header += f"  \n*{dates}*"
            blocks.append(header)

            description = entry.get("description") or entry.get("details") or entry.get("highlights") or []
            if isinstance(description, str):
                description = [description]
            bullets = [
                f"- {str(bullet).strip()}"
                for bullet in description[:3]
                if str(bullet).strip()
            ]
            if bullets:
                blocks.append("\n".join(bullets))

        elif isinstance(entry, str):
            blocks.append(entry)

    return "\n\n".join(blocks)


def _format_profile_education(education):
    if not education:
        return ""

    if isinstance(education, str):
        return education

    if not isinstance(education, list):
        return str(education)

    lines = []
    for entry in education:
        if isinstance(entry, dict):
            degree = entry.get("degree") or "Degree"
            institution = entry.get("institution") or entry.get("school") or "Institution not specified"
            year = entry.get("year") or entry.get("dates") or entry.get("period") or ""
            if year:
                lines.append(f"**{degree}** — {institution} ({year})")
            else:
                lines.append(f"**{degree}** — {institution}")
        elif isinstance(entry, str):
            lines.append(entry)

    # Blank line between degrees, same reason as work history above.
    return "\n\n".join(lines)


def _plain_text(value):
    """Strips the markdown emphasis markers so a value built for
    st.markdown can be reused inside a plain-text summary pill, where
    "**M.Tech**" would otherwise render with the asterisks visible.
    """
    text = str(value or "")
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"_{2,}", "", text)
    return " ".join(text.split())


def render_candidate_profile_summary():
    """
    Shows what Gemini actually extracted from the resume - roles,
    skills, experience, location, education. Seeing what the AI
    understood is genuinely useful for a demo and for spotting a bad
    extraction.
    """
    profile = st.session_state.candidate_profile
    if not profile:
        render_profile_empty_state()
        return

    roles = profile.get("possible_roles", []) or []
    skills = profile.get("skills", []) or []
    experience = profile.get("experience") or []
    experience_years = profile.get("experience_years") or "Not detected"
    location = profile.get("location") or "Not detected"
    education = profile.get("education", []) or []

    education_text = _format_profile_education(education)
    education_short = (
        _plain_text(education_text.splitlines()[0]) if education_text else "Not detected"
    )

    with st.container(border=True, key="candidate_profile_card"):
        st.markdown(":material/badge: **Your candidate profile**")
        st.caption("What Gemini extracted from your resume.")

        # Compact horizontal strip: each pill carries its own value inline
        # (label + short value) instead of stacking full paragraphs below,
        # so the card stays a short horizontal row rather than a tall
        # vertical block.
        fields = [
            (lucide_icon("search", size=13), "Role", ", ".join(roles[:2]) if roles else "Not detected"),
            (lucide_icon("sliders", size=13), "Skills", ", ".join(skills[:4]) if skills else "Not detected"),
            (lucide_icon("clock", size=13), "Experience", str(experience_years)),
            (lucide_icon("building", size=13), "Location", location),
            (lucide_icon("award", size=13), "Education", education_short),
        ]
        fields_html = "".join(
            f"<div class='profile-pill'>{icon}<span class='profile-pill-label'>{label}:</span>"
            f"<span class='profile-pill-value'>{html.escape(str(value))}</span></div>"
            for icon, label, value in fields
        )
        st.markdown(f"<div class='profile-pill-row'>{fields_html}</div>", unsafe_allow_html=True)

        # Inline spacer rather than a stylesheet rule: the pill strip and
        # the expander are two adjacent Streamlit blocks, and without a
        # real element between them the expander header sat flush against
        # the pills above it.
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        with st.expander("View full extracted profile"):
            st.markdown(f"**Roles:**  \n{', '.join(roles) if roles else 'Not detected'}")
            st.markdown(f"**Skills:**  \n{', '.join(skills) if skills else 'Not detected'}")
            work_history = _format_profile_work_history(experience)
            if work_history:
                st.markdown("**Work history**")
                st.markdown(work_history)
            if education_text:
                st.markdown("**Education**")
                st.markdown(education_text)
            # NOTE: the "Raw extracted profile (JSON)" dump used to live
            # here. It was a developer-debug view, it was far taller than
            # everything above it, and its collapsible tree ran into the
            # readable summary it sat under. The formatted sections above
            # already show every field it contained, so it is gone rather
            # than merely restyled.


def render_recommended_for_you():
    """
    A lightweight highlight strip pulled from the candidate's own
    latest scored results — not canned sample content. If no search
    has run yet, this shows the same honest empty state the rest of
    the app uses rather than fabricating jobs.
    """
    scored_jobs = st.session_state.scored_jobs or []

    st.session_state.setdefault("recommended_view_all", False)

    with st.container(border=True, key="recommended_card"):
        header_col, toggle_col = st.columns([3, 1.2], gap="small", vertical_alignment="center")
        with header_col:
            st.markdown("**Recommended for you**")
            st.caption("Based on your profile and recent activity.")
        with toggle_col:
            if len(scored_jobs) > 5:
                toggle_label = "Show less" if st.session_state.recommended_view_all else "See all →"
                if st.button(toggle_label, key="recommended_toggle"):
                    st.session_state.recommended_view_all = not st.session_state.recommended_view_all
                    st.rerun()

        if not scored_jobs:
            st.info("Run a job search above to get a personalized shortlist here, based on your resume and preferences.")
            return

        top_jobs = sorted(scored_jobs, key=lambda j: j.get("final_score", 0), reverse=True)
        # Default to 5 cards in one balanced row (matches the reference);
        # "See all" only appears once there are more than that to show.
        top_jobs = top_jobs if st.session_state.recommended_view_all else top_jobs[:5]

        saved_titles = {(j.get("title"), j.get("company")) for j in get_saved_jobs(session_id=WORKSPACE_ID)}

        # Chunk into rows of at most 5 cards so a long "view all" list
        # doesn't squeeze into one over-wide row.
        for row_start in range(0, len(top_jobs), 5):
            row_jobs = top_jobs[row_start:row_start + 5]
            rec_cols = st.columns(5)
            for col, job in zip(rec_cols, row_jobs):
                with col:
                    with st.container(border=True, key=f"rec_card_{job.get('title','')[:16]}_{job.get('company','')[:16]}_{row_start}"):
                        score = job.get("final_score", job.get("match_score", 0))
                        tags = (job.get("matched_skills") or [])[:3]
                        tags_html = "".join(
                            f"<span class='rec-card-tag'>{html.escape(str(tag))}</span>"
                            for tag in tags
                        )
                        # Score pill, title, company and tags are ONE markdown
                        # block. As four separate Streamlit elements the pill
                        # sat in its own zero-height wrapper and visually
                        # collided with the job title underneath it.
                        st.markdown(
                            "<div class='rec-card-body'>"
                            f"<div class='rec-card-score'>{score}% match</div>"
                            f"<div class='rec-card-title'>{html.escape(str(job.get('title', 'N/A')))}</div>"
                            f"<div class='rec-card-company'>{html.escape(str(job.get('company', 'N/A')))}</div>"
                            f"<div class='rec-card-tags{'' if tags else ' rec-card-tags-empty'}'>{tags_html}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                        # Save/Saved button rendered INSIDE the same bordered
                        # container as the rest of the card content, so it
                        # never appears floating below the card box.
                        already_saved = (job.get("title"), job.get("company")) in saved_titles
                        if already_saved:
                            st.button(
                                "Saved", icon=":material/bookmark:",
                                key=f"rec_saved_{job.get('title','')[:16]}_{job.get('company','')[:16]}_{row_start}",
                                use_container_width=True, disabled=True,
                            )
                        else:
                            if st.button(
                                "Save", icon=":material/bookmark_border:",
                                key=f"rec_save_{job.get('title','')[:16]}_{job.get('company','')[:16]}_{row_start}",
                                use_container_width=True,
                            ):
                                save_job(job, session_id=WORKSPACE_ID)
                                st.rerun()


def render_search_results():
    scored_jobs = st.session_state.scored_jobs

    if scored_jobs is None:
        return

    if len(scored_jobs) == 0:
        st.divider()
        st.subheader("Top job matches (0)")
        st.caption("No matches were found for the current search.")
        render_jobs_empty_state()
        return

    sorted_jobs = sorted(scored_jobs, key=lambda j: j.get("final_score", 0), reverse=True)

    st.divider()
    st.subheader(f"Top job matches ({len(sorted_jobs)})")
    st.caption("Ranked using your resume and preferences.")

    results_df = build_results_table(scored_jobs)
    st.dataframe(results_df, width="stretch", hide_index=True)

    pdf_bytes = generate_pdf_report(
        st.session_state.candidate_profile,
        st.session_state.questionnaire_data,
        scored_jobs,
    )
    st.download_button(
        "Export PDF Report",
        icon=":material/download:",
        data=pdf_bytes,
        file_name="job_search_match_report.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )

    suggestions = build_resume_improvement_suggestions(scored_jobs)
    if suggestions:
        st.markdown(":material/trending_up: **Resume improvement suggestions**")
        st.caption(
            "Skills that came up as missing on 2 or more of your shortlisted "
            "jobs — worth prioritizing over a skill only one job mentioned."
        )
        for item in suggestions:
            st.markdown(
                f"- **{item['skill']}** — missing from {item['count']} of your matched jobs · "
                f"{item['resource']} ({item['estimated_time']})"
            )

    for index, job in enumerate(sorted_jobs):
        with st.container(key=f"job_result_{index}"):
            with st.expander(get_expander_label(job), expanded=index < 2):
                render_match_card(job, session_id=WORKSPACE_ID)


def get_last_search_caption():
    recent_searches = get_recent_searches(limit=1, session_id=WORKSPACE_ID)
    if not recent_searches:
        return "No searches yet"

    searched_at = recent_searches[0].get("searched_at")
    try:
        if searched_at.endswith("Z"):
            searched_at = searched_at[:-1] + "+00:00"
        dt = datetime.fromisoformat(searched_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hours_since = (now - dt).total_seconds() / 3600
        return "Today" if hours_since < 24 else "This week"
    except Exception:
        return "Today"


def render_metric_cards(cards, columns=4):
    # columns=2 renders the compact 2 x 2 grid used inside the Quick
    # snapshot card, which now shares a row with the upload card and so
    # has roughly half the width the old full-width 4-across strip had.
    grid_class = "metric-card-grid" if columns == 4 else f"metric-card-grid metric-card-grid-{columns}"
    cards_html = "".join(
        f"<div class='metric-card'>"
        f"<div class='metric-card-icon {card['badge_class']}'>{card['icon']}</div>"
        f"<div class='metric-card-content'>"
        f"<div class='metric-card-label'>{card['label']}</div>"
        f"<div class='metric-card-value'>{card['value']}</div>"
        f"<div class='metric-card-caption {card.get('caption_class', '')}'>{card.get('caption', '')}</div>"
        f"</div></div>"
        for card in cards
    )
    st.markdown(f"<div class='{grid_class}'>{cards_html}</div>", unsafe_allow_html=True)


def job_search_page():
    # hero-jobs-search.svg is a wide (2.8:1) illustration that already
    # draws the document, magnifier, particles and the Full-time /
    # Internship / Remote / Part-time badges itself - so unlike the old
    # square robot icon, no extra HTML sparkles/chips/caption are layered
    # on top of it here (that would just duplicate what's already in the
    # artwork). Kept separate from hero-robot.png, used on the Career
    # Agent tab, which is untouched.
    illustration_html = build_svg_hero_html(
        "hero-jobs-search.svg", "Job search illustration", size=260
    )
    st.markdown(
        f"""
        <div class="hero-panel hero-panel-jobs">
            <div class="hero-panel-jobs-top">
                <div class="hero-copy">
                    <div class="hero-eyebrow">Explore Opportunities</div>
                    <h1>Find<span>Jobs</span></h1>
                    <p>Discover job opportunities that match your skills, interests and career goals.</p>
                    <a href="javascript:void(0)" class="hero-cta-jobs"
                       onclick="document.querySelector('.st-key-preferences_card')?.scrollIntoView({{behavior:'smooth', block:'start'}});">
                        {lucide_icon('search', size=15)}<span>Search Jobs</span>{lucide_icon('arrow-right', size=15)}
                    </a>
                </div>
                <div class="hero-visual hero-visual-jobs">
                    <div class="hero-illustration-stage hero-illustration-stage-jobs">
                        {illustration_html}
                    </div>
                </div>
            </div>
            <div class="hero-feature-strip">
                <div class="hero-feature-strip-item">
                    <div class="hero-feature-strip-icon">{lucide_icon('zap', size=16)}</div>
                    <div>
                        <div class="hero-feature-strip-title">Latest jobs</div>
                        <div class="hero-feature-strip-sub">Stay updated with new opportunities</div>
                    </div>
                </div>
                <div class="hero-feature-strip-item">
                    <div class="hero-feature-strip-icon">{lucide_icon('target', size=16)}</div>
                    <div>
                        <div class="hero-feature-strip-title">Relevant matches</div>
                        <div class="hero-feature-strip-sub">Find roles that fit you</div>
                    </div>
                </div>
                <div class="hero-feature-strip-item">
                    <div class="hero-feature-strip-icon">{lucide_icon('bookmark', size=16)}</div>
                    <div>
                        <div class="hero-feature-strip-title">Save &amp; track</div>
                        <div class="hero-feature-strip-sub">Keep your progress in one place</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    saved_jobs = get_saved_jobs(session_id=WORKSPACE_ID)
    ats_result = st.session_state.ats_result
    total_searches_ever = len(get_recent_searches(limit=1000, session_id=WORKSPACE_ID))

    sourced_jobs = st.session_state.scored_jobs or []
    ats_label = f"{ats_result['overall_score']}/100" if ats_result else "—"
    last_search_value = st.session_state.last_search_count if st.session_state.last_search_count else "—"

    active_filters = 0
    for value in (
        st.session_state.get("selected_location"),
        st.session_state.get("work_mode"),
        st.session_state.get("notice_period"),
        st.session_state.get("ctc_range"),
    ):
        if value and value not in ("No preference", NO_PREFERENCE_LOCATION_OPTION, "Immediately available"):
            active_filters += 1
    if st.session_state.get("selected_companies"):
        active_filters += 1

    ats_caption_class = ""
    ats_caption = ""
    if ats_result:
        if ats_result.get("overall_score", 0) >= 90:
            ats_caption = "Great!"
            ats_caption_class = "good"
        elif ats_result.get("overall_score", 0) >= 70:
            ats_caption = "Good"
            ats_caption_class = "warn"
        elif ats_result.get("overall_score", 0) is not None:
            ats_caption = "Needs work"
            ats_caption_class = "risk"

    # ==================================================================
    # 4. QUICK SNAPSHOT  |  UPLOAD YOUR RESUME  (side by side)
    # ------------------------------------------------------------------
    # These two are the only side-by-side row on the page. Everything
    # after them is full width, which is what keeps the old "short card
    # next to a tall card" dead space from coming back.
    # ==================================================================
    with st.container(key="snapshot_upload_row"):
        snapshot_col, upload_col = st.columns([1.1, 1], gap="large")

    with snapshot_col:
        with st.container(border=True, key="snapshot_card"):
            header_col, updated_col = st.columns([3, 2], gap="small", vertical_alignment="center")
            with header_col:
                st.markdown("**Quick snapshot**")
            with updated_col:
                searches_label = f"{total_searches_ever} search{'es' if total_searches_ever != 1 else ''} run"
                st.markdown(
                    f"<div class='quick-snapshot-meta'>{lucide_icon('bar-chart', size=13)}<span>{searches_label}</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption(
                "Every job in the shortlist shows a score, category breakdown, "
                "skill gaps and clear risk signals."
            )
            # 2 x 2 grid, not the old 4-across strip.
            render_metric_cards(
                [
                    {"icon": lucide_icon("search"), "label": "Last search", "value": last_search_value, "badge_class": "blue", "caption": (f"{last_search_value} jobs found" if st.session_state.last_search_count else "No searches yet"), "caption_class": "muted"},
                    {"icon": lucide_icon("bookmark"), "label": "Saved jobs", "value": len(saved_jobs), "badge_class": "green", "caption": "Total", "caption_class": "muted"},
                    {"icon": lucide_icon("file-check"), "label": "ATS score", "value": ats_label, "badge_class": "purple", "caption": ats_caption or "Not checked yet", "caption_class": ats_caption_class or "muted"},
                    {"icon": lucide_icon("filter"), "label": "Active filters", "value": active_filters, "badge_class": "amber", "caption": "Filters applied" if active_filters else "None set", "caption_class": "muted"},
                ],
                columns=2,
            )

    with upload_col:
        with st.container(border=True, key="resume_upload_card"):
            st.markdown("**1. Upload your resume**")
            st.caption("PDF, used to extract your skills, experience, and role preferences.")

            resume_file = st.file_uploader(
                "Resume PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="resume_upload",
            )
            ready_html = ""
            if resume_file:
                size_mb = len(resume_file.getvalue()) / (1024 * 1024)
                if size_mb > 10:
                    st.error("File is larger than 10 MB.")
                else:
                    size_label = f"{size_mb * 1024:.0f} KB" if size_mb < 1 else f"{size_mb:.2f} MB"
                    ready_html = (
                        "<div class='resume-ready-row'>"
                        f"<div class='resume-ready-icon'>{lucide_icon('file-text', size=16)}</div>"
                        "<div class='resume-ready-info'>"
                        f"<div class='resume-ready-name'>{html.escape(resume_file.name)}</div>"
                        f"<div class='resume-ready-size'>{size_label}</div>"
                        "</div>"
                        f"<div class='resume-ready-badge'>{lucide_icon('check', size=13)}Ready</div>"
                        "</div>"
                    )

            # State marker lets the scoped CSS keep the outer panel height
            # stable while compacting the uploaded-file presentation.
            upload_state = "uploaded" if resume_file and ready_html else "empty"
            st.markdown(
                f"<div class='resume-upload-state {upload_state}' aria-hidden='true'></div>",
                unsafe_allow_html=True,
            )

            # Filename, Ready badge and privacy note are one markdown block
            # rendered INSIDE the card, so nothing can escape or overlap it.
            st.markdown(
                "<div class='resume-upload-footer'>"
                f"{ready_html}"
                "<div class='resume-privacy-note'>"
                f"{lucide_icon('lock', size=13)}"
                "<span>Used only for matching — not stored permanently by this app.</span>"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ==================================================================
    # 5. CHECK ATS READINESS — full width
    # ==================================================================
    with st.container(border=True, key="ats_card"):
        st.markdown(
            "**Check ATS readiness** "
            "<span class='section-optional-tag'>Optional</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Would this resume survive an Applicant Tracking System filter, before a "
            "human even sees it? Paste a target job description for a keyword match "
            "too, or leave it blank for a formatting-only check."
        )
        ats_job_description = st.text_area(
            "Target job description (optional)",
            placeholder="Paste a job description here for a keyword match score...",
            label_visibility="collapsed",
            height=96,
            max_chars=2000,
            key="ats_job_description",
        )
        # NOTE: no manual "x/2000" counter here - st.text_area already
        # renders its own native character counter whenever max_chars is set.
        _, ats_button_col, _ = st.columns([1, 2, 1], gap="small")
        with ats_button_col:
            if st.button(
                "Check ATS readiness",
                icon=":material/fact_check:",
                use_container_width=True,
                key="check_ats",
                disabled=resume_file is None,
            ):
                run_ats_check(resume_file, ats_job_description)
        render_ats_results()

    # ==================================================================
    # 6. JOB SEARCH PREFERENCES — full width
    # ==================================================================
    with st.container(border=True, key="preferences_card"):
        st.markdown("**2. Job search preferences**")
        st.caption("Tell the agent what you're looking for.")

        role_keywords = st.selectbox(
            "Job role / keywords",
            ROLE_DROPDOWN_OPTIONS,
            index=None,
            placeholder="Search or select a job role...",
            key="role_free_text",
        )

        r1, r2, r3 = st.columns(3, gap="medium")
        with r1:
            selected_location = st.selectbox(
                "Preferred Location", LOCATION_OPTIONS, key="selected_location",
                help="Leave as \"No preference\" to search near the location detected in your resume.",
            )
        with r2:
            work_mode = st.selectbox("Work Mode", ["No preference", "Remote", "Hybrid", "On-site"], key="work_mode")
        with r3:
            notice_period = st.selectbox("Notice Period", ["Immediately available", "15 days", "30 days", "60 days", "90 days"], key="notice_period")

        r4, r5, r6 = st.columns(3, gap="medium")
        with r4:
            ctc_range = st.selectbox("Expected CTC (LPA)", ["No preference", "0-5", "5-10", "10-15", "15-20", "20-30", "30-50", "50+", "Custom (enter specific LPA)"], key="ctc_range")
        with r5:
            selected_companies = st.multiselect("Preferred Companies", COMPANY_OPTIONS, placeholder="e.g. Microsoft, Amazon, Google", key="selected_companies")
        with r6:
            max_jobs = st.selectbox("Jobs to Score", [5, 10, 15, 20], index=1, format_func=lambda n: f"{n} jobs", key="max_jobs")

        expected_ctc_exact = 0.0
        if ctc_range == "Custom (enter specific LPA)":
            expected_ctc_exact = st.number_input("Expected CTC (LPA)", min_value=0.0, max_value=200.0, step=0.5, key="expected_ctc_exact")

        custom_location = ""
        if selected_location == CUSTOM_LOCATION_OPTION:
            custom_location = st.text_input("Custom location", placeholder="e.g. Mysuru", key="custom_location")
        search_location = custom_location.strip() if selected_location == CUSTOM_LOCATION_OPTION else ("" if selected_location == NO_PREFERENCE_LOCATION_OPTION else selected_location.strip())

        custom_company = ""
        if CUSTOM_COMPANY_OPTION in selected_companies:
            custom_company = st.text_input("Custom company", placeholder="e.g. Your preferred company", key="custom_company").strip()
        preferred_companies = [c for c in selected_companies if c != CUSTOM_COMPANY_OPTION]
        if custom_company:
            preferred_companies.append(custom_company)

        submitted = st.button(
            "Find matching jobs",
            icon=":material/search:",
            type="primary",
            use_container_width=True,
            key="find_jobs",
            disabled=resume_file is None,
        )
        if submitted:
            run_search(resume_file, role_keywords, search_location, work_mode, notice_period, ctc_range, expected_ctc_exact, preferred_companies, max_jobs)

    # ==================================================================
    # 7. YOUR CANDIDATE PROFILE — full width, last
    # ==================================================================
    render_candidate_profile_summary()
    render_recommended_for_you()
    render_search_results()


# ------------------------------------------------------------------
# CAREER AGENT
# ------------------------------------------------------------------
def _career_uploader_key():
    # The nonce is bumped by "New chat" so a file that was picked but never
    # sent is discarded along with the conversation.
    return f"career_chat_attach_{st.session_state[UPLOADER_NONCE_KEY]}"


def _remove_career_attachment():
    """
    Discards whatever is currently staged in the attachment picker (the ×
    button on the attachment status card). A file_uploader can't be
    cleared programmatically, so - exactly like "New chat" already does -
    this bumps the nonce baked into its key, which makes Streamlit hand
    back a fresh, empty uploader under a new key next run.
    """
    st.session_state[UPLOADER_NONCE_KEY] = st.session_state.get(UPLOADER_NONCE_KEY, 0) + 1


def _career_attachment_extension_label(name):
    ext = os.path.splitext(name or "")[1].lstrip(".").upper()
    return ext or "FILE"


def _staged_career_attachment():
    """
    What (if anything) currently sits in the attachment picker, before
    the message is sent. Returns None, or a dict for
    _render_career_attachment_card() to display. Only does the cheap,
    non-parsing checks (src.career_chat.validate_attachment_upload) -
    the same check the send flow already runs - so an obviously bad file
    (wrong type / empty / too large) shows its error immediately without
    ever being parsed.
    """
    uploaded = st.session_state.get(_career_uploader_key())
    if uploaded is None:
        return None

    name = getattr(uploaded, "name", "the file")
    try:
        data = uploaded.getvalue()
    except Exception:
        return {"name": name, "state": "error", "detail": "Couldn't read the selected file."}

    try:
        validate_attachment_upload(name, len(data))
    except AttachmentError as exc:
        return {"name": name, "state": "error", "detail": str(exc)}

    return {"name": name, "state": "pending", "data": data}


def _render_career_attachment_card():
    """
    Compact status card shown directly above the composer whenever a
    file is staged in the attachment picker: name, a live
    reading/ready/error state, and a remove (x) button. Parsing reuses
    the exact same cached src.career_chat.get_or_parse_attachment() call
    the send flow makes (keyed by content hash), so previewing a file
    here and then sending it parses that file only once.
    """
    staged = _staged_career_attachment()
    if not staged:
        return

    with st.container(key="career_attach_status_card"):
        info_col, remove_col = st.columns([6, 1], gap="small")
        with info_col:
            placeholder = st.empty()
        with remove_col:
            if st.button(
                "×", key="career_attach_remove_btn",
                help="Remove attachment", use_container_width=True,
            ):
                _remove_career_attachment()
                st.rerun()

    ext_label = _career_attachment_extension_label(staged["name"])
    escaped_name = html.escape(staged["name"])

    def paint(state, status_text, detail_text):
        placeholder.markdown(
            f"<div class='career-attach-card state-{state}'>"
            f"<div class='career-attach-icon'>📄</div>"
            f"<div class='career-attach-info'>"
            f"<div class='career-attach-name'>{escaped_name}</div>"
            f"<div class='career-attach-status'>{status_text}</div>"
            f"<div class='career-attach-detail'>{detail_text}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if staged["state"] == "error":
        paint("error", "⚠ Couldn't read file", html.escape(staged["detail"]))
        return

    # Shown first so a slow PDF (OCR fallback) has a visible "reading"
    # moment before the placeholder is overwritten with the result below.
    paint("processing", "⟳ Reading...", f"{ext_label} • Processing…")
    try:
        get_or_parse_attachment(
            st.session_state[ATTACHMENT_CACHE_KEY], staged["name"], staged["data"]
        )
    except AttachmentError as exc:
        paint("error", "⚠ Couldn't read file", html.escape(str(exc)))
    else:
        paint("ready", "✓ Ready to use", f"{ext_label} • Successfully attached")


def _on_career_form_submit():
    """
    on_click handler of the arrow button. Streamlit runs it at the very
    start of the rerun, before the page is drawn, for BOTH ways of
    submitting the composer form: clicking the arrow and pressing Enter in
    the text box. It only reads the submitted text/file and queues them -
    the slow work (parsing, the agent call) happens afterwards in
    _process_career_queue(), so the user's message is on screen first.
    """
    typed = st.session_state.get("career_chat_input", "")
    uploaded = st.session_state.get(_career_uploader_key())

    upload = None
    if uploaded is not None:
        try:
            upload = {"name": uploaded.name, "data": uploaded.getvalue()}
        except Exception:
            upload = {"name": getattr(uploaded, "name", "the file"), "data": b""}

    enqueue_turn(st.session_state, typed, upload)

    # The picker now lives outside st.form (see career_agent_page()) so it
    # can update immediately when a file is picked, so clear_on_submit no
    # longer clears it for us. Bump the nonce here instead - the exact same
    # trick _remove_career_attachment() and "New chat" already use - so a
    # sent attachment doesn't linger, staged, in the composer afterwards.
    if uploaded is not None:
        st.session_state[UPLOADER_NONCE_KEY] = st.session_state.get(UPLOADER_NONCE_KEY, 0) + 1


def _ask_career_agent(question):
    """
    Starter prompts and "search again" buttons: queue the text exactly like
    a typed message, then rerun so the chat is drawn in its conversation
    state (no empty-state panel) before the agent starts.
    """
    if enqueue_turn(st.session_state, question):
        st.rerun()


class _LazySpinner:
    """
    The existing "Agent is working..." spinner, but closable early: it is
    shown while nothing has streamed yet and removed as soon as the answer
    starts to appear (or when the turn ends / is interrupted).
    """

    def __init__(self, label):
        self._context = st.spinner(label)
        self._is_open = False

    def open(self):
        if not self._is_open:
            self._context.__enter__()
            self._is_open = True

    def close(self):
        if self._is_open:
            self._is_open = False
            self._context.__exit__(None, None, None)


def _process_career_queue(viewport):
    """
    Answers every queued chat turn, drawing into the message viewport as it
    goes: the user's bubble immediately, then the assistant's answer
    progressively as tokens arrive. All state handling lives in
    career_chat.run_turn(); this function is only the drawing.
    """
    queue = st.session_state[QUEUE_KEY]
    if not queue:
        return

    assistant_avatar = get_chat_avatar_data_url("assistant-chat-avatar.png")
    user_avatar = get_chat_avatar_data_url("user-avatar.png")

    def get_agent():
        # Cached inside career_agent: rebuilt only when the resume,
        # profile or search preferences have changed.
        return get_cached_career_agent(
            candidate_profile=st.session_state.candidate_profile,
            resume_text=st.session_state.get("resume_text"),
            questionnaire_data=st.session_state.questionnaire_data,
            session_id=WORKSPACE_ID,
        )

    while queue:
        turn = queue[0]
        assistant_bubble = {"placeholder": None, "last_paint": 0.0}
        spinner = _LazySpinner("Agent is working...")

        def answer_placeholder():
            if assistant_bubble["placeholder"] is None:
                with viewport:
                    with st.chat_message("assistant", avatar=assistant_avatar):
                        assistant_bubble["placeholder"] = st.empty()
            return assistant_bubble["placeholder"]

        def show_user_message(text):
            with viewport:
                with st.chat_message("user", avatar=user_avatar):
                    st.markdown(text)

        def show_partial_answer(text):
            spinner.close()
            now = time.monotonic()
            # ~15 repaints a second is smooth and keeps websocket traffic low.
            if now - assistant_bubble["last_paint"] >= 0.06:
                answer_placeholder().markdown(format_chat_answer(text), unsafe_allow_html=True)
                assistant_bubble["last_paint"] = now

        # Opened inside `viewport` (career_messages_area), like every other
        # piece of conversation UI above (show_user_message,
        # answer_placeholder) - NOT as a bare child of the chat card. Before
        # this, "Agent is working..." was the one element drawn outside the
        # card's managed layout, so it briefly broke the card's width/rows
        # in the specific window between sending the first message and the
        # first streamed token (PDF-attached or not) - fixed here rather
        # than by widening or restyling the container itself.
        with viewport:
            spinner.open()
        try:
            answer = run_turn(
                st.session_state,
                turn,
                get_agent=get_agent,
                stream_fn=stream_agent_response,
                on_user_message=show_user_message,
                on_partial=show_partial_answer,
            )
        finally:
            spinner.close()

        # Final text (also replaces an in-progress or error state).
        answer_placeholder().markdown(format_chat_answer(answer), unsafe_allow_html=True)


# Shared fixed height for the two bottom Career Agent panels (Your saved
# jobs / Recent searches) so both scroll internally and stay perfectly
# aligned regardless of how many rows either one has.
AGENT_BOTTOM_PANEL_HEIGHT = 320


def career_agent_page():
    robot_svg_html = build_svg_hero_html("hero-robot.png", "Robot illustration", size=250)
    st.markdown(
        f"""
        <div class="hero-panel hero-panel-agent">
            <div class="hero-copy">
                <div class="hero-eyebrow">Your AI Career Assistant</div>
                <h1>Career <span style="color:var(--accent-text);">Agent</span></h1>
                <p>Ask for job search help, company research, skill-gap guidance, or application tracking in a natural conversation.</p>
                <div class="hero-feature-chips">
                    <span class="hero-feature-chip chip-blue">Real-time insights</span>
                    <span class="hero-feature-chip chip-purple">Personalized guidance</span>
                    <span class="hero-feature-chip chip-green">Smarter career moves</span>
                </div>
            </div>
            <div class="hero-visual">
                <div class="hero-illustration-stage">
                    {robot_svg_html}
                </div>
                <div class="hero-bubble hero-bubble-agent">Smarter career moves.<br>Brighter tomorrow.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_row = st.container(key="career_agent_top_row")

    capabilities = [
        (lucide_icon("search", size=20), "Search for jobs", "Discover relevant openings based on your skills, role, and preferences.", "blue"),
        (lucide_icon("sliders", size=20), "Refine searches", "Fine-tune roles, locations, skills, and other job preferences.", "blue"),
        (lucide_icon("compare", size=20), "Compare jobs", "Compare job matches, scores, requirements, and key differences.", "green"),
        (lucide_icon("bar-chart", size=20), "Analyze skill gaps", "Identify missing skills and get clear suggestions to improve your profile.", "amber"),
        (lucide_icon("building", size=20), "Research companies", "Explore companies, roles, requirements, and career opportunities.", "purple"),
        (lucide_icon("bookmark", size=20), "Save applications", "Save promising jobs and keep your applications organized.", "purple"),
    ]

    picked_prompt = None

    side, main = top_row.columns([0.96, 2.04], gap="medium")

    with side:
        with st.container(border=True, key="agent_capabilities_panel"):
            st.markdown("**What the agent can help with**")
            capability_cols = st.columns(2)
            for idx, (icon, title, desc, badge_class) in enumerate(capabilities):
                with capability_cols[idx % 2]:
                    st.markdown(
                        f"<div class='agent-capability-card'><div class='agent-capability-icon {badge_class}'>{icon}</div><div class='agent-capability-title'>{title}</div><div class='agent-capability-desc'>{desc}</div></div>",
                        unsafe_allow_html=True,
                    )

    with main:
        chat_box = st.container(border=True, key="career_chat_box")
        with chat_box:
            # Wrapped in its own keyed container purely so the fixed-height
            # internal layout below (see "CAREER AGENT CONVERSATION - FIXED
            # HEIGHT LAYOUT" in theme.css) has a stable selector to pin this
            # row to its natural size; no visual/behavioral change here.
            with st.container(key="career_chat_header_row"):
                header_col, new_chat_col = st.columns([4, 1], gap="small")
                with header_col:
                    st.markdown("**Assistant conversation**")
                with new_chat_col:
                    if st.button("New chat", icon=":material/add:", key="new_chat_btn", use_container_width=True):
                        reset_chat_state(st.session_state)
                        st.rerun()

            # The message viewport's real sizing now comes entirely from
            # CSS (flex: 1 1 auto inside the fixed-height chat card - see
            # "CAREER AGENT CONVERSATION - FIXED HEIGHT LAYOUT" in
            # theme.css): it always fills whatever space is left after the
            # header, the attachment card (if any) and the composer, so the
            # outer card's height never changes with the conversation, an
            # attachment, or the empty vs. non-empty state. The height=
            # value below is only Streamlit's own initial/no-JS height and
            # is overridden by CSS; kept as a constant rather than removed
            # so the container still behaves sensibly if CSS fails to load.
            has_conversation = bool(st.session_state.chat_history) or bool(st.session_state[QUEUE_KEY])
            messages_viewport = st.container(key="career_messages_area")
            with messages_viewport:
                if not has_conversation:
                    with st.container(key="career_empty_state_wrap"):
                        empty_left, empty_right = st.columns(
                            [0.38, 0.62], gap="medium", vertical_alignment="center"
                        )
                        with empty_left:
                            st.markdown(
                                "<div class='career-agent-empty-text'>"
                                "<div class='career-agent-empty-title'>Hi! I'm your "
                                "<span class='career-agent-empty-title-accent'>Career Agent</span></div>"
                                "<div class='career-agent-empty-subtitle'>"
                                "Ask me anything about job search, company research, skill gaps, "
                                "or applications. I'm here to help you make smarter career decisions."
                                "</div>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        with empty_right:
                            empty_state_robot_html = build_svg_hero_html(
                                "hero-robot-avatar.png", "Career Agent", size=580
                            )
                            st.markdown(
                                f"<div class='career-agent-empty-avatar-wrap'>{empty_state_robot_html}</div>",
                                unsafe_allow_html=True,
                            )

                        st.markdown("<div class='career-agent-prompt-row'>", unsafe_allow_html=True)
                        starter_prompt_cols = st.columns(3, gap="small")
                        starter_prompts = [
                            "Find relevant jobs",
                            "What skills should I learn next?",
                            "Research a company",
                        ]
                        for col, prompt_text in zip(starter_prompt_cols, starter_prompts):
                            with col:
                                if st.button(
                                    prompt_text,
                                    key=f"empty_state_{prompt_text[:10]}",
                                    use_container_width=True,
                                ):
                                    picked_prompt = prompt_text
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # Use the compact robot avatar and the neutral, gender-free
                    # blue/purple user avatar from the approved reference UI.
                    # The avatars are presentation-only; chat history/state and
                    # all agent functionality remain unchanged.
                    assistant_avatar = get_chat_avatar_data_url("assistant-chat-avatar.png")
                    user_avatar = get_chat_avatar_data_url("user-avatar.png")
                    for role, message in st.session_state.chat_history:
                        avatar = assistant_avatar if role == "assistant" else user_avatar
                        with st.chat_message(role, avatar=avatar):
                            if role == "assistant":
                                st.markdown(format_chat_answer(message), unsafe_allow_html=True)
                            else:
                                st.markdown(message)

            # Attachment status card: only drawn once a file is staged in
            # the picker below, directly above the composer row, inside
            # this same chat card - see _render_career_attachment_card().
            _render_career_attachment_card()

            # This used to be a single st.form (attachment + text + send).
            # A file_uploader inside a form only reports its selection to
            # session_state - and reruns the app - when the form is
            # submitted, so a newly-picked file sat invisible until Send
            # was pressed. The picker now lives in this plain (non-form)
            # container instead, so picking a file reruns the app right
            # away and _render_career_attachment_card() (drawn just above,
            # unchanged) can paint "Reading..." then "Ready to use" before
            # anything is sent. Text input + send stay together in their
            # own nested form so Enter-to-submit and the arrow button both
            # still land in _on_career_form_submit() exactly as before.
            with st.container(key="career_chat_form"):
                # Attachment and send are compact icon buttons (see the
                # "COMPACT COMPOSER ICONS" CSS block) so the text input gets
                # the largest share of the row's width.
                clip_col, rest_col = st.columns([0.5, 6.5], gap="small")
                with clip_col:
                    # Real file picker, drawn as a compact paperclip button
                    # (see the "CAREER AGENT ATTACHMENT ICON" CSS block near
                    # the end of assets/theme.css). Wrapped in its own keyed
                    # container - a static, always-present class Streamlit
                    # guarantees on it - rather than styled purely by the
                    # uploader's internal test-ids, which turned out not to
                    # match this Streamlit version and left the native
                    # "Upload / limit 200MB" widget showing through.
                    with st.container(key="career_chat_attach_box"):
                        st.file_uploader(
                            "Attach a file",
                            key=_career_uploader_key(),
                            label_visibility="collapsed",
                        )
                with rest_col:
                    with st.form("career_chat_form_inner", clear_on_submit=True, border=False):
                        input_col, send_col = st.columns([6.0, 0.5], gap="small")
                        with input_col:
                            st.text_input(
                                "Ask your career question...",
                                key="career_chat_input",
                                placeholder="Ask your career question...",
                                label_visibility="collapsed",
                            )
                        with send_col:
                            # Now the form's only submit button, so Enter in
                            # the text box and a click on the arrow both
                            # land in _on_career_form_submit().
                            st.form_submit_button(
                                "", icon=":material/arrow_forward:", type="primary", use_container_width=True,
                                help="Send", key="career_chat_send_btn",
                                on_click=_on_career_form_submit,
                            )

            if picked_prompt:
                _ask_career_agent(picked_prompt)

            _process_career_queue(messages_viewport)

    st.markdown("<div class='agent-bottom-row-spacer'></div>", unsafe_allow_html=True)
    saved_col, recent_col = st.columns([1.7, 1], gap="medium")
    with saved_col:
        with st.container(border=True, key="saved_jobs_bottom_panel"):
            render_saved_jobs_panel()
    with recent_col:
        with st.container(border=True, key="recent_searches_bottom_panel"):
            render_recent_searches_panel()


def render_recent_searches_panel():
    # The "View all" toggle button has been removed (task: remove it
    # entirely, no reserved space) - the panel now always shows the
    # same fixed-size recent-searches list, so there's no longer a
    # second (view-all) limit or session_state flag to track.
    recent_searches = get_recent_searches(limit=20, session_id=WORKSPACE_ID)

    # Header now spans the full row on its own (previously split with
    # a "View all" toggle column) - the heading/description use the
    # available width naturally instead of leaving empty space where
    # the button used to be.
    st.markdown(
        f"<div class='saved-jobs-panel-title' style='display:flex;align-items:center;gap:0.45rem;'>{lucide_icon('clock', size=17)}"
        "<span>Recent searches</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Click the refresh icon to run a search again.")

    picked_prompt = None

    if not recent_searches:
        st.info("No searches yet. Run a search from the Find Jobs tab to see it here.")
        return picked_prompt

    list_area = st.container(height=AGENT_BOTTOM_PANEL_HEIGHT, key="recent_searches_scroll_area")
    with list_area:
        last_idx = len(recent_searches) - 1
        for idx, search in enumerate(recent_searches):
            # Only the role/title part is shown — the location
            # subtext is dropped here to keep each row a single
            # clean line (still summarized in the search itself).
            title_part = search["summary"].split(" · ")[0]
            search_meta = format_time_ago(search["searched_at"])

            # Previous attempt made the search icon clickable by layering
            # an invisible button on top of decorative markup via CSS
            # absolute-positioning - that overlay was not receiving
            # clicks reliably. Replaced with a real Streamlit button in
            # its own column (the same proven pattern the working X
            # button already uses below), so the click is guaranteed to
            # register. action_col's width fraction (1 out of the same
            # 7 total units as before: 0.6 + 5.4 + 1 = 7, same as the
            # prior 6 + 1 = 7) is unchanged, so the X button keeps its
            # exact previous size/position.
            icon_col, text_col, action_col = st.columns(
                [0.6, 5.4, 1], gap="small", vertical_alignment="center"
            )
            with icon_col:
                # Real, independent button - its own key/callback, so a
                # click here can only ever run search-again, never the
                # delete action in action_col.
                if st.button(
                    "",
                    icon=":material/search:",
                    key=f"recent_search_run_{search['id']}",
                    help=f"Search again for {title_part}",
                    use_container_width=True,
                ):
                    picked_prompt = f"Search again for: {search['summary']}"
            with text_col:
                # No per-row inline style overrides here anymore — every
                # row now shares exactly the same '.recent-search-row'
                # box (fixed height, same padding), so rows line up.
                # The divider between rows is drawn once below, as its
                # own full-width element, instead of also being part of
                # this row (that duplication was what made spacing
                # inconsistent/row heights vary). The icon itself now
                # lives in icon_col as a real button, so this markup
                # only holds the title + time.
                st.markdown(
                    "<div class='recent-search-row'>"
                    "<div class='recent-search-text'>"
                    f"<div class='recent-search-title'>{title_part}</div>"
                    "</div>"
                    f"<div class='recent-search-time'>{search_meta}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with action_col:
                # The X button is its own independent Streamlit widget
                # with its own key/callback - a click here can only ever
                # fire this branch, never the search-icon button above
                # (Streamlit gives each button its own click event; they
                # don't share state or bubble into one another). This is
                # now wired to delete_recent_search() instead of
                # re-running the search - previously it incorrectly
                # reused the search-again action. Icon/size/position/
                # border/background/radius/spacing are unchanged from
                # the prior pass.
                if st.button(
                    "",
                    icon=":material/close:",
                    key=f"recent_search_delete_{search['id']}",
                    help="Remove this search",
                    use_container_width=True,
                ):
                    delete_recent_search(search["id"], session_id=WORKSPACE_ID)
                    st.rerun()
            if idx != last_idx:
                st.markdown("<div class='recent-search-divider'></div>", unsafe_allow_html=True)

    if picked_prompt:
        _ask_career_agent(picked_prompt)
    return picked_prompt


def render_saved_jobs_panel():
    saved_jobs = get_saved_jobs(session_id=WORKSPACE_ID)
    status_options = ["saved", "applied", "interview", "rejected", "offer"]

    st.markdown(
        f"<div class='saved-jobs-panel-title' style='display:flex;align-items:center;gap:0.45rem;'>{lucide_icon('bookmark', size=17)}"
        "<span>Your saved jobs</span></div>",
        unsafe_allow_html=True,
    )
    st.caption('Ask the agent to "save this job" or "mark X as applied" — or manage them directly below.')
    st.caption("Saved jobs are private to this browser link — bookmark this page's URL to return to them later.")

    if not saved_jobs:
        st.info(
            "No jobs saved yet. Save one from the Find Jobs tab's results, "
            'or ask the agent above to "save this job".'
        )
        return

    visible_jobs = saved_jobs

    avatar_palette = ["blue", "green", "purple", "amber"]

    list_area = st.container(height=AGENT_BOTTOM_PANEL_HEIGHT, key="saved_jobs_scroll_area")
    with list_area:
        for row_idx, saved in enumerate(visible_jobs):
            with st.container(border=True, key=f"saved_row_{saved['id']}"):
                c1, c2, c3, c4, c5 = st.columns([0.5, 2.7, 1, 1.3, 0.6], gap="small", vertical_alignment="center")
                title = saved.get("title", "Untitled")
                company = saved.get("company", "Unknown company")
                initials = "".join(word[0] for word in title.split()[:2] if word).upper() or "?"
                palette_class = avatar_palette[row_idx % len(avatar_palette)]
                with c1:
                    st.markdown(
                        f"<div class='saved-job-avatar {palette_class}'>{initials}</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    if saved.get("url"):
                        st.link_button(f"{title} · {company}", saved["url"], use_container_width=True)
                    else:
                        st.markdown(f"**{title}** · {company}")
                with c3:
                    score = saved.get("final_score")
                    st.markdown(score_badge_html(score) if score is not None else "—", unsafe_allow_html=True)
                with c4:
                    date_saved = saved.get("date_saved")
                    saved_ago = format_time_ago(date_saved) if date_saved else "recently"
                    st.caption(f"Saved {saved_ago}")
                with c5:
                    current = saved.get("status", "saved")
                    current = current if current in status_options else "saved"
                    with st.popover("⋯", use_container_width=True):
                        new_status = st.selectbox(
                            "Status", status_options, index=status_options.index(current),
                            key=f"agent_tab_status_{saved['id']}",
                        )
                        if new_status != current:
                            update_job_status(saved["id"], new_status, session_id=WORKSPACE_ID)
                            st.rerun()
                        if st.button(
                            "Remove from saved", icon=":material/close:",
                            key=f"agent_tab_delete_{saved['id']}", use_container_width=True,
                        ):
                            delete_saved_job(saved["id"], session_id=WORKSPACE_ID)
                            st.rerun()


# ------------------------------------------------------------------
# ANALYTICS
# ------------------------------------------------------------------
def analytics_page():
    st.title("Career analytics")
    st.caption("Insights and trends from your job-search journey.")

    saved_jobs = get_saved_jobs(session_id=WORKSPACE_ID)
    status_order = ["saved", "applied", "interview", "offer", "rejected"]
    counts = compute_status_counts(saved_jobs)
    total_viewed = max(st.session_state.jobs_viewed, len(saved_jobs))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Jobs viewed", total_viewed)
    k2.metric("Jobs saved", len(saved_jobs))
    k3.metric("Applied", counts["applied"])
    k4.metric("Interviews", counts["interview"])
    k5.metric("Offers", counts["offer"])

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("**Application status distribution**")
            chart_df = pd.DataFrame({"Status": status_order, "Jobs": [counts[s] for s in status_order]}).set_index("Status")
            st.bar_chart(chart_df, height=260)

    with c2:
        with st.container(border=True):
            st.markdown("**Top missing skills**")
            st.caption("Based on the latest analyzed jobs.")
            top_missing = aggregate_missing_skills(st.session_state.scored_jobs or [], top_n=8)
            if top_missing:
                skill_df = pd.DataFrame(top_missing, columns=["Skill", "Jobs"])
                st.bar_chart(skill_df.set_index("Skill"), height=220)
            else:
                st.info("Run a job search to generate missing-skill analytics.")

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        with st.container(border=True):
            st.markdown("**Most suitable roles**")
            st.caption("Job titles scoring 70+ in your latest search.")
            suitable_roles = aggregate_suitable_roles(st.session_state.scored_jobs or [], top_n=8)
            if suitable_roles:
                st.write("\n".join(f"• {title} ({count})" for title, count in suitable_roles))
            else:
                st.info("Upload a resume and run a search to see suitable roles.")

    with c4:
        with st.container(border=True):
            st.markdown("**Top companies in your searches**")
            companies = {}
            for job in st.session_state.scored_jobs or []:
                company = str(job.get("company", "")).strip()
                if company:
                    companies[company] = companies.get(company, 0) + 1
            if companies:
                company_df = pd.DataFrame(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:8], columns=["Company", "Jobs"])
                st.bar_chart(company_df.set_index("Company"), height=220)
            else:
                st.info("Run a search to populate company analytics.")


# ------------------------------------------------------------------
# RENDER
# ------------------------------------------------------------------
# Analytics remains available as an internal helper module, but the
# visible Streamlit navigation is intentionally limited to the two
# main product tabs requested for the redesigned UI.

render_topbar()

tab_search, tab_agent = render_navbar()

with tab_search:
    job_search_page()

with tab_agent:
    career_agent_page()
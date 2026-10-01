"""
ui_helpers.py
=============
Pure functions that turn scored_jobs (the list your pipeline already
produces) into shapes the UI can render — a table, a chart's data.

These have NO Streamlit imports and NO st.* calls. That's deliberate:
it means we can test them with plain Python (see test_ui_helpers.py)
without needing to launch a browser or a Streamlit server.

NOTE ON THE CHART: Gemini currently returns one holistic match_score,
not separate sub-scores per category (Skill/Role/Location/etc.) — that
granular breakdown is what Step 6 (5-field Gemini scoring) will add.
Until then, the most honest chart we can show is the pre-rank score
(cheap, rule-based, computed before Gemini ever runs) next to the
Gemini score (the real match evaluation) — this at least shows the
user how much Gemini's reasoning changed the ranking versus a simple
keyword match.
"""

import re

import pandas as pd


SCORE_PATTERN = re.compile(r"\b(\d{1,3})\s*/\s*100\b")


def score_badge_class(score):
    """Good/warn/risk bucket for a 0-100 score, matching the same
    75/50 thresholds match_card.py's _score_color() already uses,
    so a score reads the same color everywhere in the app."""
    if score is None:
        return "muted"
    if score >= 75:
        return "good"
    if score >= 50:
        return "warn"
    return "risk"


def score_badge_html(score, css_class="saved-job-score"):
    """A small colored pill for a 'NN/100' score, e.g. for saved-job rows."""
    bucket = score_badge_class(score)
    return f'<span class="{css_class} {bucket}">{score}/100</span>'


def format_chat_answer(text):
    """
    Wraps any 'NN/100' mention in the career agent's reply in a small
    colored score badge (matching the same visual language as the
    Find Jobs tab's match cards), so a job or ATS score the agent
    talks about reads as a scored chip instead of plain prose.

    Safe to pass into st.markdown(..., unsafe_allow_html=True): normal
    markdown syntax (bold, lists, links) is untouched, only literal
    'NN/100' substrings are replaced with an inline <span>.
    """
    if not text:
        return text

    def _replace(match):
        score = int(match.group(1))
        return score_badge_html(score, css_class="chat-score-badge")

    return SCORE_PATTERN.sub(_replace, text)


def build_results_table(scored_jobs):
    """
    Turns scored_jobs into a pandas DataFrame for the top ranked table,
    sorted best-score-first.
    """
    if not scored_jobs:
        return pd.DataFrame(
            columns=["Title", "Company", "Score", "Recommendation", "Source"]
        )

    rows = []
    for job in scored_jobs:
        rows.append({
            "Title": job.get("title", "N/A"),
            "Company": job.get("company", "N/A"),
            "Score": job.get("match_score", 0),
            "Recommendation": job.get("recommendation", ""),
            "Source": job.get("source", ""),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


def get_score_comparison_chart_data(job):
    """
    Returns a small DataFrame with two bars: the cheap pre-rank score
    vs. the real Gemini score, for one job. Used inside each job's
    expander.
    """
    pre_score = job.get("_pre_score", 0)
    gemini_score = job.get("match_score", 0)

    return pd.DataFrame(
        {"Score": [pre_score, gemini_score]},
        index=["Pre-rank (rule-based)", "Gemini (AI evaluated)"],
    )


def format_skill_list(skills):
    """Turns a list of skills into a display string, or an em-dash if empty."""
    if not skills:
        return "—"
    return ", ".join(skills)


def get_expander_label(job):
    """Builds the one-line label shown on each collapsed job expander."""
    title = job.get("title", "N/A")
    company = job.get("company", "N/A")
    score = job.get("match_score", 0)
    recommendation = job.get("recommendation", "")
    return f"{title} — {company}  ·  {score}/100  ·  {recommendation}"
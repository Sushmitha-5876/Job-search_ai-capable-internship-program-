"""
match_card.py
=============
Renders one job as an "Explainable Job Match Card" — a score ring,
a 7-category breakdown (including company preference), matching/
missing skills, and a red-flag risk box. Replaces the old "pre-rank
vs Gemini" bar chart, which only compared two numbers and never
showed WHY a job scored what it did.

Design note: this is plain Streamlit + a small HTML/CSS block (no new
dependency like plotly), since the ring and colored bars are easier
to get pixel-control over with CSS than with Streamlit's own chart
widgets.
"""

import html
from urllib.parse import quote_plus

import streamlit as st

from .skill_gap_roadmap import build_skill_roadmap
from .database import DEFAULT_SESSION_ID, save_job


CATEGORY_LABELS = {
    "skill_match": "Skills",
    "role_relevance": "Role Relevance",
    "technology_domain": "Technology / Domain",
    "experience": "Experience",
    "location": "Location",
    "salary": "Salary",
    "company_preference": "Company Preference",
    "notice_period": "Notice Period",
}

CARD_CSS = """
<style>
/* Ring + score number + recommendation label are rendered together as
   ONE flex-column block (see render_match_card) so the label can never
   overlap the ring above it - there is no separate Streamlit element
   boundary between them to misalign. */
.match-card-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    gap: 8px;
    /* A real minimum footprint. Without one the wrapper could collapse
       shorter than the 92px ring it contains, letting the ring bleed
       over whatever Streamlit rendered underneath it. */
    min-height: 132px;
    padding: 4px 0 10px;
}
.match-card-ring {
    width: 92px; height: 92px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: conic-gradient(var(--ring-color) calc(var(--pct) * 1%), var(--score-track) 0);
    flex: 0 0 auto;
}
.match-card-ring-inner {
    width: 74px; height: 74px; border-radius: 50%;
    background: var(--score-panel); display: flex; align-items: center;
    justify-content: center; font-size: 21px; font-weight: 600;
    color: var(--text);
}
.match-card-recommendation {
    text-align: center;
    font-weight: 700;
    font-size: 0.85rem;
    line-height: 1.3;
}
.match-card-why {
    margin-bottom: 6px;
    line-height: 1.5;
}
/* A clear top border + margin separates the primary score/why block
   above from the breakdown grid, the same "own group" treatment the
   skills grid already gets below - so the overall score reads as the
   headline result rather than blending into the categories under it. */
.match-cat-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px 18px;
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid var(--score-border);
}
@media (max-width: 760px) {
    .match-cat-grid { grid-template-columns: minmax(0, 1fr); }
}
.match-cat-box {
    border: 1px solid var(--score-border); border-radius: var(--r-sm); padding: 12px 14px;
    background: var(--score-panel);
    /* No margin-bottom: the grid's own gap now owns the spacing, so
       boxes can't stack unevenly or run into the row below. */
}
.match-cat-box-head {
    display: flex; align-items: baseline; justify-content: space-between; gap: 10px;
    font-size: 0.88rem;
}
.match-cat-box-score { font-family: var(--font-mono); white-space: nowrap; }
.match-cat-box-head { flex-wrap: wrap; row-gap: 2px; }
.match-cat-box-score-na { font-family: inherit; font-size: 0.78rem; white-space: nowrap; margin-left: auto; }
.match-cat-bar-bg {
    background: var(--score-track); border-radius: 4px; height: 6px; margin-top: 6px;
}
.match-cat-bar-fill {
    height: 6px; border-radius: 4px;
}
/* Skills section. This whole block (both headings AND both tag lists)
   is rendered as ONE html element, with a clear top border + generous
   margin, so the headings can never sit flush against - or on top of -
   the category boxes above them. */
.match-skills-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px 20px;
    margin-top: 24px;
    padding-top: 18px;
    border-top: 1px solid var(--score-border);
}
@media (max-width: 760px) {
    .match-skills-grid { grid-template-columns: minmax(0, 1fr); }
}
.match-skills-col { min-width: 0; }
.match-skills-title {
    font-size: 0.9rem; font-weight: 700; color: var(--text);
    margin-bottom: 8px;
}
.match-skills-tags {
    display: flex; flex-wrap: wrap; gap: 8px;
}
.skill-tag {
    display: inline-block; border: 1px solid var(--score-border); border-radius: 14px;
    padding: 4px 11px; margin: 0; font-size: 13px; color: var(--muted);
    line-height: 1.5;
}
.skill-tag-match { border-color: var(--good); color: var(--good); background: var(--good-soft); }
.skill-tag-missing { border-color: var(--risk); color: var(--risk); background: var(--risk-soft); }
.risk-box {
    border-radius: var(--r-sm); padding: 12px 14px; margin-top: 24px; margin-bottom: 8px;
    line-height: 1.5;
}
.risk-box-high { background: var(--risk-soft); border: 1px solid var(--risk); }
.risk-box-clear { background: var(--good-soft); border: 1px solid var(--good); }
</style>
"""


def _score_color(score):
    """Good / accent / risk, matching the same thresholds get_recommendation() uses.
    Returns CSS custom-property references so this always stays in sync
    with the app's shared design tokens (assets/theme.css) instead of
    carrying its own separate hardcoded palette.
    """
    if score is None:
        return "var(--muted-soft)"
    if score >= 75:
        return "var(--good)"
    if score >= 50:
        return "var(--warn)"
    return "var(--risk)"


def _category_bar_html(label, score):
    if score is None:
        return (
            f'<div class="match-cat-box"><div class="match-cat-box-head">'
            f'<span>{label}</span>'
            f'<span class="match-cat-box-score match-cat-box-score-na" style="color:var(--muted-soft);">Not available</span>'
            f'</div><div class="match-cat-bar-bg"></div></div>'
        )
    color = _score_color(score)
    # float:right was what let a long label and its score sit on the same
    # line and visually collide; a flex header keeps them apart instead.
    return (
        f'<div class="match-cat-box"><div class="match-cat-box-head">'
        f'<span>{label}</span>'
        f'<span class="match-cat-box-score">{score}/100</span>'
        f'</div>'
        f'<div class="match-cat-bar-bg">'
        f'<div class="match-cat-bar-fill" style="width:{score}%;background:{color};"></div>'
        f'</div></div>'
    )


def _skill_tags_html(skills, kind):
    # Skill names ultimately trace back to job-listing text from
    # external, unauthenticated APIs (RemoteOK/Adzuna/IndianAPI), so
    # they're escaped before going into raw HTML — a skill name like
    # "<script>" should render as literal text, not run as markup.
    css_class = "skill-tag-match" if kind == "match" else "skill-tag-missing"
    if not skills:
        return '<span style="color:var(--muted-soft);font-size:13px;">None listed</span>'
    return "".join(
        f'<span class="skill-tag {css_class}">{html.escape(str(s))}</span>'
        for s in skills
    )


def render_match_card(job, session_id=DEFAULT_SESSION_ID):
    """
    Renders one job as a full match card. Call once per job, inside
    an st.expander (see app.py) — mirrors how the old bar-chart call
    was used.
    """
    score = job.get("final_score")
    recommendation = job.get("recommendation", "")
    reason = job.get("reason", "No reason provided.")
    category_scores = job.get("category_scores", {})
    categories_excluded = job.get("categories_excluded", [])
    red_flags = job.get("red_flags", []) or []
    score_capped = job.get("score_capped", False)

    st.markdown(CARD_CSS, unsafe_allow_html=True)

    ring_color = _score_color(score)
    ring_pct = score if score is not None else 0

    col_ring, col_why = st.columns([1, 3], vertical_alignment="center")

    with col_ring:
        recommendation_safe = html.escape(str(recommendation)) if recommendation else ""
        st.markdown(
            f'<div class="match-card-ring-wrap">'
            f'<div class="match-card-ring" style="--pct:{ring_pct};--ring-color:{ring_color};">'
            f'<div class="match-card-ring-inner">{score if score is not None else "N/A"}</div>'
            f'</div>'
            f'<div class="match-card-recommendation">{recommendation_safe}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_why:
        st.markdown(f'<div class="match-card-why"><b>Why this score:</b> {html.escape(str(reason))}</div>', unsafe_allow_html=True)
        if categories_excluded:
            excluded_labels = ", ".join(
                CATEGORY_LABELS.get(c, c) for c in categories_excluded
            )
            st.caption(f"Not scored (missing data): {excluded_labels}")

    # ------------------------------------------------------
    # Category breakdown — 2 per row
    # ------------------------------------------------------
    # One CSS grid rendered as a single element, instead of four separate
    # st.columns rows. Each of those rows was its own Streamlit block with
    # its own height, which is how the last row ended up butting into the
    # skills headings below it.
    cat_html = "".join(
        _category_bar_html(label, category_scores.get(key))
        for key, label in CATEGORY_LABELS.items()
    )
    st.markdown(
        '<div class="match-cat-grid" style="display:grid;'
        'grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 18px;'
        'margin-top:18px;padding-top:16px;border-top:1px solid var(--score-border);">'
        f'{cat_html}</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------
    # Skills
    # ------------------------------------------------------
    grid_style = (
        "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));"
        "gap:14px 20px;margin-top:24px;padding-top:18px;"
        "border-top:1px solid var(--score-border);"
    )
    tags_style = "display:flex;flex-wrap:wrap;gap:8px;"
    title_style = (
        "font-size:0.92rem;font-weight:700;color:var(--text);margin-bottom:8px;"
    )
    st.markdown(
        f'<div class="match-skills-grid" style="{grid_style}">'
        '<div class="match-skills-col" style="min-width:0;">'
        f'<div class="match-skills-title" style="{title_style}">Matching Skills</div>'
        f'<div class="match-skills-tags" style="{tags_style}">'
        f'{_skill_tags_html(job.get("matching_skills", []), "match")}</div>'
        '</div>'
        '<div class="match-skills-col" style="min-width:0;">'
        f'<div class="match-skills-title" style="{title_style}">Missing Skills</div>'
        f'<div class="match-skills-tags" style="{tags_style}">'
        f'{_skill_tags_html(job.get("missing_skills", []), "missing")}</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------
    # Skill Gap Roadmap — turns missing_skills into concrete next
    # steps. Built from a static curated map (skill_gap_roadmap.py),
    # not an extra Gemini call — see that file's docstring for why.
    # ------------------------------------------------------
    roadmap = build_skill_roadmap(job.get("missing_skills", []))
    if roadmap:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        with st.expander("How to close these skill gaps"):
            for item in roadmap:
                st.markdown(
                    f"- **{item['skill']}** — {item['resource']} "
                    f"(~{item['estimated_time']})"
                )

    # ------------------------------------------------------
    # Risk status
    # ------------------------------------------------------
    if red_flags:
        flags_html = "".join(
            f"<div>{html.escape(str(flag))}</div>" for flag in red_flags
        )
        capped_note = (
            "<div style='margin-top:6px;'><b>Score capped due to red flags.</b></div>"
            if score_capped else ""
        )
        # A red flag is a "look closer" signal, not a verdict - give the
        # candidate a one-click way to actually check the company instead
        # of having to switch tabs and type the search themselves.
        company = (job.get("company") or "").strip()
        verify_links_html = ""
        if company:
            google_url = f"https://www.google.com/search?q={quote_plus(company + ' reviews')}"
            linkedin_url = f"https://www.linkedin.com/search/results/companies/?keywords={quote_plus(company)}"
            company_safe = html.escape(company)
            verify_links_html = (
                "<div style='margin-top:8px;'>"
                f"Verify <b>{company_safe}</b>: "
                f"<a href='{html.escape(google_url)}' target='_blank'>Search reviews</a>"
                " &nbsp;|&nbsp; "
                f"<a href='{html.escape(linkedin_url)}' target='_blank'>Find on LinkedIn</a>"
                "</div>"
            )
        st.markdown(
            f'<div class="risk-box risk-box-high" style="margin-top:24px;"><b>Risk Status: HIGH</b>{flags_html}{capped_note}{verify_links_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="risk-box risk-box-clear" style="margin-top:24px;"><b>Risk Status: CLEAR</b> — no red flags detected.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # Action row
    # ------------------------------------------------------
    if job.get("url"):
        st.link_button("View job posting", job["url"])

    save_key = f"save_{job.get('url', '')}_{job.get('title', '')}_{job.get('company', '')}"
    if st.button("Save job", key=save_key):
        was_saved = save_job(job, session_id=session_id)
        if was_saved:
            st.success("Job saved! Check the Saved Jobs section below.")
        else:
            st.info("This job is already in your saved list.")
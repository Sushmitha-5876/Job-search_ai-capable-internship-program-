"""
pdf_export.py
=============
Generates a downloadable PDF report of a completed job search:
search preferences, candidate profile summary, a results overview
table, and a full scoring breakdown for every scored job.

WHY THIS EXISTS: closes the "Export functionality (PDF job lists)"
item from the Track A project checklist. Uses reportlab's Platypus
API (SimpleDocTemplate + flowables), per this environment's pdf
skill guidance for creating multi-page documents with tables.

DESIGN NOTE: this module only READS from job dicts, candidate_profile,
and questionnaire_answers - it doesn't recompute or alter any score.
It's a presentation layer over data that's already been produced by
the scoring pipeline, so there's no risk of the PDF ever disagreeing
with what the app itself displays.

SCOPE: per-job detail is the full 8-category breakdown for every
scored job (not just the top N) - agreed as the intended behavior
even though it makes the PDF run long (~half a page per job) for
15-20 job searches. A candidate exporting this wants the whole
picture to reference later or share, not a truncated version.

------------------------------------------------------------------
FIX LOG (layout pass):
1. Long values (Preferred Companies, Experience, Education, job
   titles/companies) were being placed in Table cells as raw strings.
   reportlab Table cells do NOT wrap plain strings - they clip/overflow
   instead. Every cell that can hold more than a couple words is now
   wrapped in a Paragraph, which wraps within its column width.
2. "experience" and "education" are lists of dicts (e.g.
   {"role": ..., "company": ..., "duration": ...}) - the old code ran
   str(dict) on them, which prints the literal Python repr. New
   _format_experience_entry()/_format_education_entry() pull out the
   real fields into readable text. Gemini's resume-parsing prompt
   doesn't pin down exact key names, so these formatters check several
   likely keys AND fall back to showing any other keys as "Label:
   value" - so nothing is ever silently dropped even if a resume
   produces slightly different field names.
3. All dynamic text (job titles/descriptions, company names, resume
   data, anything from Gemini or the job APIs) now goes through a
   small XML-escaping helper before being placed in a Paragraph.
   Paragraph text is parsed as a mini markup language - an unescaped
   "&" (e.g. "Johnson & Johnson", "AT&T") or "<"/">" in real-world data
   would previously either crash the PDF build or mangle the output.
4. The Results Overview table now has explicit column padding and
   slightly re-balanced column widths that use the full page width.
5. Each job's compact header (title, score line, "why this score",
   category table) is now wrapped in KeepTogether, so it either fits
   fully on the current page or moves entirely to the next one -
   it can no longer be split with the category table starting at the
   very bottom of a page. The looser content below it (skill lists,
   roadmap, red flags) is left to flow normally, which is the right
   behavior for open-ended-length bullet lists.
------------------------------------------------------------------
"""

from datetime import datetime
from io import BytesIO
from urllib.parse import quote_plus
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)


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

# Keys checked (in order) when formatting one experience/education dict
# entry. Any OTHER key present on the dict is still shown (as a small
# "Label: value" line) - these lists only control which fields get top
# billing in the header line, not what's allowed to display at all.
_EXPERIENCE_ROLE_KEYS = ["role", "title", "position", "job_title"]
_EXPERIENCE_COMPANY_KEYS = ["company", "organization", "employer"]
_EXPERIENCE_DURATION_KEYS = ["duration", "dates", "period", "years"]

_EDUCATION_DEGREE_KEYS = ["degree", "qualification"]
_EDUCATION_INSTITUTION_KEYS = ["institution", "university", "college", "school"]
_EDUCATION_YEAR_KEYS = ["year", "graduation_year", "duration", "dates"]


def _safe(value):
    """
    XML-escapes a value before it goes inside a reportlab Paragraph.
    Paragraph text is parsed as a small HTML-like markup, so raw
    "&"/"<"/">" from real job/company/resume data (e.g. "AT&T",
    "R&D", "C++ <-> Java") would otherwise break parsing or render
    incorrectly. Use this on every piece of DYNAMIC text; leave your
    own literal tags (<b>, <br/>, etc.) outside of this call.
    """
    if value is None:
        return ""
    return _xml_escape(str(value))


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], fontSize=9,
        textColor=colors.grey, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="JobHeading", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4,
    ))
    # Wrap-capable styles for table cells (kv tables + overview table).
    styles.add(ParagraphStyle(
        name="CellLabel", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9.5, leading=12,
    ))
    styles.add(ParagraphStyle(
        name="CellValue", parent=styles["Normal"], fontSize=9.5, leading=13,
    ))
    styles.add(ParagraphStyle(
        name="CellValueSmall", parent=styles["Normal"], fontSize=8.5, leading=11,
    ))
    styles.add(ParagraphStyle(
        name="ExtraDetail", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#6b7280"),
    ))
    return styles


def _kv_table(rows, styles):
    """
    A 2-column label/value table used for the preferences and profile
    summary sections. Every value is wrapped in a Paragraph so long
    text (a long company list, multi-line experience/education) wraps
    within the column instead of overflowing past the table's edge.
    """
    data = [
        [Paragraph(_safe(label), styles["CellLabel"]), Paragraph(value, styles["CellValue"])]
        for label, value in rows
    ]
    table = Table(data, colWidths=[5 * cm, 11 * cm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    return table


def _join_or_none_listed(items):
    """For simple flat lists of strings (skills, companies) - safe to
    escape-and-join since these are always plain strings, not dicts."""
    items = [_safe(item).strip() for item in (items or []) if str(item).strip()]
    return ", ".join(items) if items else "None listed"


def _format_dict_entry(item, primary_key_groups):
    """
    Shared logic for one experience/education dict entry.

    primary_key_groups: list of lists, e.g.
        [["role","title"], ["company","organization"], ["duration"]]
    Each group contributes at most one header segment (first matching
    key found). Any key on the dict NOT covered by any group is still
    shown, as a small "Label: value" line - so an unexpected field name
    from Gemini never silently disappears.
    """
    if not isinstance(item, dict):
        text = _safe(item).strip()
        return text or None

    used_keys = set()
    header_parts = []
    for group in primary_key_groups:
        value = None
        for key in group:
            if item.get(key):
                value = item.get(key)
                used_keys.add(key)
                break
        if value:
            header_parts.append(_safe(value).strip())

    header = " — ".join(part for part in header_parts if part)

    extras = []
    for key, value in item.items():
        if key in used_keys:
            continue
        value_text = _safe(value).strip() if value is not None else ""
        if not value_text:
            continue
        label = key.replace("_", " ").strip().title()
        extras.append(f"{label}: {value_text}")

    if header and extras:
        return f"{header}<br/><font size=8 color='#6b7280'>{' | '.join(extras)}</font>"
    if header:
        return header
    if extras:
        return " | ".join(extras)
    return None


def _format_experience_entry(item):
    return _format_dict_entry(
        item,
        [_EXPERIENCE_ROLE_KEYS, _EXPERIENCE_COMPANY_KEYS, _EXPERIENCE_DURATION_KEYS],
    )


def _format_education_entry(item):
    return _format_dict_entry(
        item,
        [_EDUCATION_DEGREE_KEYS, _EDUCATION_INSTITUTION_KEYS, _EDUCATION_YEAR_KEYS],
    )


def _format_entries(entries, formatter):
    """Formats a list of experience/education dicts (or plain strings)
    into one Paragraph-ready string, each entry separated by a blank line."""
    lines = []
    for item in (entries or []):
        formatted = formatter(item)
        if formatted:
            lines.append(formatted)
    return "<br/><br/>".join(lines) if lines else "Not specified"


def _score_cell_color(score):
    if score is None:
        return colors.grey
    if score >= 75:
        return colors.HexColor("#22c55e")
    if score >= 50:
        return colors.HexColor("#b45309")
    return colors.HexColor("#dc2626")


def _build_header(styles):
    generated_on = datetime.now().strftime("%d %B %Y, %I:%M %p")
    return [
        Paragraph("AI Job Search Agent — Match Report", styles["ReportTitle"]),
        Paragraph(f"Generated on {generated_on}", styles["ReportSubtitle"]),
    ]


def _build_preferences_section(questionnaire_answers, styles):
    qa = questionnaire_answers or {}
    rows = [
        ["Preferred Location", _safe(qa.get("search_location") or "No preference")],
        ["Work Mode", _safe(qa.get("work_mode", "No preference"))],
        ["Expected CTC Range", _safe(qa.get("expected_ctc_range", "No preference"))],
        ["Expected CTC (exact)", (
            # Reportlab's built-in Helvetica font has no rupee glyph -
            # it renders as a solid black box - so "Rs." is used
            # instead of the rupee sign here.
            f"Rs. {_safe(qa.get('expected_ctc_exact'))} LPA"
            if qa.get("expected_ctc_exact") else "Not specified"
        )],
        ["Preferred Companies", _join_or_none_listed(qa.get("preferred_companies"))],
        ["Notice Period", _safe(qa.get("notice_period", "Not specified"))],
        ["Jobs Scored with Gemini", _safe(qa.get("max_jobs", "N/A"))],
    ]
    return KeepTogether([
        Paragraph("Your Search Preferences", styles["SectionHeading"]),
        _kv_table(rows, styles),
    ])


def _build_profile_section(candidate_profile, styles):
    profile = candidate_profile or {}
    rows = [
        ["Possible Roles", _join_or_none_listed(profile.get("possible_roles"))],
        ["Key Skills", _join_or_none_listed(profile.get("skills"))],
        ["Experience", _format_entries(profile.get("experience"), _format_experience_entry)],
        ["Location (from resume)", _safe(profile.get("location") or "Not specified")],
        ["Education", _format_entries(profile.get("education"), _format_education_entry)],
    ]
    return KeepTogether([
        Paragraph("Candidate Profile Summary", styles["SectionHeading"]),
        _kv_table(rows, styles),
    ])


def _build_overview_table(scored_jobs, styles):
    header = ["Title", "Company", "Score", "Recommendation", "Source"]
    data = [header]
    for job in scored_jobs:
        score = job.get("final_score")
        data.append([
            Paragraph(_safe(job.get("title", "N/A")), styles["CellValueSmall"]),
            Paragraph(_safe(job.get("company") or "Company not listed"), styles["CellValueSmall"]),
            Paragraph(_safe(score) if score is not None else "N/A", styles["CellValueSmall"]),
            Paragraph(_safe(job.get("recommendation", "")), styles["CellValueSmall"]),
            Paragraph(_safe(job.get("source", "")), styles["CellValueSmall"]),
        ])

    # Column widths add up to the full usable page width on A4
    # (21cm page - 1.5cm left/right margins = 18cm) instead of leaving
    # unused space, and Recommendation gets more room since it's the
    # longest fixed label ("Possible Match").
    table = Table(data, colWidths=[5.6 * cm, 4.2 * cm, 1.7 * cm, 4.0 * cm, 2.5 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("ALIGN", (4, 0), (4, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
    ]))
    return [
        Paragraph(f"Results Overview — {len(scored_jobs)} jobs ranked", styles["SectionHeading"]),
        table,
    ]


def _build_job_header_block(job, styles):
    """
    The compact, bounded part of a job's detail: title, score line,
    "why this score", and the category table. Kept as one unit so
    KeepTogether can move it as a whole - this is what stops the
    category table from starting at the very bottom of a page.
    """
    flow = []

    score = job.get("final_score")
    title_line = f"{_safe(job.get('title', 'N/A'))} — {_safe(job.get('company') or 'Company not listed')}"
    flow.append(Paragraph(title_line, styles["JobHeading"]))
    flow.append(Paragraph(
        f"<b>Score:</b> {_safe(score) if score is not None else 'N/A'}/100 &nbsp;&nbsp; "
        f"<b>Recommendation:</b> {_safe(job.get('recommendation', 'N/A'))}",
        styles["Normal"],
    ))

    reason = job.get("reason") or "No reason provided."
    flow.append(Paragraph(f"<b>Why this score:</b> {_safe(reason)}", styles["Normal"]))
    flow.append(Spacer(1, 4))

    # Category breakdown table
    category_scores = job.get("category_scores", {}) or {}
    cat_rows = [
        [Paragraph("Category", styles["CellLabel"]), Paragraph("Score", styles["CellLabel"])]
    ]
    for key, label in CATEGORY_LABELS.items():
        cat_score = category_scores.get(key)
        cat_rows.append([
            Paragraph(_safe(label), styles["CellValue"]),
            Paragraph(f"{cat_score}/100" if cat_score is not None else "Not available", styles["CellValue"]),
        ])

    cat_table = Table(cat_rows, colWidths=[8 * cm, 4 * cm])
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]
    for i, key in enumerate(CATEGORY_LABELS.keys(), start=1):
        style_commands.append(
            ("TEXTCOLOR", (1, i), (1, i), _score_cell_color(category_scores.get(key)))
        )
    cat_table.setStyle(TableStyle(style_commands))
    flow.append(cat_table)

    return KeepTogether(flow)


def _build_job_detail(job, styles):
    """
    Full scoring breakdown for one job. The compact header block
    (title/score/reason/category table) is kept together as a unit;
    the more open-ended content below it (skill lists, roadmap, red
    flags) flows normally, since bullet lists are fine to continue
    across a page break.
    """
    flow = [_build_job_header_block(job, styles), Spacer(1, 4)]

    # Matching / missing skills
    flow.append(Paragraph(
        f"<b>Matching Skills:</b> {_join_or_none_listed(job.get('matching_skills'))}",
        styles["Normal"],
    ))
    flow.append(Paragraph(
        f"<b>Missing Skills:</b> {_join_or_none_listed(job.get('missing_skills'))}",
        styles["Normal"],
    ))

    # Skill gap roadmap (imported lazily to avoid a circular import,
    # since skill_gap_roadmap has no dependency on this module)
    from .skill_gap_roadmap import build_skill_roadmap
    roadmap = build_skill_roadmap(job.get("missing_skills", []))
    if roadmap:
        flow.append(Spacer(1, 2))
        flow.append(Paragraph("<b>How to close these skill gaps:</b>", styles["Normal"]))
        for item in roadmap:
            flow.append(Paragraph(
                f"\u2022 {_safe(item['skill'])} \u2014 {_safe(item['resource'])} (~{_safe(item['estimated_time'])})",
                styles["Normal"],
            ))

    # Red flags / risk status
    red_flags = job.get("red_flags") or []
    if red_flags:
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(
            "<b>Risk Status: HIGH</b> — " + "; ".join(_safe(flag) for flag in red_flags),
            ParagraphStyle(name="RiskHigh", parent=styles["Normal"], textColor=colors.HexColor("#dc2626")),
        ))
        # Same "verify before you invest time" links as the app UI -
        # a reader working from a printed/saved PDF gets the same
        # one-click way to check the company.
        company = (job.get("company") or "").strip()
        if company:
            google_url = f"https://www.google.com/search?q={quote_plus(company + ' reviews')}"
            linkedin_url = f"https://www.linkedin.com/search/results/companies/?keywords={quote_plus(company)}"
            flow.append(Paragraph(
                f"Verify <b>{_safe(company)}</b>: "
                f"<a href='{_safe(google_url)}' color='blue'>Search reviews</a> &nbsp;|&nbsp; "
                f"<a href='{_safe(linkedin_url)}' color='blue'>Find on LinkedIn</a>",
                styles["Normal"],
            ))
    else:
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(
            "<b>Risk Status: CLEAR</b> — no red flags detected.",
            ParagraphStyle(name="RiskClear", parent=styles["Normal"], textColor=colors.HexColor("#16a34a")),
        ))

    if job.get("url"):
        flow.append(Paragraph(f"<a href='{_safe(job['url'])}' color='blue'>View job posting</a>", styles["Normal"]))

    flow.append(Spacer(1, 6))
    flow.append(HRFlowable(width="100%", color=colors.lightgrey))

    return flow


def generate_pdf_report(candidate_profile, questionnaire_answers, scored_jobs):
    """
    Builds the full PDF report and returns it as bytes, ready to be
    handed to st.download_button (no temp file needed).

    Args:
        candidate_profile: dict from resume_parser.extract_candidate_profile()
        questionnaire_answers: dict from st.session_state.questionnaire_data
        scored_jobs: list of job dicts, each already containing
                     final_score/category_scores/etc from calculate_final_score()

    Returns:
        bytes - the complete PDF file content
    """
    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
    )

    story = []
    story.extend(_build_header(styles))

    story.append(_build_preferences_section(questionnaire_answers, styles))
    story.append(_build_profile_section(candidate_profile, styles))

    sorted_jobs = sorted(
        scored_jobs, key=lambda job: job.get("final_score") or 0, reverse=True,
    )

    story.extend(_build_overview_table(sorted_jobs, styles))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Detailed Job Breakdown", styles["SectionHeading"]))

    for job in sorted_jobs:
        story.extend(_build_job_detail(job, styles))

    doc.build(story)
    return buffer.getvalue()
 
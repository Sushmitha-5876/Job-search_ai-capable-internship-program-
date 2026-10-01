"""
ats_scoring.py
==============
Deterministic ATS (Applicant Tracking System) readiness scoring.

WHY THIS IS SEPARATE FROM scorer.py / final_scoring.py:
Those modules answer "is this JOB a good fit for the candidate" - an
AI judgment call made per job listing, using Gemini. This module
answers a completely different, EARLIER question: "would this RESUME
survive being scanned by the software a company's careers page runs
before a human ever opens it?"

That's mostly a mechanical question, not a fit judgment - so, same
reasoning as skill_matching.SYNONYM_MAP and
skill_gap_roadmap.SKILL_RESOURCES, this is deterministic, rule-based
scoring with ZERO Gemini calls: no added latency/cost, and the same
resume always gets the same score.

HONEST LIMITATION: real ATS software also inspects things like
multi-column layouts, tables, and text boxes, which need real PDF
layout analysis this project doesn't attempt - faking a check for
that would violate this project's own "don't guess at data you don't
have" principle (see analytics.py's docstring for the same stance).
What IS checked here, reliably:
    - whether the PDF's text extracted cleanly without needing OCR
      (a strong, real signal - most ATS software cannot OCR at all)
    - whether standard resume section headings are present
    - whether contact info is present as plain text
    - resume length
    - keyword overlap against a pasted job description, reusing the
      same synonym-aware matching already used elsewhere in this
      project (skill_matching.find_matching_skills)
"""

import re

from src.skill_matching import find_matching_skills


# ==========================================================
# FORMATTING / PARSEABILITY CHECKS
# ==========================================================

# label -> phrases that count as that section being present, and how
# many points it's worth if found. Kept as one flat, ordered list (not
# separate dicts) so check order == priority order == summary order.
_SECTION_KEYWORDS = [
    ("Has an 'Experience' section", ["experience", "employment history", "work history"]),
    ("Has an 'Education' section", ["education", "academic background"]),
    ("Has a 'Skills' section", ["skills", "technical skills", "core competencies"]),
    ("Has a 'Projects' section", ["projects", "academic projects", "personal projects"]),
    ("Has a 'Summary' or 'Objective' section", ["summary", "objective", "profile"]),
]
_POINTS_PER_SECTION = 6  # 5 sections x 6 = 30

_POINTS_OCR = 40
_POINTS_EMAIL = 10
_POINTS_PHONE = 10
_POINTS_LENGTH = 10
# 40 + 30 + 10 + 10 + 10 = 100

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\d[\s.-]?){9,10}\d")

_MIN_WORDS = 150
_MAX_WORDS = 1200


def check_resume_formatting(resume_text, used_ocr=False):
    """
    Runs deterministic, rule-based parseability checks on already-
    extracted resume text.

    Args:
        resume_text: plain text, as returned by
                      resume_parser.extract_resume_text().
        used_ocr: True if extract_resume_text() had to fall back to
                  OCR for this resume (image-based/scanned PDF). Most
                  real ATS software cannot OCR at all, so this is
                  treated as the single strongest signal here.

    Returns:
        dict:
            {
                "score": int 0-100,
                "checks": [
                    {"label": str, "passed": bool, "detail": str},
                    ...
                ],
            }
        Checks are always returned in the same fixed order (OCR,
        then sections, then contact info, then length), regardless
        of pass/fail, so a UI can render them consistently.
    """
    if not resume_text or not resume_text.strip():
        return {
            "score": 0,
            "checks": [{
                "label": "Readable text",
                "passed": False,
                "detail": "No text could be extracted from this resume at all.",
            }],
        }

    text_lower = resume_text.lower()
    word_count = len(resume_text.split())

    checks = []
    score = 0

    ocr_passed = not used_ocr
    checks.append({
        "label": "Text extracts cleanly (no OCR needed)",
        "passed": ocr_passed,
        "detail": (
            "Your resume's text can be extracted directly, the way an ATS reads it."
            if ocr_passed else
            "This resume needed OCR to read - most ATS software can't OCR at all, "
            "so a real ATS may see a blank or unreadable resume."
        ),
    })
    if ocr_passed:
        score += _POINTS_OCR

    for label, phrases in _SECTION_KEYWORDS:
        found = any(phrase in text_lower for phrase in phrases)
        section_name = label.split("'")[1]
        checks.append({
            "label": label,
            "passed": found,
            "detail": (
                f"Found a recognizable {section_name.lower()} heading."
                if found else
                f"No '{section_name}' heading found - ATS systems often look for this exact section name."
            ),
        })
        if found:
            score += _POINTS_PER_SECTION

    has_email = bool(_EMAIL_PATTERN.search(resume_text))
    checks.append({
        "label": "Email address in plain text",
        "passed": has_email,
        "detail": (
            "Found an email address."
            if has_email else
            "No email address detected as plain text - if it's inside an image or icon, ATS software can't read it."
        ),
    })
    if has_email:
        score += _POINTS_EMAIL

    has_phone = bool(_PHONE_PATTERN.search(resume_text))
    checks.append({
        "label": "Phone number in plain text",
        "passed": has_phone,
        "detail": "Found a phone number." if has_phone else "No phone number detected as plain text.",
    })
    if has_phone:
        score += _POINTS_PHONE

    length_ok = _MIN_WORDS <= word_count <= _MAX_WORDS
    if length_ok:
        length_detail = f"{word_count} words - a healthy range for a 1-2 page resume."
    elif word_count < _MIN_WORDS:
        length_detail = f"{word_count} words - this looks unusually short; an ATS may find very little to match against."
    else:
        length_detail = f"{word_count} words - this is quite long; consider trimming to the most relevant, recent content."
    checks.append({"label": "Reasonable resume length", "passed": length_ok, "detail": length_detail})
    if length_ok:
        score += _POINTS_LENGTH

    return {"score": score, "checks": checks}


# ==========================================================
# KEYWORD OVERLAP (against a pasted job description)
# ==========================================================

_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "will",
    "this", "that", "have", "has", "from", "our", "their", "they", "who",
    "what", "when", "where", "why", "how", "all", "any", "can", "into",
    "about", "than", "then", "them", "these", "those", "such", "some",
    "each", "other", "more", "most", "over", "also", "job", "role", "work",
    "team", "company", "years", "year", "experience", "candidate", "ability",
    "including", "using", "used", "strong", "good", "excellent", "must",
    "should", "would", "could", "please", "looking", "responsibilities",
    "requirements", "preferred", "required", "apply", "opportunity", "etc",
    "including", "within", "across", "per", "new", "join", "help", "make",
    "well", "high", "level", "based", "including", "one", "two", "three",
})

_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.]{2,}")


def _extract_keywords(job_description, max_keywords=25):
    """
    Pulls out the most-frequent, non-generic tokens from a job
    description - a deliberately simple keyword extraction (word
    tokens only, no phrase detection), matching the beginner-scope
    intent of this check: a plain keyword-overlap score, not a
    semantic analysis.
    """
    tokens = _TOKEN_PATTERN.findall(job_description.lower())

    counts = {}
    order = []
    for token in tokens:
        token = token.strip(".")
        if len(token) < 3 or token in _STOPWORDS:
            continue
        if token not in counts:
            order.append(token)
        counts[token] = counts.get(token, 0) + 1

    ranked = sorted(order, key=lambda t: (-counts[t], order.index(t)))
    return ranked[:max_keywords]


def calculate_keyword_overlap(resume_text, job_description):
    """
    Args:
        resume_text: plain resume text.
        job_description: plain text of a job posting, pasted by the user.

    Returns:
        dict:
            {
                "score": int 0-100 or None,
                "matched_keywords": list[str],
                "missing_keywords": list[str],
            }
        score is None (never 0) when no job_description was given, or
        when no usable keywords could be extracted from it - "no
        keywords found" and "no job description given" are different
        situations and shouldn't render as the same score.
    """
    if not job_description or not job_description.strip():
        return {"score": None, "matched_keywords": [], "missing_keywords": []}

    keywords = _extract_keywords(job_description)
    if not keywords:
        return {"score": None, "matched_keywords": [], "missing_keywords": []}

    matched = find_matching_skills(keywords, resume_text or "")
    matched_lower = {m.lower() for m in matched}
    missing = [k for k in keywords if k.lower() not in matched_lower]

    score = round(100 * len(matched) / len(keywords))
    return {"score": score, "matched_keywords": matched, "missing_keywords": missing}


# ==========================================================
# COMBINED SCORE
# ==========================================================

def get_ats_verdict(score):
    """Convert a numerical ATS score into a short, plain-English verdict."""
    if score is None:
        return "Unknown"
    if score >= 80:
        return "ATS-friendly"
    if score >= 60:
        return "Needs minor fixes"
    if score >= 40:
        return "At risk of being filtered out"
    return "High risk of being missed by ATS software"


def _build_summary(formatting, keyword):
    """
    Up to 3 short, human-readable takeaways: the highest-priority
    failed formatting checks first (already in priority order - OCR,
    then sections, then contact info, then length), then a keyword
    takeaway if a job description was provided.
    """
    bullets = []
    for check in formatting["checks"]:
        if not check["passed"]:
            bullets.append(check["detail"])
        if len(bullets) >= 2:
            break

    if keyword["score"] is not None:
        if keyword["missing_keywords"]:
            shown = ", ".join(keyword["missing_keywords"][:5])
            bullets.append(
                f"Missing {len(keyword['missing_keywords'])} keyword(s) from the job "
                f"description, including: {shown}."
            )
        else:
            bullets.append("Your resume contains every keyword extracted from the job description.")

    if not bullets:
        bullets.append("No major ATS red flags found in this resume.")

    return bullets


def calculate_ats_score(resume_text, used_ocr=False, job_description=""):
    """
    Combines the formatting checks and keyword-overlap check into one
    ATS readiness result. This is the main entry point other modules
    (the Streamlit app) should call.

    Args:
        resume_text: plain resume text, from resume_parser.extract_resume_text().
        used_ocr: whether OCR was needed to read the resume (see
                  check_resume_formatting()'s docstring).
        job_description: optional plain text of a target job posting.
                  If omitted, the overall score is formatting-only.

    Returns:
        dict:
            {
                "overall_score": int 0-100,
                "formatting_score": int 0-100,
                "keyword_score": int or None,
                "verdict": str,
                "checks": [...],                 # from check_resume_formatting()
                "matched_keywords": [...],
                "missing_keywords": [...],
                "summary": [str, ...],
            }
    """
    formatting = check_resume_formatting(resume_text, used_ocr=used_ocr)
    keyword = calculate_keyword_overlap(resume_text, job_description)

    if keyword["score"] is None:
        overall = formatting["score"]
    else:
        overall = round(0.5 * formatting["score"] + 0.5 * keyword["score"])

    return {
        "overall_score": overall,
        "formatting_score": formatting["score"],
        "keyword_score": keyword["score"],
        "verdict": get_ats_verdict(overall),
        "checks": formatting["checks"],
        "matched_keywords": keyword["matched_keywords"],
        "missing_keywords": keyword["missing_keywords"],
        "summary": _build_summary(formatting, keyword),
    }
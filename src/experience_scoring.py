"""
experience_scoring.py
======================
Deterministic (Python-only, no Gemini) experience-gap scoring.

Two independent things happen here:
    1. extract_candidate_years()      - how many years of experience
                                         does the CANDIDATE actually have?
    2. estimate_required_years()      - how many years does the JOB
                                         likely require, based on title?

Then score_experience_gap() compares the two and returns a 0-100 score.

WHY THIS IS SEPARATE FROM GEMINI: years of experience is a number you
can extract from text with regex — asking an LLM to "judge" this adds
hallucination risk for zero benefit, and makes results non-reproducible
(same resume could get slightly different experience scores on
different runs). This keeps that category 100% deterministic.

IMPORTANT: if the candidate's years can't be determined from the resume
text (no explicit mention, no parseable date ranges), extract_candidate_years()
returns None — NOT a guessed number. Callers must handle None as
"insufficient data," not silently default to some value. This mirrors
the same "don't fake missing data" principle used for salary.
"""

import re
from datetime import datetime


# ==========================================================
# CANDIDATE YEARS EXTRACTION
# ==========================================================

MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# Matches explicit statements like "5 years of experience" or "5+ years"
_EXPLICIT_YEARS_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)

# Matches date ranges like "Jan 2020 - Present", "2019 - 2021", "06/2020-12/2022"
_DATE_RANGE_PATTERN = re.compile(
    r"""
    (?P<start>
        (?:[A-Za-z]{3,9}\s+\d{4})      # "January 2020" / "Jan 2020"
        | (?:\d{1,2}/\d{4})            # "06/2020"
        | (?:\d{4})                    # "2020"
    )
    \s*(?:-|–|—|to)\s*
    (?P<end>
        (?:[A-Za-z]{3,9}\s+\d{4})
        | (?:\d{1,2}/\d{4})
        | (?:\d{4})
        | (?:present|current|now)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_single_date(text, is_end=False):
    """Parses one side of a date range into a (year, month) tuple, or None."""
    text = text.strip().lower()

    if is_end and text in {"present", "current", "now"}:
        today = datetime.today()
        return (today.year, today.month)

    # "06/2020"
    slash_match = re.match(r"(\d{1,2})/(\d{4})", text)
    if slash_match:
        return (int(slash_match.group(2)), int(slash_match.group(1)))

    # "January 2020" / "Jan 2020"
    month_match = re.match(r"([a-z]{3,9})\s+(\d{4})", text)
    if month_match:
        month_name, year = month_match.groups()
        month_num = MONTH_MAP.get(month_name[:3] if month_name not in MONTH_MAP else month_name)
        if month_num is None:
            # fall back to first 3 letters lookup already attempted above;
            # try full name match one more time defensively
            month_num = MONTH_MAP.get(month_name)
        if month_num:
            return (int(year), month_num)

    # bare "2020"
    year_match = re.match(r"^(\d{4})$", text)
    if year_match:
        return (int(year_match.group(1)), 1)  # assume January if only year given

    return None


def _years_between(start, end):
    """start/end are (year, month) tuples. Returns fractional years, or None if invalid."""
    if not start or not end:
        return None
    months = (end[0] - start[0]) * 12 + (end[1] - start[1])
    if months < 0:
        return None
    return round(months / 12, 1)


def extract_candidate_years(experience_entries):
    """
    Determines the candidate's total years of experience from their
    resume's experience section.

    Strategy (in order of reliability):
        1. Look for an EXPLICIT statement like "5+ years of experience"
           across all entries — most reliable, resumes often state this directly.
        2. Otherwise, parse date ranges (e.g. "Jan 2020 - Present") and
           sum the durations.
        3. If neither works, return None — do NOT guess.

    Args:
        experience_entries: list of strings (from candidate_profile["experience"])

    Returns:
        float years, or None if it genuinely can't be determined.
    """
    if not experience_entries:
        return None

    combined_text = " ".join(str(entry) for entry in experience_entries)

    # Strategy 1: explicit "N years" mention — take the MAX mentioned,
    # since resumes sometimes mention smaller numbers for individual
    # roles and a larger total separately.
    explicit_matches = _EXPLICIT_YEARS_PATTERN.findall(combined_text)
    if explicit_matches:
        try:
            values = [float(m) for m in explicit_matches]
            # Sanity check: ignore absurd values (e.g. "24/7" false-matching as "7 years")
            plausible = [v for v in values if 0 < v <= 50]
            if plausible:
                return max(plausible)
        except ValueError:
            pass

    # Strategy 2: sum durations from parseable date ranges
    total_months = 0
    found_any = False

    for entry in experience_entries:
        for match in _DATE_RANGE_PATTERN.finditer(str(entry)):
            start = _parse_single_date(match.group("start"), is_end=False)
            end = _parse_single_date(match.group("end"), is_end=True)
            years = _years_between(start, end)
            if years is not None:
                total_months += years * 12
                found_any = True

    if found_any and total_months > 0:
        return round(total_months / 12, 1)

    return None


# ==========================================================
# JOB'S REQUIRED YEARS ESTIMATION (from title keywords)
# ==========================================================

_SENIORITY_KEYWORDS = [
    (["principal", "staff", "distinguished", "architect"], 8),
    (["senior", "sr.", "sr ", "lead"], 5),
    (["mid-level", "mid level", " ii ", " ii)", "intermediate"], 3),
    (["junior", "jr.", "jr ", "entry level", "entry-level", "associate",
      "graduate", "fresher", "trainee", "intern"], 1),
]

# Matches an EXPLICIT requirement stated in a job description, e.g.
# "3-5 years of experience required", "minimum 2 years experience".
# Deliberately requires "experience"/"exp" to immediately follow the
# years mention — unlike the looser candidate-side pattern above, a
# scraped job description contains many unrelated numbers ("5 team
# members", "24/7 support", "3 office locations"), so this pattern is
# stricter on purpose to avoid false-positive requirement extraction.
_REQUIRED_YEARS_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*(?:-\s*\d+(?:\.\d+)?\+?\s*)?"
    r"(?:years?|yrs?)\s*(?:of\s+)?"
    r"(?:relevant\s+|professional\s+|work\s+|industry\s+)?"
    r"(?:experience|exp\.?)\b",
    re.IGNORECASE,
)


def _extract_required_years_from_description(description):
    """
    Looks for an explicit, employer-stated years-of-experience
    requirement in the job description text. Returns the LOWER bound
    when a range is given (e.g. "3-5 years" -> 3.0), since that's the
    actual minimum bar to clear. Returns None if nothing matches —
    caller falls back to the title-keyword heuristic.
    """
    if not description:
        return None

    match = _REQUIRED_YEARS_PATTERN.search(str(description))
    if not match:
        return None

    try:
        years = float(match.group(1))
    except (TypeError, ValueError):
        return None

    if 0 < years <= 20:  # sanity bound - ignore implausible values
        return years

    return None


def estimate_required_years(job_title, job_description=""):
    """
    Estimates the years of experience a job likely requires.

    Priority:
        1. An EXPLICIT requirement stated in the job description
           (e.g. "3+ years of experience required") - this is the
           employer's own stated number, so it's used whenever found.
        2. Otherwise, a heuristic based on title keywords (Senior,
           Junior, etc.) - a reasonable default when the description
           doesn't state years explicitly.
        3. Otherwise, defaults to 2.0 (a typical unspecified
           "Software Engineer" baseline).

    Args:
        job_title: string
        job_description: string, optional - pass job.get("description", "")
                          to get the more accurate description-based estimate.

    Returns:
        float estimated required years.
    """
    from_description = _extract_required_years_from_description(job_description)
    if from_description is not None:
        return from_description

    title = (job_title or "").lower()

    for keywords, years in _SENIORITY_KEYWORDS:
        if any(keyword in title for keyword in keywords):
            return float(years)

    return 2.0  # no stated requirement and no seniority keyword found


# ==========================================================
# GAP SCORING
# ==========================================================

def score_experience_gap(candidate_years, required_years):
    """
    Scores how well candidate_years matches required_years.

    Uses a continuous linear falloff instead of hard buckets, so two
    candidates with slightly different gaps (e.g. 1.1 years vs 1.9
    years) get slightly different scores instead of landing on the
    exact same number just because they fell in the same bucket.
    Floors at 20 so any determinable gap still contributes something
    to the average rather than going to zero.

    Returns:
        int score 0-100, or None if candidate_years is None (insufficient
        data — caller should treat this as N/A, not a bad score).
    """
    if candidate_years is None:
        return None

    gap = abs(candidate_years - required_years)

    # Special case: a senior role (5+ required) matched against someone
    # with little to no experience is a much bigger mismatch than the
    # numeric gap alone suggests — cap it low regardless of exact gap.
    if required_years >= 5 and candidate_years <= 1:
        return 5

    # Continuous falloff: gap 0 -> 100, gap 1 -> 80, gap 2 -> 60, ...
    # same anchor points the old bucket table used, but everything
    # between them is now smooth instead of a step function.
    score = max(20, round(100 - gap * 20))
    return score


def calculate_experience_score(candidate_profile, job):
    """
    Convenience wrapper: takes the full candidate_profile and job dict,
    extracts what's needed, and returns (score, candidate_years, required_years).

    score is None if candidate_years couldn't be determined — this is
    the signal that this category should be excluded from the weighted
    average and its weight redistributed (handled in Step 4).
    """
    candidate_years = extract_candidate_years(candidate_profile.get("experience", []))
    required_years = estimate_required_years(job.get("title", ""), job.get("description", ""))
    score = score_experience_gap(candidate_years, required_years)
    return score, candidate_years, required_years
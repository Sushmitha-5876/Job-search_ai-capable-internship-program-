import re


INDIA_FOCUSED_SOURCES = {"adzuna", "jobspipe"}

# --------------------------------------------------------------
# CTC range support
#
# WHY: app.py now collects a range (e.g. "15-20") instead of, or in
# addition to, one exact number. This maps each range label to a
# (low, high) tuple in Lakhs so we can turn it into a single number
# to score against. "50+" has no upper bound, so high is None.
# --------------------------------------------------------------

CTC_RANGES = {
    "0-5": (0, 5),
    "5-10": (5, 10),
    "10-15": (10, 15),
    "15-20": (15, 20),
    "20-30": (20, 30),
    "30-50": (30, 50),
    "50+": (50, None),
}


def _resolve_expected_ctc(expected_ctc_range, expected_ctc_exact):
    """
    Turns whatever the candidate gave us (an exact number, a range,
    or nothing) into one plain number in ₹ Lakhs to score against.

    Priority: exact number wins if given, since it's more precise.
    Otherwise fall back to the midpoint of the chosen range.
    Returns None if the candidate gave no preference at all.
    """
    if expected_ctc_exact and expected_ctc_exact > 0:
        return expected_ctc_exact

    bounds = CTC_RANGES.get(expected_ctc_range)
    if not bounds:
        return None  # "No preference" or an unrecognized value

    low, high = bounds
    if high is None:
        # Open-ended range like "50+" — use the floor as a fair estimate
        return low
    return (low + high) / 2


def _extract_job_salary_range(job):
    """
    Extract salary in INR from structured fields or job text.

    Supported examples:
    ₹8-20 LPA
    8-20 LPA
    8 LPA
    """

    raw = job.get("raw", {}) if isinstance(job.get("raw"), dict) else {}

    # First try structured salary fields
    min_salary = (
        job.get("_salary_min")
        or raw.get("salary_min")
        or raw.get("min_salary")
        or raw.get("min_annual_salary")
    )

    max_salary = (
        job.get("_salary_max")
        or raw.get("salary_max")
        or raw.get("max_salary")
        or raw.get("max_annual_salary")
    )

    def _to_float(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    min_salary = _to_float(min_salary)
    max_salary = _to_float(max_salary)

    if min_salary is not None or max_salary is not None:
        return min_salary, max_salary

    # If structured salary is unavailable, search job text.
    text = " ".join(
        str(job.get(field, "") or "")
        for field in ["title", "description"]
    )

    # Match ranges such as:
    # ₹8-20 LPA
    # 8-20 LPA
    # 8 - 20 LPA
    range_match = re.search(
        r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*[-–]\s*"
        r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?)\b",
        text,
        flags=re.IGNORECASE,
    )

    if range_match:
        min_lpa = float(range_match.group(1))
        max_lpa = float(range_match.group(2))

        return min_lpa * 100_000, max_lpa * 100_000

    # Match a single salary such as:
    # ₹8 LPA
    # 8 LPA
    single_match = re.search(
        r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?)\b",
        text,
        flags=re.IGNORECASE,
    )

    if single_match:
        salary_lpa = float(single_match.group(1))
        salary_rupees = salary_lpa * 100_000

        return salary_rupees, salary_rupees

    return None, None


def calculate_salary_score(job, expected_ctc_range="No preference", expected_ctc_exact=0.0):
    expected_ctc_lakhs = _resolve_expected_ctc(expected_ctc_range, expected_ctc_exact)

    if not expected_ctc_lakhs or expected_ctc_lakhs <= 0:
        return None

    if job.get("source") not in INDIA_FOCUSED_SOURCES:
        return None

    min_salary, max_salary = _extract_job_salary_range(job)

    if min_salary is None and max_salary is None:
        return None

    job_salary = max_salary if max_salary is not None else min_salary

    if min_salary is not None and max_salary is not None:
        job_salary = (min_salary + max_salary) / 2

    if not job_salary or job_salary <= 0:
        return None

    expected_ctc_rupees = expected_ctc_lakhs * 100_000

    ratio = job_salary / expected_ctc_rupees

    if ratio >= 1.0:
        return 100
    if ratio >= 0.85:
        return 80
    if ratio >= 0.70:
        return 60
    if ratio >= 0.50:
        return 30

    return 10
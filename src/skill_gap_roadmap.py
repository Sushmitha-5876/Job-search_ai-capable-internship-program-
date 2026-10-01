"""
skill_gap_roadmap.py
=====================
Turns a job's missing_skills (already produced by Gemini in
scorer.score_job_with_gemini) into a short, actionable learning
suggestion per skill.

WHY THIS IS A STATIC MAP, NOT A GEMINI CALL: the missing_skills list
is almost always drawn from a well-known set of technical skills
(AWS, Docker, React, SQL, ...) that job postings actually ask for.
A curated map means zero extra Gemini calls (no added latency/cost
per job) and fully reproducible suggestions — the same missing skill
always gets the same suggestion. This mirrors the same reasoning
already used for SYNONYM_MAP in skill_matching.py: a maintained list
beats an LLM guess for a bounded, well-known vocabulary.

For a skill not in the map, a clear generic fallback is returned
instead of nothing — see build_skill_roadmap()'s docstring.

Uses the same normalization + fuzzy-fallback approach as
skill_matching.py (RapidFuzz), so "React.js" or "reactjs" still
matches the "react" entry below without needing every spelling
variant listed explicitly.
"""

from rapidfuzz import fuzz

from .skill_matching import SYNONYM_MAP


# Each entry: normalized skill name -> (suggested resource, rough time estimate)
# Keep this alphabetized and easy to extend — same maintenance pattern
# as SYNONYM_MAP in skill_matching.py.
SKILL_RESOURCES = {
    "amazon web services": ("AWS Cloud Practitioner Essentials (AWS Skill Builder, free)", "1-2 weeks"),
    "angular": ("Angular's official 'First App' tutorial + a small CRUD project", "2 weeks"),
    "aws": ("AWS Cloud Practitioner Essentials (AWS Skill Builder, free)", "1-2 weeks"),
    "azure": ("Microsoft Learn's Azure Fundamentals path (free)", "1-2 weeks"),
    "c++": ("learncpp.com free tutorial series", "3-4 weeks"),
    "c#": ("Microsoft Learn's C# fundamentals path (free)", "2-3 weeks"),
    "django": ("Official Django tutorial (build the polls app)", "1-2 weeks"),
    "docker": ("Docker's official 'Get Started' guide + containerize one of your own projects", "1 week"),
    "flask": ("Official Flask 'Quickstart' + build a small REST API", "1 week"),
    "git": ("'Learn Git Branching' (interactive, free)", "3-5 days"),
    "google cloud platform": ("Google Cloud Skills Boost 'Cloud Fundamentals' path (free tier)", "1-2 weeks"),
    "graphql": ("'How to GraphQL' free tutorial", "1 week"),
    "java": ("'Java Programming Masterclass' fundamentals or Oracle's own Java tutorials", "3-4 weeks"),
    "javascript": ("javascript.info (free, comprehensive)", "3-4 weeks"),
    "kubernetes": ("Kubernetes' official 'Learn Kubernetes Basics' interactive tutorial", "2 weeks"),
    "machine learning": ("Andrew Ng's Machine Learning Specialization (Coursera, audit free)", "4-6 weeks"),
    "mongodb": ("MongoDB University's free 'M001: MongoDB Basics' course", "1-2 weeks"),
    "node": ("Node.js official guides + build a small REST API", "2 weeks"),
    "postgresql": ("PostgreSQL Tutorial (postgresqltutorial.com, free)", "1-2 weeks"),
    "python": ("'Automate the Boring Stuff with Python' (free online)", "2-3 weeks"),
    "react": ("Official React 'Quick Start' docs + build one small project", "2 weeks"),
    "rest api": ("Build one small CRUD API in a framework you already know", "1 week"),
    "sql": ("Mode's free interactive SQL tutorial", "1-2 weeks"),
    "typescript": ("Official TypeScript Handbook (free)", "1 week"),
    "vue": ("Official Vue.js 'Quick Start' guide + a small project", "2 weeks"),
}

_GENERIC_FALLBACK = "Search for a well-reviewed free tutorial or official documentation for this skill and build one small project with it"

_FUZZY_THRESHOLD = 85  # same conservative bar used in skill_matching.py


def _normalize(text):
    return str(text).strip().lower()


def _lookup_resource(skill):
    """
    Returns (resource, estimated_time) for a skill, checking:
        1. Exact match against SKILL_RESOURCES
        2. Fuzzy match (handles "React.js" vs "react", minor typos)
    Falls back to a generic suggestion if nothing matches closely.
    """
    skill_norm = _normalize(skill)

    if skill_norm in SKILL_RESOURCES:
        return SKILL_RESOURCES[skill_norm]

    best_match = None
    best_score = 0
    for known_skill in SKILL_RESOURCES:
        score = fuzz.partial_ratio(skill_norm, known_skill)
        if score > best_score:
            best_score = score
            best_match = known_skill

    if best_match and best_score >= _FUZZY_THRESHOLD:
        return SKILL_RESOURCES[best_match]

    return (_GENERIC_FALLBACK, "varies")


def build_skill_roadmap(missing_skills):
    """
    Args:
        missing_skills: list[str] — a job's missing_skills, exactly as
                         produced by scorer.score_job_with_gemini().

    Returns:
        list[dict] — one entry per skill, each:
            {"skill": str, "resource": str, "estimated_time": str}
        Same order as the input list. Empty list in -> empty list out.
    """
    if not missing_skills:
        return []

    roadmap = []
    for skill in missing_skills:
        resource, estimated_time = _lookup_resource(skill)
        roadmap.append({
            "skill": skill,
            "resource": resource,
            "estimated_time": estimated_time,
        })
    return roadmap


# ==========================================================
# RESUME IMPROVEMENT SUGGESTIONS (recurring gaps across ALL jobs)
# ==========================================================
# WHY THIS IS SEPARATE FROM build_skill_roadmap() ABOVE:
# build_skill_roadmap() explains the gaps for ONE job's card.
# This function looks across every job in the current shortlist and
# surfaces skills that show up as "missing" on MULTIPLE jobs — those
# are the gaps worth actually adding to your resume/learning plan,
# as opposed to a one-off requirement from a single unusual posting.
# Uses the same static SKILL_RESOURCES map, so no extra Gemini calls.

def _canonical_key(skill):
    """
    Maps a raw skill string to a canonical grouping key, so "React",
    "React.js", and "ReactJS" all count as the SAME recurring skill
    instead of three separate ones. Reuses the SYNONYM_MAP already
    maintained in skill_matching.py rather than keeping a second list.
    """
    skill_norm = _normalize(skill)
    for canonical, variants in SYNONYM_MAP.items():
        if skill_norm == canonical or skill_norm in variants:
            return canonical
    return skill_norm


def build_resume_improvement_suggestions(scored_jobs, min_occurrences=2, top_n=8):
    """
    Args:
        scored_jobs: list[dict] — the scored jobs already sitting in
                     st.session_state.scored_jobs, each with a
                     "missing_skills" list (exactly as produced by
                     scorer.score_job_with_gemini()).
        min_occurrences: a skill must be missing from at least this
                     many DIFFERENT jobs to count as "recurring"
                     (default 2 — a gap only one job flagged isn't
                     a pattern worth acting on).
        top_n: max number of suggestions returned, most-frequent first.

    Returns:
        list[dict], each:
            {"skill": str, "count": int, "resource": str, "estimated_time": str}
        Sorted by how many jobs asked for it (descending).
        Empty list if nothing recurs at least min_occurrences times,
        or if scored_jobs is empty.
    """
    if not scored_jobs:
        return []

    counts = {}        # canonical_key -> number of jobs missing it
    display_name = {}  # canonical_key -> first-seen original spelling

    for job in scored_jobs:
        missing = job.get("missing_skills") or []
        seen_in_this_job = set()  # avoid double-counting one job's own duplicates
        for skill in missing:
            key = _canonical_key(skill)
            if key in seen_in_this_job:
                continue
            seen_in_this_job.add(key)
            counts[key] = counts.get(key, 0) + 1
            display_name.setdefault(key, skill)

    recurring = [
        (key, count) for key, count in counts.items()
        if count >= min_occurrences
    ]
    recurring.sort(key=lambda pair: pair[1], reverse=True)

    suggestions = []
    for key, count in recurring[:top_n]:
        resource, estimated_time = _lookup_resource(key)
        suggestions.append({
            "skill": display_name[key],
            "count": count,
            "resource": resource,
            "estimated_time": estimated_time,
        })
    return suggestions
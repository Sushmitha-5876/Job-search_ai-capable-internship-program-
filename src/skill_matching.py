"""
skill_matching.py
==================
Shared skill-matching logic used by BOTH:
    - job_fetcher.py's is_relevant_job()   (decides whether to keep a job at all)
    - scorer.py's calculate_job_score()    (decides the pre-rank score)

WHY THIS EXISTS: previously, skill matching was exact-substring-only.
A candidate with "JavaScript" on their resume would NOT match a job
requiring "JS", and a candidate with "Node" would not match "Node.js".
This silently filtered out real matches before they ever reached
Gemini. This file fixes that with two layers:

    1. SYNONYM_MAP - known equivalent terms (JS <-> JavaScript, etc.)
       checked first because it's instant and 100% correct when it hits.
    2. RapidFuzz partial-ratio fuzzy matching - catches near-misses
       that aren't in the synonym map (typos, spacing differences,
       "ReactJS" vs "React JS" vs "React.js").

Both layers run on the SAME searchable text (title + description + tags)
that the existing code already builds — this file doesn't change what
text gets searched, only HOW a skill is judged to be present in it.
"""

import re

from rapidfuzz import fuzz


# ==========================================================
# SYNONYM MAP
# ==========================================================
# Each key maps to a list of equivalent terms. Matching is
# bidirectional: if the candidate has "javascript" and the job
# mentions "js", that counts as a match, and vice versa.

SYNONYM_MAP = {
    "javascript": ["js"],
    "typescript": ["ts"],
    "node": ["nodejs", "node.js"],
    "react": ["reactjs", "react.js"],
    "vue": ["vuejs", "vue.js"],
    "angular": ["angularjs", "angular.js"],
    "python": ["py"],
    "postgresql": ["postgres"],
    "mongodb": ["mongo"],
    "kubernetes": ["k8s"],
    "amazon web services": ["aws"],
    "google cloud platform": ["gcp"],
    "microsoft azure": ["azure"],
    "continuous integration": ["ci/cd", "ci cd", "cicd"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "natural language processing": ["nlp"],
    "object oriented programming": ["oop"],
    "restful api": ["rest api", "rest"],
    "user interface": ["ui"],
    "user experience": ["ux"],
    "structured query language": ["sql"],
    "c sharp": ["c#"],
    "c plus plus": ["c++"],
    "golang": ["go"],
    "docker": ["dockerized", "containerization"],
}


def _build_lookup():
    """
    Turns SYNONYM_MAP into a flat dict: every variant term (including
    the canonical key itself) maps to the FULL set of its equivalents.
    Built once at import time, not on every call.
    """
    lookup = {}
    for canonical, variants in SYNONYM_MAP.items():
        group = set(variants) | {canonical}
        for term in group:
            lookup[term] = group
    return lookup


_SYNONYM_LOOKUP = _build_lookup()


def get_synonym_group(skill):
    """
    Returns the set of terms considered equivalent to `skill`
    (including `skill` itself). If `skill` has no known synonyms,
    returns a set containing only itself.
    """
    skill = skill.strip().lower()
    return _SYNONYM_LOOKUP.get(skill, {skill})


def _word_boundary_search(term, text):
    """Exact match, but respecting word boundaries (so 'go' doesn't match inside 'going')."""
    if not term:
        return False
    pattern = rf"(?<![a-z0-9+#.]){re.escape(term)}(?![a-z0-9+#.])"
    return bool(re.search(pattern, text, re.IGNORECASE))


def skill_matches_text(skill, text, fuzzy_threshold=88):
    """
    Returns True if `skill` is present in `text`, checking:
        1. The skill itself (word-boundary exact match)
        2. Any known synonym of the skill (word-boundary exact match)
        3. A fuzzy partial-ratio match against the raw text, as a
           last resort for near-misses the above two don't catch.

    fuzzy_threshold: 0-100, RapidFuzz partial_ratio score required to
    count as a match. 88 is deliberately strict — this stage isn't
    meant to catch every possible near-miss, just avoid missing exact
    synonyms with slightly different formatting. Too low a threshold
    would start matching unrelated skills to each other.
    """
    if not skill or not text:
        return False

    text_lower = text.lower()

    # Layer 1 + 2: exact match against the skill and all its synonyms
    for variant in get_synonym_group(skill):
        if _word_boundary_search(variant, text_lower):
            return True

    # Layer 3: fuzzy match as a fallback for near-misses
    # (partial_ratio checks if `skill` closely matches ANY substring
    # of similar length within `text`, not just the whole text)
    skill_lower = skill.strip().lower()
    if len(skill_lower) >= 3:  # skip fuzzy matching for very short skills (too noisy)
        score = fuzz.partial_ratio(skill_lower, text_lower)
        if score >= fuzzy_threshold:
            return True

    return False


def find_matching_skills(candidate_skills, text, fuzzy_threshold=88):
    """
    Given a list of candidate skills and a block of searchable text,
    returns the subset of skills that are considered present in the
    text (via exact, synonym, or fuzzy matching).
    """
    if not candidate_skills or not text:
        return []

    return [
        skill for skill in candidate_skills
        if skill_matches_text(skill, text, fuzzy_threshold)
    ]


# ==========================================================
# ROLE MATCHING (used by scorer.py's pre-rank)
# ==========================================================
# WHY THIS EXISTS: calculate_job_score() previously checked role match
# with plain `if role in title` — a candidate with "Software Developer"
# would NOT match a job titled "Software Development Engineer", even
# though those are clearly the same role. This mirrors the same fix
# already applied to skills above: exact match first (fast, catches
# the common case), fuzzy as a fallback for reordered/expanded titles.
#
# NOTE: this does NOT catch abbreviations like "SDE" for "Software
# Development Engineer" — those share no common substring or token
# overlap, so fuzzy matching can't bridge that gap. That would need
# a synonym map (like SYNONYM_MAP above) if you want to handle it
# later; out of scope for this fix.

def role_matches_title(role, title, fuzzy_threshold=75):
    """
    Returns True if `role` (one of the candidate's possible_roles,
    e.g. "software developer") reasonably matches `title` (a job's
    title, e.g. "software development engineer").

    Checks:
        1. Word-boundary-free substring match (fast path — catches
           the common case where the role appears in the title as-is)
        2. RapidFuzz token_set_ratio as a fallback — this compares
           the SET of words in each string regardless of order or
           extra words, so "backend developer" reasonably matches
           "senior backend developer (java)" without needing an
           exact substring hit.
    """
    if not role or not title:
        return False

    role = role.strip().lower()
    title_lower = title.lower()

    if role in title_lower:
        return True

    score = fuzz.token_set_ratio(role, title_lower)
    return score >= fuzzy_threshold
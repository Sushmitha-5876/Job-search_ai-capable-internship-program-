"""
Tests for src/skill_matching.py

Covers the three matching layers: exact/word-boundary, synonym, and
fuzzy — plus the specific bug this module was built to fix (exact-only
matching silently dropping real skill matches like JS <-> JavaScript).
"""

from src.skill_matching import skill_matches_text, find_matching_skills


def test_exact_match():
    assert skill_matches_text("Python", "3 years of Python experience")


def test_synonym_match_js_to_javascript():
    # The candidate's resume says "JavaScript"; the job posting says "JS".
    assert skill_matches_text("JavaScript", "Looking for a JS developer")


def test_synonym_match_is_bidirectional():
    # And the reverse direction should also work.
    assert skill_matches_text("JS", "Strong JavaScript background required")


def test_word_boundary_prevents_false_positive():
    # "go" must not match inside "going" or "algorithm".
    assert not skill_matches_text("go", "We are going to review your algorithm")


def test_word_boundary_allows_real_standalone_match():
    assert skill_matches_text("go", "Experience with Go is a plus")


def test_fuzzy_match_catches_formatting_near_miss():
    # "ReactJS" vs "React JS" vs "React.js" - not in the synonym map,
    # so this must be caught by the fuzzy fallback layer.
    assert skill_matches_text("ReactJS", "3 years of React JS development")


def test_no_match_for_unrelated_skill():
    assert not skill_matches_text("Kubernetes", "Frontend role using React and CSS")


def test_find_matching_skills_filters_to_present_only():
    candidate_skills = ["Python", "Kubernetes", "SQL", "Rust"]
    job_text = "Backend role requiring Python and structured query language (SQL)."

    matched = find_matching_skills(candidate_skills, job_text)

    assert "Python" in matched
    assert "SQL" in matched
    assert "Kubernetes" not in matched
    assert "Rust" not in matched


def test_find_matching_skills_handles_empty_inputs():
    assert find_matching_skills([], "some text") == []
    assert find_matching_skills(["Python"], "") == []
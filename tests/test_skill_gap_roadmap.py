"""
Tests for src/skill_gap_roadmap.py
"""

from src.skill_gap_roadmap import build_skill_roadmap


def test_empty_missing_skills_returns_empty_list():
    assert build_skill_roadmap([]) == []
    assert build_skill_roadmap(None) == []


def test_known_skill_returns_its_curated_resource():
    roadmap = build_skill_roadmap(["AWS"])
    assert len(roadmap) == 1
    assert roadmap[0]["skill"] == "AWS"
    assert "AWS" in roadmap[0]["resource"] or "Cloud" in roadmap[0]["resource"]
    assert roadmap[0]["estimated_time"] != "varies"


def test_fuzzy_variant_still_matches_known_skill():
    # "React.js" isn't a key in SKILL_RESOURCES, but should fuzzy-match "react"
    roadmap = build_skill_roadmap(["React.js"])
    assert "react" in roadmap[0]["resource"].lower()


def test_unknown_skill_gets_generic_fallback():
    roadmap = build_skill_roadmap(["SomeVeryObscureNicheThing123"])
    assert roadmap[0]["estimated_time"] == "varies"
    assert "tutorial" in roadmap[0]["resource"].lower() or "documentation" in roadmap[0]["resource"].lower()


def test_preserves_order_and_count_for_multiple_skills():
    roadmap = build_skill_roadmap(["Python", "Docker", "AWS"])
    assert [item["skill"] for item in roadmap] == ["Python", "Docker", "AWS"]
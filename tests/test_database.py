"""
Tests for src/database.py

Uses a temporary database file per test (pytest's tmp_path fixture)
so tests never touch the real jobs.db the app actually uses.
"""

from src.database import (
    init_db,
    save_job,
    get_saved_jobs,
    update_job_status,
    delete_saved_job,
)


def _sample_job(title="Software Engineer", company="Acme", score=80, url="http://example.com/job1"):
    return {"title": title, "company": company, "final_score": score, "url": url}


def test_save_and_retrieve_job(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)

    saved = save_job(_sample_job(), db_path)
    assert saved is True

    jobs = get_saved_jobs(db_path)
    assert len(jobs) == 1
    assert jobs[0]["title"] == "Software Engineer"
    assert jobs[0]["status"] == "saved"


def test_saving_same_job_twice_is_rejected(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)

    first = save_job(_sample_job(), db_path)
    second = save_job(_sample_job(), db_path)

    assert first is True
    assert second is False
    assert len(get_saved_jobs(db_path)) == 1


def test_different_jobs_can_both_be_saved(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)

    save_job(_sample_job(title="Job A", url="http://example.com/a"), db_path)
    save_job(_sample_job(title="Job B", url="http://example.com/b"), db_path)

    assert len(get_saved_jobs(db_path)) == 2


def test_update_job_status(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)
    save_job(_sample_job(), db_path)

    job_id = get_saved_jobs(db_path)[0]["id"]
    update_job_status(job_id, "applied", db_path)

    jobs = get_saved_jobs(db_path)
    assert jobs[0]["status"] == "applied"
    assert jobs[0]["status_updated_at"] is not None


def test_delete_saved_job(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)
    save_job(_sample_job(), db_path)

    job_id = get_saved_jobs(db_path)[0]["id"]
    delete_saved_job(job_id, db_path)

    assert get_saved_jobs(db_path) == []

# ==========================================================
# Per-session (workspace) isolation
#
# On a hosted deployment every visitor shares one jobs.db, so each
# row is scoped to a session_id. These tests pin that guarantee.
# ==========================================================

import sqlite3

from src.database import (
    DEFAULT_SESSION_ID,
    new_session_id,
    log_search,
    get_recent_searches,
    delete_recent_search,
)


def test_saved_jobs_are_private_to_each_session(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)

    save_job(_sample_job(title="Alice's job"), db_path, session_id="alice")
    save_job(_sample_job(title="Bob's job", url="http://example.com/b"), db_path, session_id="bob")

    alice_jobs = get_saved_jobs(db_path, session_id="alice")
    bob_jobs = get_saved_jobs(db_path, session_id="bob")

    assert [j["title"] for j in alice_jobs] == ["Alice's job"]
    assert [j["title"] for j in bob_jobs] == ["Bob's job"]


def test_same_job_can_be_saved_by_two_different_sessions(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)

    first = save_job(_sample_job(), db_path, session_id="alice")
    second = save_job(_sample_job(), db_path, session_id="bob")
    duplicate = save_job(_sample_job(), db_path, session_id="alice")

    assert first is True
    assert second is True          # not a duplicate - different workspace
    assert duplicate is False      # duplicate inside the same workspace


def test_one_session_cannot_update_or_delete_another_sessions_job(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)
    save_job(_sample_job(), db_path, session_id="alice")
    alice_job_id = get_saved_jobs(db_path, session_id="alice")[0]["id"]

    # Bob guesses Alice's row id and tries to tamper with it.
    update_job_status(alice_job_id, "rejected", db_path, session_id="bob")
    delete_saved_job(alice_job_id, db_path, session_id="bob")

    alice_jobs = get_saved_jobs(db_path, session_id="alice")
    assert len(alice_jobs) == 1
    assert alice_jobs[0]["status"] == "saved"


def test_search_history_is_private_to_each_session(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)

    log_search("Alice search", db_path, session_id="alice")
    log_search("Bob search", db_path, session_id="bob")

    assert [s["summary"] for s in get_recent_searches(10, db_path, session_id="alice")] == ["Alice search"]
    assert [s["summary"] for s in get_recent_searches(10, db_path, session_id="bob")] == ["Bob search"]

    # Bob cannot delete Alice's history entry by id.
    alice_entry_id = get_recent_searches(10, db_path, session_id="alice")[0]["id"]
    delete_recent_search(alice_entry_id, db_path, session_id="bob")
    assert len(get_recent_searches(10, db_path, session_id="alice")) == 1


def test_omitting_session_id_uses_the_default_workspace(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    init_db(db_path)

    save_job(_sample_job(), db_path)

    assert len(get_saved_jobs(db_path)) == 1
    assert len(get_saved_jobs(db_path, session_id=DEFAULT_SESSION_ID)) == 1
    assert get_saved_jobs(db_path, session_id="someone-else") == []


def test_old_database_without_session_column_is_migrated(tmp_path):
    """A jobs.db created before session_id existed must keep its rows."""
    db_path = tmp_path / "old_jobs.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE saved_jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, company TEXT, "
        "final_score INTEGER, url TEXT, status TEXT DEFAULT 'saved', "
        "date_saved TEXT DEFAULT CURRENT_TIMESTAMP, status_updated_at TEXT)"
    )
    connection.execute("CREATE TABLE search_history (id INTEGER PRIMARY KEY AUTOINCREMENT, summary TEXT, "
                       "searched_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    connection.execute("INSERT INTO saved_jobs (title, company, url) VALUES ('Legacy', 'OldCo', 'http://old')")
    connection.execute("INSERT INTO search_history (summary) VALUES ('legacy search')")
    connection.commit()
    connection.close()

    init_db(db_path)

    assert [j["title"] for j in get_saved_jobs(db_path)] == ["Legacy"]
    assert [s["summary"] for s in get_recent_searches(5, db_path)] == ["legacy search"]
    assert get_saved_jobs(db_path, session_id="new-visitor") == []


def test_new_session_ids_are_unique_and_url_safe():
    ids = {new_session_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(i.isalnum() and len(i) == 16 for i in ids)
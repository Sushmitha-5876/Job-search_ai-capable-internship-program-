"""
database.py
============
A tiny SQLite layer for saving jobs the candidate wants to track.

This is the "database" piece Track A's grading rubric asks for
(Career Functionality section, Option A1: "Add SQLite database for
saving job applications and preferences").

Uses Python's built-in sqlite3 module - nothing new to install.
The database file (jobs.db) is created automatically the first time
the app runs, right next to app.py.

PER-SESSION ISOLATION
---------------------
On a hosted deployment (e.g. Streamlit Cloud) every visitor talks to
the SAME jobs.db file. Without isolation, one visitor would see - and
could edit or delete - another visitor's saved jobs and search history.

Every row therefore carries a `session_id` (a "workspace id"), and every
read/write function filters on it. app.py generates a random workspace
id per visitor and keeps it in the page URL, so a refresh keeps the same
workspace while different visitors never see each other's data.

Every function takes `session_id` as an optional argument that defaults
to DEFAULT_SESSION_ID, so local single-user use (and the existing tests)
keep working unchanged. Rows saved before this column existed are
migrated into the default workspace automatically.

Every function also accepts an optional db_path so tests can point at a
temporary, throwaway database instead of the real one.
"""

import sqlite3
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "jobs.db"

DEFAULT_SESSION_ID = "default"


def new_session_id():
    """Returns a fresh, unguessable workspace id (16 hex characters)."""
    return uuid.uuid4().hex[:16]


def _has_column(cursor, table, column):
    return any(row[1] == column for row in cursor.execute(f"PRAGMA table_info({table})").fetchall())


def init_db(db_path=DB_PATH):
    """
    Creates the saved_jobs and search_history tables if they don't
    already exist, and migrates older databases that pre-date the
    session_id column. Safe to call every time the app starts - does
    nothing if everything is already in place.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT,
            searched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT DEFAULT 'default'
        )
        """
    )
    if not _has_column(cursor, "search_history", "session_id"):
        cursor.execute("ALTER TABLE search_history ADD COLUMN session_id TEXT DEFAULT 'default'")
    cursor.execute("UPDATE search_history SET session_id = 'default' WHERE session_id IS NULL")

    connection.commit()
    connection.close()

    _init_saved_jobs_table(db_path)


def _init_saved_jobs_table(db_path=DB_PATH):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            company TEXT,
            final_score INTEGER,
            url TEXT,
            status TEXT DEFAULT 'saved',
            date_saved TEXT DEFAULT CURRENT_TIMESTAMP,
            status_updated_at TEXT,
            session_id TEXT DEFAULT 'default'
        )
        """
    )

    if not _has_column(cursor, "saved_jobs", "status_updated_at"):
        cursor.execute("ALTER TABLE saved_jobs ADD COLUMN status_updated_at TEXT")

    if not _has_column(cursor, "saved_jobs", "session_id"):
        cursor.execute("ALTER TABLE saved_jobs ADD COLUMN session_id TEXT DEFAULT 'default'")

    cursor.execute(
        "UPDATE saved_jobs SET status_updated_at = date_saved WHERE status_updated_at IS NULL"
    )
    cursor.execute("UPDATE saved_jobs SET session_id = 'default' WHERE session_id IS NULL")

    connection.commit()
    connection.close()


def save_job(job, db_path=DB_PATH, session_id=DEFAULT_SESSION_ID):
    """
    Saves one job into the given workspace.

    Args:
        job: a job dict (the same dict used everywhere else in the
             app) - reads 'title', 'company', 'final_score', 'url'.

    Returns:
        True if it saved successfully, False if this exact job
        (same title + company + url) was already saved in this workspace.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Don't save the same job twice (within this workspace).
    cursor.execute(
        "SELECT id FROM saved_jobs WHERE title = ? AND company = ? AND url = ? AND session_id = ?",
        (job.get("title", ""), job.get("company", ""), job.get("url", ""), session_id),
    )
    if cursor.fetchone():
        connection.close()
        return False

    cursor.execute(
        """
        INSERT INTO saved_jobs (title, company, final_score, url, status, session_id)
        VALUES (?, ?, ?, ?, 'saved', ?)
        """,
        (
            job.get("title", ""),
            job.get("company", ""),
            job.get("final_score"),
            job.get("url", ""),
            session_id,
        ),
    )

    connection.commit()
    connection.close()
    return True


def get_saved_jobs(db_path=DB_PATH, session_id=DEFAULT_SESSION_ID):
    """
    Returns every job saved in this workspace as a list of dicts,
    most recently saved first.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row  # lets us read columns by name
    cursor = connection.cursor()

    cursor.execute(
        "SELECT * FROM saved_jobs WHERE session_id = ? ORDER BY date_saved DESC, id DESC",
        (session_id,),
    )
    rows = cursor.fetchall()

    connection.close()
    return [dict(row) for row in rows]


def update_job_status(job_id, new_status, db_path=DB_PATH, session_id=DEFAULT_SESSION_ID):
    """
    Updates a saved job's status - e.g. 'saved' -> 'applied' -> 'interview'.
    Only touches the job if it belongs to this workspace.
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(
        "UPDATE saved_jobs SET status = ?, status_updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ? AND session_id = ?",
        (new_status, job_id, session_id),
    )

    connection.commit()
    connection.close()


def delete_saved_job(job_id, db_path=DB_PATH, session_id=DEFAULT_SESSION_ID):
    """Removes a saved job from this workspace entirely."""
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM saved_jobs WHERE id = ? AND session_id = ?", (job_id, session_id)
    )

    connection.commit()
    connection.close()


def log_search(summary, db_path=DB_PATH, session_id=DEFAULT_SESSION_ID):
    """
    Logs one real search to search_history - this is what powers the
    "Recent Searches" panel with actual history instead of a
    hardcoded/fake list.

    Args:
        summary: short human-readable string describing the search,
                 e.g. "Software Engineer · Bengaluru · Remote"
    """
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO search_history (summary, session_id) VALUES (?, ?)",
        (summary, session_id),
    )

    connection.commit()
    connection.close()


def get_recent_searches(limit=5, db_path=DB_PATH, session_id=DEFAULT_SESSION_ID):
    """
    Returns this workspace's most recent searches, most recent first, as
    a list of dicts with 'summary' and 'searched_at'.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute(
        "SELECT * FROM search_history WHERE session_id = ? "
        "ORDER BY searched_at DESC, id DESC LIMIT ?",
        (session_id, limit),
    )
    rows = cursor.fetchall()

    connection.close()
    return [dict(row) for row in rows]


def delete_recent_search(search_id, db_path=DB_PATH, session_id=DEFAULT_SESSION_ID):
    """Removes a single entry from this workspace's search_history (used
    by the Recent Searches panel's X/remove button)."""
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM search_history WHERE id = ? AND session_id = ?", (search_id, session_id)
    )

    connection.commit()
    connection.close()
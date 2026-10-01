# Project Plan – AI Job Search Agent

## Goal

Build an agent that takes a candidate's resume and returns a ranked, explained shortlist of real job listings that best match it – end to end – originally prototyped in a Google Colab notebook (`notebooks/`), now a Streamlit web app.

## Timeline

> Weeks 1–2 record the original Colab prototype. Tools have changed since (Claude → Gemini, pdfplumber → PyMuPDF/pypdf with OCR fallback, Colab → Streamlit); the current stack is described in the README.

### Week 1 – Core pipeline

- [x] Set up Colab notebook structure *(later replaced by the Streamlit app)*
- [x] Integrate live job source (RemoteOK API)
- [x] Resume PDF upload + text extraction *(pdfplumber at the time; now PyMuPDF/pypdf + OCR fallback)*
- [x] LLM-based resume-to-job matching: score + reason + missing skills per job *(first built with Claude; now Gemini)*
- [x] Ranked output table + PDF export
- [x] Push initial version to GitHub *(repository is being set up; the link will be added to the README)*

### Week 2 – Robustness & UX

- [x] Handle malformed/unreadable resumes and scanned/image-based resumes
- [x] Handle API failures and rate-limit errors gracefully
- [x] Validate Gemini responses and match scores
- [x] Add user-controlled job search location
- [x] Improve location handling and common location corrections
- [x] Improve job relevance filtering
- [x] Add duplicate job removal
- [x] Test with multiple resumes
- [x] Verify complete notebook execution from top to bottom

### Week 3 – Expand job sources

- [x] Add a second job source (Adzuna) for non-remote/location-based listings
- [x] Integrate a third job source for India-focused job listings (**JobsPipe**, not IndianAPI — no free public API was available for IndianAPI, so JobsPipe was substituted; see the README's "API Substitution Rationale")
- [x] Combine results from multiple job sources
- [x] De-duplicate listings across job sources
- [x] Add source information to job results
- [x] Compare match quality across different job sources

### Week 4 – Polish & extras

- [x] Resume improvement suggestions based on recurring missing skills
- [x] Simple web UI (Streamlit)
- [x] Final report and documentation cleanup *(README rewritten to match the app; see Week 6)*

### Week 5 – Explainable scoring, saved jobs, and the chat agent

- [x] Rebuilt matching as 8 independently-weighted categories (skill, experience, role, location/work-mode, salary, tech/domain fit, company preference, notice period) with dynamic renormalization when a category's data is missing, instead of one opaque Gemini score
- [x] Added a red-flag score cap for scam-like signals (e.g. upfront payment requests)
- [x] Explainable Job Match Card UI: score ring, per-category bars, matching/missing skill tags, skill-gap learning roadmap, risk-status banner
- [x] Saved Jobs tracker on SQLite (`jobs.db`) — save, update status (saved → applied → interview → rejected → offer), remove
- [x] Career Assistant Chat: a standalone LangChain agent (`langchain.agents.create_agent`) with `job_search_tool`, `company_research_tool`, `match_jobs_to_my_resume_tool` (when a resume is loaded), and `save_job_tool` / `list_saved_jobs_tool` / `update_application_status_tool`, with streaming responses
- [x] Added Google Search grounding to `company_research_tool`, with an automatic fallback to ungrounded Gemini and a label showing which path ran
- [x] Career Analytics helpers (`src/analytics.py`): KPI counts, status distribution, recurring missing skills, top-scoring titles, most-seen companies. The dashboard page exists in code but is intentionally **not shown** in the UI (only Find Jobs and Career Agent tabs are exposed)
- [x] ATS readiness check (deterministic, no Gemini calls) with optional job-description keyword match
- [x] Chat file attachments (PDF/TXT/MD) with per-file parse caching, conversation-history trimming and a queued-turn design that survives Streamlit reruns (`src/career_chat.py`)
- [x] Automated test suite grew to 160 tests, including mocked-network tests of every live API call's request-building and error handling

### Week 6 – Production hardening

- [x] Per-visitor workspaces: every saved job and search-history row is scoped to a private workspace id kept in the page URL, so visitors sharing one hosted SQLite file can't see or change each other's data (including via the chat agent's tools); older databases migrate automatically
- [x] Rate limiting (minimum gap per source) and retry with exponential backoff for transient API failures; permanent errors (401, JobsPipe 402) are never retried
- [x] `packages.txt` (`tesseract-ocr`) so scanned-PDF OCR works on Streamlit Cloud
- [x] `tests/test_career_chat.py` – chat plumbing was previously untested
- [x] Test suite grew from 160 to 283 tests (all 283 pass)
- [x] Every Gemini call goes through a retry-with-backoff wrapper with fallback models (`src/gemini_retry.py`, 6 tests); job scoring runs up to 5 Gemini calls at a time; the resume profile falls back to an offline parser if Gemini is unavailable
- [x] Fallback model list in `src/config.py` refreshed to models that exist for the API key (`gemini-3.1-flash-lite`, `gemini-3.5-flash`)
- [x] UI layout fixes: the Assistant Conversation keeps its full width from the first message; the Saved jobs and Recent searches boxes line up at equal height; the search status text no longer shows internal wording
- [x] Repo housekeeping: removed duplicate/misplaced files from `tests/` and renamed the retry test file so pytest collects it
- [x] README rewritten to match the app's actual current state (features, project tree, deployment steps, honest limitations)

## Current Status

The core AI Job Search Agent pipeline is working end to end, with an ATS readiness check, saved-job tracking (private per visitor), and a conversational career agent with file attachments alongside the main matching flow. All 283 automated tests pass. What remains is the GitHub push, deployment to Streamlit Cloud, and a real-world speed check of a full search on the live APIs.

Current workflow:

Resume (PDF, OCR fallback for scans) → Candidate Profile (Gemini with retry + fallback models; offline resume parser if Gemini is unavailable) → Questionnaire (location, work mode, notice period, CTC, preferred companies) → Job APIs (RemoteOK, Adzuna, JobsPipe) → Relevance Filtering → Duplicate Removal → Pre-Ranking → Gemini scoring (up to 5 jobs at a time, each with retry + fallback) → 8-category scoring (Gemini + deterministic Python rules) → Weighted Final Score (dynamic renormalization + red-flag cap at 30) → Explainable Match Cards → Save/Track Jobs (SQLite, per-visitor workspace) → PDF Export; alongside: ATS readiness check and the Career Agent chat

## Future Improvements

- [ ] Hosted database for permanent saved-job storage (SQLite on Streamlit Cloud resets when the app restarts)
- [ ] Optional password/login for workspaces (today a workspace is protected only by its unguessable URL id)
- [ ] Fetch the job sources concurrently to shorten search time (today RemoteOK, Adzuna and JobsPipe are queried one after another)
- [ ] Email notifications for highly relevant jobs
- [ ] Additional job sources

## Notes

This plan is a living document. Completed tasks are marked with [x], while remaining tasks are marked with [ ].
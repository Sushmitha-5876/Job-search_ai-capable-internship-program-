# 🔎 AI Job Search Agent

**Live app:** [https://ai-job-search-agent-pro.streamlit.app](https://ai-job-search-agent-pro.streamlit.app)
**GitHub repo:** [https://github.com/Sushmitha-5876/Job-search_ai-capable-internship-program-](https://github.com/Sushmitha-5876/Job-search_ai-capable-internship-program-)

An AI-powered career assistant that analyzes a candidate's resume, finds relevant live job listings, produces an **explainable, category-by-category match score** for each job, checks how well a resume would survive an ATS scan, tracks saved jobs in a SQLite database, and answers career questions through a conversational LangChain agent that can search jobs, research companies, match jobs to your resume and manage your saved applications.

The app has two tabs: **Find Jobs** (resume → scored matches, ATS check, PDF export) and **Career Agent** (chat, saved-job tracker, recent searches).

## What It Does

### 1. Resume-driven job matching (Find Jobs tab)
1. Accepts a candidate resume in PDF format (up to 10 MB) — text-based resumes are read directly, scanned/image-based ones fall back to OCR automatically.
2. Extracts the candidate's possible job roles, technical skills, experience, location, and education using Gemini. Every Gemini call retries automatically on temporary errors (such as 503 "high demand") and falls back to backup models; if Gemini is still unavailable, a built-in offline resume parser builds the profile so the search can continue.
3. Collects preferences through a questionnaire: preferred job-search location, work-mode preference, notice period/availability, expected annual CTC (range or exact figure, capped so a typo can't corrupt scoring), and preferred companies.
4. Fetches live job listings from **RemoteOK**, **Adzuna**, and **JobsPipe**, filters irrelevant listings, and removes duplicates.
5. Pre-ranks jobs with a cheap, deterministic Python score before spending Gemini calls on the best candidates. The shortlist is then scored by Gemini up to 5 jobs at a time (bounded concurrency), each job with its own retry/fallback.
6. Scores each shortlisted job across **8 independent categories**:

   | Category                | Weight | Computed by                                                                                        |
   | ----------------------- | ------ | -------------------------------------------------------------------------------------------------- |
   | Skill Match             | 30     | Gemini, cross-checked with fuzzy + synonym matching (e.g. "JS" ↔ "JavaScript")                     |
   | Experience Fit          | 20     | Python — parses years from the resume and the job description, compares to the required experience |
   | Role Relevance          | 15     | Gemini                                                                                             |
   | Location / Work Mode    | 10     | Python — fuzzy match against the candidate's stated preference                                     |
   | Salary Fit              | 10     | Python — compares job salary (when available) to expected CTC                                      |
   | Technology / Domain Fit | 10     | Gemini                                                                                             |
   | Company Preference      | 5      | Python — personalized to the candidate's own stated preferences, not a hardcoded company tier list |
   | Notice Period Fit       | 5      | Python — compares the candidate's availability to the job's stated joining requirement             |

7. Combines these into one final score using a **weighted average with dynamic renormalization**: a category with genuinely unavailable data (e.g. a job with no salary listed) is excluded from both the score and the weight total, rather than guessed at with a fake neutral value.
8. Applies a **red-flag score cap**: if Gemini detects scam-like signals (e.g. upfront payment requests), the final score is capped at 30 regardless of how well other categories matched.
9. Displays results as an **Explainable Job Match Card** per job: a score ring, a bar for each category (showing "Not available" rather than a fake number when data is missing), matching/missing skill tags with a **skill-gap learning roadmap** for missing ones, and a clear risk-status banner.
10. Lets you **save any job** with one click, and export the full breakdown as a formatted PDF report.
11. Aggregates missing skills across every job in the shortlist and surfaces **Resume Improvement Suggestions** — skills that recurred on multiple jobs, each paired with a free learning resource and a rough time estimate.

### 2. ATS readiness check (Find Jobs tab)
A deterministic, rule-based check (no Gemini calls, so the same resume always gets the same score) of whether a resume would survive an Applicant Tracking System scan: did the text extract cleanly without OCR, are standard section headings present, is contact info plain text, is the length reasonable, and — if you paste a job description — how well do the keywords overlap. It is deliberately honest about what it *can't* check (multi-column layouts, tables, text boxes).

### 3. Career Agent (LangChain)
A conversational assistant built with `langchain.agents.create_agent`, with streaming responses. It works without a resume, and becomes more personal once one is loaded. Its tools:

| Tool                             | What it does                                                                                                |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `job_search_tool`                | Searches Adzuna from a plain-text query (e.g. "python jobs in Pune")                                        |
| `company_research_tool`          | Researches a company using Gemini with **Google Search grounding** (falls back to plain Gemini and says so) |
| `match_jobs_to_my_resume_tool`   | *(when a resume is loaded)* runs the full 8-category scoring from chat                                      |
| `save_job_tool`                  | Saves a job to your tracker                                                                                 |
| `list_saved_jobs_tool`           | Lists your saved jobs and their statuses                                                                    |
| `update_application_status_tool` | Moves a job: saved → applied → interview → offer / rejected                                                 |

You can also attach a **PDF, TXT or Markdown file** (up to 10 MB) to a chat message — for example a job description or offer letter. Each file is parsed once (OCR fallback for scanned PDFs) and stays available for follow-up questions, with the conversation history trimmed automatically if it grows long.

### 4. Saved jobs tracker and recent searches (SQLite)
A local database (`jobs.db`, created automatically) stores saved jobs and search history. From the **Your saved jobs** panel you can change a job's status or remove it; the agent can do the same by chat.

**Per-visitor workspaces.** On a hosted deployment every visitor shares one database file, so every row is scoped to a private workspace id. The app puts a random id in the page URL (`?ws=…`); a refresh or a bookmarked link returns to the same workspace, while other visitors never see or modify your data — including through the chat agent, whose saved-job tools are bound to your workspace and cannot be pointed elsewhere by a prompt.

## Project Structure

```
job-search-ai-agent/
├── app.py                        # Streamlit entrypoint — Find Jobs + Career Agent tabs
├── conftest.py                   # pytest import-path setup
├── diag.py, check_gemini.py, diagnose_remoteok.py   # optional diagnostics (which Gemini models work, RemoteOK check)
├── src/
│   ├── config.py                 # API keys, Gemini client, model + fallback-model names
│   ├── gemini_retry.py           # Retry with backoff + fallback models for every Gemini call
│   ├── resume_parser.py          # PDF/OCR text extraction, candidate profile extraction
│   ├── job_fetcher.py            # RemoteOK / Adzuna / JobsPipe fetching, rate limiting + retry, filtering, dedup
│   ├── skill_matching.py         # Fuzzy + synonym skill matching (shared across the app)
│   ├── scorer.py                 # Pre-ranking + Gemini's 3-field scoring call
│   ├── experience_scoring.py     # Deterministic experience-gap scoring
│   ├── location_scoring.py       # Deterministic location/work-mode scoring
│   ├── salary_scoring.py         # Deterministic salary/CTC scoring (range or exact input)
│   ├── company_score.py          # Deterministic company-preference scoring
│   ├── notice_period_scoring.py  # Deterministic notice-period/availability scoring
│   ├── skill_gap_roadmap.py      # Per-job skill roadmap + cross-job Resume Improvement Suggestions
│   ├── final_scoring.py          # Combines all 8 categories into the final weighted score
│   ├── ats_scoring.py            # Deterministic ATS readiness check
│   ├── match_card.py             # Explainable Job Match Card + Save Job button
│   ├── pdf_export.py             # PDF report of the scored results
│   ├── database.py               # SQLite layer, scoped per visitor workspace
│   ├── career_agent.py           # LangChain agent, tools, agent cache, streaming
│   ├── career_chat.py            # Chat plumbing: attachments, history trimming, turn queue, error text
│   ├── analytics.py              # Aggregation helpers for saved-job/session insights (not shown in the UI)
│   ├── ui_helpers.py             # Results-table formatting helpers
│   └── data/
│       ├── companies.py          # Company list used by the questionnaire's multiselect
│       └── locations.py          # Indian city list used by the location picker
├── tests/                        # 283 automated pytest tests (see Testing)
├── notebooks/                    # Original notebook pipeline, kept for reference
├── docs/PLAN.md                  # Project plan and progress log
├── assets/                       # Theme CSS, logo, illustrations
├── .streamlit/
│   ├── config.toml               # Theme configuration
│   └── secrets.toml.example      # Example Streamlit secrets file for API keys
├── packages.txt                  # System packages for Streamlit Cloud (tesseract-ocr, for scanned-PDF OCR)
├── requirements.txt              # Production dependencies (what Streamlit Cloud installs)
├── requirements-dev.txt          # requirements.txt + pytest, for running the tests locally
└── jobs.db                       # SQLite database (created automatically, gitignored)
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Scanned-PDF resumes need the Tesseract OCR program** (text-based PDFs work without it). Install it locally — Windows: the UB Mannheim installer; macOS: `brew install tesseract`; Ubuntu/Debian: `sudo apt install tesseract-ocr`. On Streamlit Cloud this is handled by `packages.txt`.
3. Create a `.env` file in the project root for local development, or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:
   ```
   GEMINI_API_KEY=your_key_here
   ADZUNA_APP_ID=your_id_here
   ADZUNA_APP_KEY=your_key_here
   JOBSPIPE_API_KEY=your_jobspipe_key_here
   ```
   RemoteOK requires no API key. Never commit `.env` or `secrets.toml` (both are gitignored).
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

1. Push the repo to GitHub (make sure `.env`, `jobs.db` and any personal PDFs are **not** committed).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app from the repo with `app.py` as the main file.
3. In the app's **Settings → Secrets**, paste the four keys in the same format as `.streamlit/secrets.toml.example`.
4. Streamlit Cloud installs `requirements.txt` and, from `packages.txt`, the Tesseract OCR system package automatically.

One deployment behaviour to be aware of is listed under Known Limitations: SQLite storage is not permanent on Streamlit Cloud. Also note that workspace links are not password-protected — anyone who has your link can open your workspace, so don't share it.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

283 tests, none of which need an API key or network access:

| Area                                                     | Tests |
| -------------------------------------------------------- | ----- |
| Scoring: 8 categories, final score, skill-gap roadmap    | 66    |
| ATS readiness scoring                                    | 13    |
| SQLite layer (incl. per-workspace isolation, migration)  | 12    |
| Analytics helpers                                        | 12    |
| Job fetchers: aggregation + mocked-network request tests | 25    |
| Job fetchers: rate limiting and retry/backoff            | 10    |
| Career agent: tools, workspace binding, cache, streaming | 44    |
| Career chat: attachments, history trimming, turn queue   | 95    |
| Gemini retry / fallback models                           | 6     |

Network behaviour is tested by mocking `requests.get` / `requests.post`; see Known Limitations for what that does and doesn't prove.

## API Substitution Rationale

The assignment brief names Naukri.com, TimesJobs, Indeed and Glassdoor as reference job/company data sources, and suggests BeautifulSoup-based scraping. This project deliberately uses different sources:

- **Naukri.com, TimesJobs and Indeed have no public, free API**, and scraping them violates their Terms of Service and breaks whenever their layout changes or an IP is blocked.
- **RemoteOK, Adzuna and JobsPipe** provide legitimate, documented free-tier APIs. Adzuna and JobsPipe are queried against their India-specific endpoints (`/jobs/in/` and `job_country_code_or: ["IN"]`), so the India-focused requirement is still met — through compliant integrations rather than scraping.
- **No BeautifulSoup/scraping is used**: every job source is a JSON API called with `requests`.
- **Company research** uses Gemini with Google Search grounding rather than a structured company-data API, because Glassdoor has no accessible free API either.

**Responsible API usage.** Every job-source request goes through one function (`_send_request` in `job_fetcher.py`) that keeps a minimum gap between calls to the same source (RemoteOK 1.0 s, Adzuna 0.5 s, JobsPipe 1.0 s) and retries transient failures — timeouts, connection errors, HTTP 429 and 5xx — up to 2 times with exponential backoff (1.5 s, then 3 s), honouring a numeric `Retry-After` header (capped at 10 s). Permanent errors such as JobsPipe's 402 "quota exceeded" or a 401 are never retried, so quota isn't wasted.

## Known Limitations

- **RemoteOK is excluded from salary-fit scoring.** `calculate_salary_score` skips any job whose source isn't `adzuna` or `jobspipe`. RemoteOK is a global board whose salaries are typically in USD, while the salary logic assumes ₹ LPA; feeding a USD figure into that comparison would misread `$80,000` as `₹80,000` (about 66× too low) and produce a confidently wrong score instead of an honest "not available". RemoteOK also doesn't reliably indicate whether a listing is still open.
- **SQLite is not permanent storage on Streamlit Cloud.** The `jobs.db` file lives on the app container's disk and is reset when the app restarts or is redeployed, so saved jobs and search history can disappear. It is fully persistent when you run the app locally. A hosted database would be the fix for long-term persistence.
- **Mocked network tests are not live tests.** The fetcher tests prove that requests are built correctly and that errors are handled, without hitting the network. They cannot tell you a third-party API is still up or unchanged, and the Gemini calls (resume parsing, job scoring, company research) are not covered by network-level tests — a real end-to-end run with valid keys is still the way to confirm those.
- **Free-tier quotas apply.** Adzuna (~1,000 calls/month), JobsPipe (100 credits/month) and Gemini's free tier can run out; when they do, the app degrades gracefully (that source returns no jobs, or the chat shows a clear message) rather than crashing.
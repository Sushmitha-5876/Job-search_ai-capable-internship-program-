"""
diag_score.py - run ONE real job through the project's real scoring code
and show exactly what Gemini sends back, and where it breaks.

Run from the project folder:   python diag_score.py
It does not print your API key.
"""
import traceback

from src import scorer
from src.config import MODEL, get_gemini_client

RESUME = (
    "Sushmitha - 4th semester engineering student. Skills: Python, SQL, "
    "Streamlit, Pandas, Git. Projects: AI job search agent using Gemini, "
    "resume parser, data dashboards."
)
JOB = {
    "title": "Python Developer Intern",
    "company": "Example Corp",
    "location": "Bengaluru, India",
    "description": (
        "We need a Python developer with SQL and REST API knowledge. "
        "Docker and cloud basics are a plus. Work on data pipelines."
    ),
    "url": "https://example.com/job",
}

# Wrap the real retry call so we can look at the RAW reply.
real_call = scorer.generate_with_retry


def spy(client, model, prompt, **kw):
    print(f"\n[1] Calling Gemini: model={model}, fallbacks={kw.get('fallback_model')}")
    resp = real_call(client, model, prompt, **kw)
    print("[2] Call returned (no HTTP error).")
    try:
        for i, c in enumerate(resp.candidates or []):
            print(f"    candidate {i} finish_reason = {c.finish_reason}")
    except Exception as e:
        print("    (could not read finish_reason:", e, ")")
    try:
        fb = resp.prompt_feedback
        if fb:
            print("    prompt_feedback =", fb)
    except Exception:
        pass
    text = resp.text
    print("[3] response.text is None?", text is None)
    print("    RAW TEXT START >>>")
    print((text or "")[:1500])
    print("    <<< RAW TEXT END")
    return resp


scorer.generate_with_retry = spy

client = get_gemini_client()
scorer.LAST_GEMINI_ERROR["text"] = ""
try:
    result = scorer.score_job_with_gemini(JOB, RESUME, client, MODEL)
except Exception:
    print("\nEXCEPTION escaped score_job_with_gemini:")
    traceback.print_exc()
    result = None

print("\n================ RESULT ================")
if result:
    print("SCORING WORKED. Scores:",
          result["skill_match_score"], result["role_relevance_score"],
          result["technology_domain_score"], "| match_score:", result["match_score"])
    print("matching:", result["matching_skills"], "| missing:", result["missing_skills"])
else:
    print("SCORING FAILED.")
    print("Recorded error:", scorer.LAST_GEMINI_ERROR["text"])
    print("\nCopy everything printed above (without any keys) and send it back.")
"""
config.py
=========
Central place for API keys and the Gemini client.

API keys are stored in the project-root .env file and loaded once.
Other modules can import the keys they need from this file.

Required .env variables:

    GEMINI_API_KEY=your_key_here
    ADZUNA_APP_ID=your_id_here
    ADZUNA_APP_KEY=your_key_here
    JOBSPIPE_API_KEY=your_jobspipe_key_here

RemoteOK does not require an API key.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from google import genai


# Load variables from the project-root .env file
load_dotenv()


# ==========================================================
# API KEYS
# ==========================================================


def _get_secret_value(name):
    """Prefer Streamlit secrets, then fall back to local .env values."""
    try:
        return st.secrets.get(name, os.getenv(name))
    except Exception:
        return os.getenv(name)


GEMINI_API_KEY = _get_secret_value("GEMINI_API_KEY")

ADZUNA_APP_ID = _get_secret_value("ADZUNA_APP_ID")
ADZUNA_APP_KEY = _get_secret_value("ADZUNA_APP_KEY")

JOBSPIPE_API_KEY = _get_secret_value("JOBSPIPE_API_KEY")


# ==========================================================
# JOB ROLE OPTIONS
# ==========================================================

JOB_ROLE_OPTIONS = [
    "Software Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "Data Analyst",
    "Data Scientist",
    "Data Engineer",
    "Machine Learning Engineer",
    "AI Engineer",
    "DevOps Engineer",
    "Cloud Engineer",
    "Site Reliability Engineer",
    "Cybersecurity Analyst",
    "Security Engineer",
    "QA Engineer",
    "Test Automation Engineer",
    "Mobile App Developer",
    "Android Developer",
    "iOS Developer",
    "Blockchain Developer",
    "Database Administrator",
    "Systems Administrator",
    "Network Engineer",
    "Product Manager",
    "Project Manager",
    "Business Analyst",
    "UI/UX Designer",
    "Graphic Designer",
    "Digital Marketing Specialist",
    "SEO Specialist",
    "Content Writer",
    "HR Executive",
    "Recruiter",
    "Sales Executive",
    "Business Development Executive",
    "Customer Support Executive",
    "Financial Analyst",
    "Operations Manager",
]


# ==========================================================
# GEMINI MODEL
# ==========================================================

MODEL = "gemini-3.5-flash-lite"

# Used automatically when MODEL keeps returning 503 "high demand".
# Verify this name is enabled for your API key.
FALLBACK_MODEL = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]
# Models that don't exist for your key are skipped automatically.


# ==========================================================
# GEMINI CLIENT
# ==========================================================

def get_gemini_client():
    """
    Returns a ready-to-use Gemini client.

    Raises a clear error if the Gemini API key is missing.
    """

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Add it to a .env file in your project root."
        )

    return genai.Client(api_key=GEMINI_API_KEY)
"""
check_gemini.py  -  READ-ONLY diagnostic. Changes nothing in your project.

Put this file in the project ROOT (next to app.py) and run:

    python check_gemini.py

It answers three questions:
  1. Which project files is Python really loading?
  2. Does your API key work, and which models can it actually use?
  3. Is the 503 coming from ONE busy model or from ALL of them?
"""
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("1) FILES BEING LOADED")
print("=" * 60)
for name in ("src.gemini_retry", "src.resume_parser", "src.config"):
    try:
        mod = importlib.import_module(name)
        print(f"{name:20} -> {mod.__file__}")
        if name == "src.gemini_retry":
            print("   has generate_with_retry:", hasattr(mod, "generate_with_retry"))
        if name == "src.resume_parser":
            print("   has offline fallback   :", hasattr(mod, "_heuristic_profile"))
    except Exception as exc:
        print(f"{name:20} -> FAILED TO IMPORT: {type(exc).__name__}: {exc}")

try:
    from src.config import GEMINI_API_KEY, MODEL
    fb = getattr(importlib.import_module("src.config"), "FALLBACK_MODEL", None)
except Exception as exc:
    print("Cannot read config:", exc)
    sys.exit(1)

print("\nMODEL          =", MODEL)
print("FALLBACK_MODEL =", fb)
print("API key loaded =", bool(GEMINI_API_KEY),
      f"(ends with ...{GEMINI_API_KEY[-4:]})" if GEMINI_API_KEY else "")

from google import genai
from google.genai import types

client = genai.Client(
    api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=30000)
)

print("\n" + "=" * 60)
print("2) MODELS YOUR KEY CAN SEE (flash family)")
print("=" * 60)
try:
    names = [m.name.replace("models/", "") for m in client.models.list()]
    for n in sorted(n for n in names if "flash" in n):
        print("  ", n)
except Exception as exc:
    print("Could not list models:", type(exc).__name__, str(exc)[:300])

print("\n" + "=" * 60)
print("3) LIVE TEST OF EACH MODEL (3 tries each)")
print("=" * 60)
candidates = [MODEL]
if isinstance(fb, str):
    candidates.append(fb)
elif fb:
    candidates.extend(fb)
candidates += ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
seen = []
for name in candidates:
    if name in seen:
        continue
    seen.append(name)
    results = []
    for i in range(3):
        try:
            r = client.models.generate_content(model=name, contents="Reply with the word OK")
            results.append("OK")
            break
        except Exception as exc:
            text = str(exc)
            code = next((c for c in ("503", "429", "404", "403", "401", "400") if c in text[:40]), "ERR")
            results.append(code)
            time.sleep(2)
    print(f"{name:28} -> {' , '.join(results)}")

print("\nMeaning:  OK = works | 503 = Google overloaded | 429 = quota/rate limit")
print("          404 = model name not available for your key | 403/401 = key problem")
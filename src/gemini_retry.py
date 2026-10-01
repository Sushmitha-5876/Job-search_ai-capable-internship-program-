"""
gemini_retry.py
===============
One place for calling Gemini with retry + backoff + fallback models.

- Transient errors (503 / 429 / overloaded): retried with exponential backoff.
- Model-unavailable errors (404 / NOT_FOUND / not supported): that model is
  skipped immediately and the next model in the chain is tried.
- Fallback models may be given as ONE string or a LIST of strings.
- If every model fails, the last error is raised (never swallowed).
- Permanent errors that no other model can fix (400 bad request, 401/403 key
  problems) fail immediately.
"""

import threading
import time

TRANSIENT_MARKERS = (
    "503",
    "UNAVAILABLE",
    "429",
    "RESOURCE_EXHAUSTED",
    "high demand",
    "overloaded",
    "500",
    "INTERNAL",
    "DEADLINE_EXCEEDED",
    "timed out",
)

# Filled in by generate_with_retry so callers can report WHICH models were
# tried and how many attempts were made when everything failed.
class _PerThreadInfo:
    """Dict-like store with one copy per thread, so jobs scored at the same
    time do not overwrite each other's 'models tried' diagnostics."""

    def __init__(self):
        self._local = threading.local()

    def __getitem__(self, key):
        return getattr(self._local, key, [])

    def __setitem__(self, key, value):
        setattr(self._local, key, value)


LAST_CALL_INFO = _PerThreadInfo()

MODEL_MISSING_MARKERS = (
    "404",
    "NOT_FOUND",
    "not found",
    "is not supported",
    "no longer available",
)


def is_transient_error(exc):
    """True if the error looks temporary (worth retrying)."""
    text = str(exc).lower()
    return any(marker.lower() in text for marker in TRANSIENT_MARKERS)


def is_model_missing_error(exc):
    """True if this model name simply isn't available for the API key."""
    text = str(exc).lower()
    return any(marker.lower() in text for marker in MODEL_MISSING_MARKERS)


def _call(client, model, prompt, config):
    kwargs = {"model": model, "contents": prompt}
    if config is not None:
        kwargs["config"] = config
    return client.models.generate_content(**kwargs)


def _as_list(fallback_model):
    if not fallback_model:
        return []
    if isinstance(fallback_model, str):
        return [fallback_model]
    return [m for m in fallback_model if m]


def discover_text_models(client):
    """Names of Flash-family models this API key can actually see."""
    try:
        names = [m.name.replace("models/", "") for m in client.models.list()]
    except Exception:
        return []
    return sorted(n for n in names if "flash" in n)


def generate_with_retry(
    client,
    model,
    prompt,
    fallback_model=None,
    config=None,
    max_attempts=3,
    base_delay=2.0,
    sleep=time.sleep,
):
    """
    Call client.models.generate_content, walking through [model, *fallbacks].

    Returns the Gemini response, or raises the last error if all models fail.
    """
    chain = []
    for name in [model, *_as_list(fallback_model)]:
        if name and name not in chain:
            chain.append(name)

    last_error = None
    LAST_CALL_INFO["tried"] = []

    for name in chain:
        for attempt in range(max_attempts):
            _began = time.perf_counter()
            try:
                _response = _call(client, name, prompt, config)
                print(
                    f"[gemini] {name} attempt {attempt + 1}: OK in "
                    f"{time.perf_counter() - _began:.1f}s"
                )
                return _response
            except Exception as exc:
                last_error = exc
                print(
                    f"[gemini] {name} attempt {attempt + 1}: "
                    f"{type(exc).__name__} after "
                    f"{time.perf_counter() - _began:.1f}s - {str(exc)[:90]}"
                )
                LAST_CALL_INFO["tried"].append(
                    f"{name}#{attempt + 1}:{type(exc).__name__}"
                )
                if is_model_missing_error(exc) and not is_transient_error(exc):
                    break  # this model doesn't exist for the key -> next model
                if not is_transient_error(exc):
                    raise  # bad key / bad request: other models won't help
                if attempt < max_attempts - 1:
                    _delay = base_delay * (2 ** attempt)
                    print(f"[gemini] waiting {_delay:.0f}s before retrying {name}")
                    sleep(_delay)
        # transient errors exhausted on this model -> try the next one

    raise last_error
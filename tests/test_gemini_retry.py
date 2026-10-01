import pytest

from src.gemini_retry import generate_with_retry, is_transient_error


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs["model"])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


ERR_503 = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")


def test_succeeds_first_try():
    client = FakeClient(["ok"])
    assert generate_with_retry(client, "m", "p", sleep=lambda s: None) == "ok"
    assert client.models.calls == ["m"]


def test_retries_503_then_succeeds():
    client = FakeClient([ERR_503, ERR_503, "ok"])
    assert generate_with_retry(client, "m", "p", sleep=lambda s: None) == "ok"
    assert len(client.models.calls) == 3


def test_non_transient_error_fails_immediately():
    client = FakeClient([RuntimeError("400 INVALID_ARGUMENT bad request")])
    with pytest.raises(RuntimeError):
        generate_with_retry(client, "m", "p", sleep=lambda s: None)
    assert len(client.models.calls) == 1


def test_uses_fallback_model_after_exhausting_retries():
    client = FakeClient([ERR_503] * 4 + ["fallback ok"])
    result = generate_with_retry(
        client, "primary", "p", fallback_model="backup", sleep=lambda s: None
    )
    assert result == "fallback ok"
    assert client.models.calls[-1] == "backup"


def test_raises_original_error_when_everything_fails():
    client = FakeClient([ERR_503] * 4)
    with pytest.raises(RuntimeError, match="503"):
        generate_with_retry(client, "m", "p", sleep=lambda s: None)


def test_is_transient_error():
    assert is_transient_error(ERR_503)
    assert is_transient_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not is_transient_error(RuntimeError("403 PERMISSION_DENIED"))

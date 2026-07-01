"""Deploy/gate tests — build blocker + DoS bound + no-op output rail."""
import pathlib

from fastapi.testclient import TestClient

from ola_guardrails.app import create_app
from ola_guardrails.engine import GuardrailsEngine


def test_requirements_has_no_absolute_local_path():
    """A `pip freeze` can capture a shared package as `name @ file:///.../wheels/...whl` — an
    absolute host path that does not exist in the container, so `docker build` fails. It must not
    be pinned this way (the shared helper is installed from the vendored wheel in the Dockerfile)."""
    req = (pathlib.Path(__file__).resolve().parent.parent / "requirements.txt").read_text()
    assert "file://" not in req
    assert "@ file://" not in req


class _StubEngine(GuardrailsEngine):
    def check_input(self, text):
        return (True, "")

    def check_output(self, text):
        raise NotImplementedError("output self-check needs a dedicated check model")


def _app():
    app = create_app(engine=_StubEngine())
    app.dependency_overrides[app.state.caller_cn_dep] = lambda: "client.example.internal"
    return TestClient(app)


def test_input_length_is_bounded():
    """An unbounded prompt pins gpt2-large CPU + blocks the event loop; oversize must be rejected (422)."""
    r = _app().post("/check/input", json={"prompt": "x" * 100_000})
    assert r.status_code == 422


def test_output_rail_not_silently_allowing():
    """A not-yet-implemented output rail must NOT return allowed=true (no silent-pass); 501 instead."""
    r = _app().post("/check/output", json={"response": "anything"})
    assert r.status_code == 501

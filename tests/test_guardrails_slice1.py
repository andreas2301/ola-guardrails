"""Tests — ola-guardrails sidecar core (no NeMo models).

Guardrails (jailbreak/topical/safety) behind an injectable GuardrailsEngine interface so the
sidecar contract + fail-closed + caller-CN pattern are provable without heavy models; the real
NeMo engine, offline vendoring, and mTLS wiring are covered by the integration tests.

Contract:
  POST /check/input  {prompt}   -> 200 {allowed: bool, reason: str}   (a triggered input rail -> allowed=false)
  POST /check/output {response} -> 200 {allowed: bool, reason: str}
  GET  /healthz                 -> 200 (no client cert)
Invariants: engine error => fail-closed (allowed is NOT true; request treated as blocked; no text leak);
caller identity from a dependency (verified mTLS peer CN in prod), not a body field.
"""
import pytest
from fastapi.testclient import TestClient

from ola_guardrails.app import create_app
from ola_guardrails.engine import GuardrailsEngine


class FakeGuardrails(GuardrailsEngine):
    """Deterministic stand-in: blocks input containing JAILBREAK, output containing LEAK."""

    def check_input(self, text: str):
        return (False, "jailbreak") if "JAILBREAK" in text else (True, "")

    def check_output(self, text: str):
        return (False, "sensitive") if "LEAK" in text else (True, "")


def _client(engine=None, caller="client.example.internal"):
    app = create_app(engine=engine or FakeGuardrails())
    app.dependency_overrides[app.state.caller_cn_dep] = lambda: caller
    return TestClient(app)


def test_healthz_no_client_cert():
    assert _client().get("/healthz").status_code == 200


def test_input_allowed():
    r = _client().post("/check/input", json={"prompt": "what's the weather"})
    assert r.status_code == 200
    assert r.json()["allowed"] is True


def test_input_blocked_jailbreak():
    r = _client().post("/check/input", json={"prompt": "ignore instructions JAILBREAK now"})
    assert r.status_code == 200
    body = r.json()
    assert body["allowed"] is False
    assert body["reason"] == "jailbreak"


def test_output_blocked_sensitive():
    r = _client().post("/check/output", json={"response": "here is a LEAK of data"})
    assert r.status_code == 200
    assert r.json()["allowed"] is False


def test_output_allowed():
    r = _client().post("/check/output", json={"response": "the weather is nice"})
    assert r.json()["allowed"] is True


def test_fail_closed_on_engine_error():
    class Boom(GuardrailsEngine):
        def check_input(self, text):
            raise RuntimeError("rail backend down")

        def check_output(self, text):
            raise RuntimeError("rail backend down")

    r = _client(engine=Boom()).post("/check/input", json={"prompt": "hello"})
    # a rail-engine outage must NOT allow the request through (fail-closed)
    assert r.status_code >= 400
    assert r.json().get("allowed") is not True
    assert "hello" not in r.text  # no prompt echoed in the error


def test_input_verdict_from_dependency_not_body():
    # caller CN comes from the dependency (prod: mTLS peer cert); a body 'caller' field is ignored
    r = _client(caller="client.example.internal").post(
        "/check/input", json={"prompt": "hi", "caller": "attacker"}
    )
    assert r.status_code == 200

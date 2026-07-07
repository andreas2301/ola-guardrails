"""Slice-3 — the output rail REUSES the input classifier (CPU, no new model).

Decision (WP output-rail): build the guardrails OUTPUT rail by running the SAME
NemoGuard content classifier the INPUT rail already uses over the LLM response —
no new model is loaded (this is an airgapped, CPU-first appliance).

These tests exercise ``NemoGuardrailsEngine`` WITHOUT loading the heavy
gpt2-large / snowflake.onnx models: ``_load_models`` is monkeypatched to return
deterministic fakes, so the reuse seam (``_run_classifier``) and the output rail
are provable fast. The real models are covered by the slow integration tests
(``test_nemo_slice2``).
"""
import pytest

from ola_guardrails import engine as engine_mod
from ola_guardrails.engine import GuardrailsEngine, NemoGuardrailsEngine


class _FakeHeuristics:
    """Stand-in for the NeMo input-only jailbreak perplexity heuristics.

    Records how many times each heuristic ran so a test can prove the OUTPUT
    rail does NOT invoke them (they score adversarial PROMPT structure, not
    unsafe response content).
    """

    def __init__(self):
        self.lp_calls = 0
        self.ps_calls = 0

    def check_jailbreak_length_per_perplexity(self, text, threshold):
        self.lp_calls += 1
        return {"jailbreak": False}

    def check_jailbreak_prefix_suffix_perplexity(self, text, threshold):
        self.ps_calls += 1
        return {"jailbreak": False}


class _FakeClassifier:
    """Stand-in for the shared snowflake.onnx content classifier.

    Returns a high score (block) for any text containing ``UNSAFE``, a low score
    otherwise — mirroring the ``(classification, score)`` contract of the real
    ``JailbreakClassifier``.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return ("label", 0.9 if "UNSAFE" in text else 0.02)


@pytest.fixture
def engine(monkeypatch):
    heur = _FakeHeuristics()
    clf = _FakeClassifier()
    # Bypass the heavy gpt2-large / snowflake.onnx load; inject deterministic fakes.
    monkeypatch.setattr(engine_mod, "_load_models", lambda: (heur, clf))
    eng = NemoGuardrailsEngine()
    return eng, heur, clf


def test_output_rail_blocks_unsafe_response(engine):
    eng, _heur, clf = engine
    allowed, reason = eng.check_output("here is some UNSAFE content")
    assert allowed is False
    assert reason  # non-empty verdict reason (same (allowed, reason) shape as input)
    assert clf.calls == 1  # the SHARED classifier ran over the response


def test_output_rail_allows_clean_response(engine):
    eng, _heur, clf = engine
    allowed, reason = eng.check_output("the library opens at 9am on Saturday")
    assert allowed is True
    assert reason == ""
    assert clf.calls == 1


def test_output_rail_reuses_classifier_not_input_heuristics(engine):
    # The output rail REUSES the shared content classifier but must NOT run the
    # input-only jailbreak perplexity heuristics (those score PROMPT structure).
    eng, heur, clf = engine
    eng.check_output("any response")
    assert clf.calls == 1
    assert heur.lp_calls == 0
    assert heur.ps_calls == 0


def test_input_rail_still_runs_heuristics_and_shared_classifier(engine):
    # Regression guard for the refactor: the input rail keeps running BOTH the
    # perplexity heuristics AND the shared classifier the output rail reuses.
    eng, heur, clf = engine
    allowed, _ = eng.check_input("what's the weather")
    assert allowed is True
    assert heur.lp_calls == 1
    assert heur.ps_calls == 1
    assert clf.calls == 1


def test_input_and_output_share_the_same_classifier(engine):
    # Both rails funnel through the SAME classifier instance — the
    # "reuse, don't duplicate / no new model" invariant.
    eng, _heur, clf = engine
    eng.check_input("clean prompt")
    eng.check_output("clean response")
    assert clf.calls == 2  # one shared classifier instance served both rails


def test_output_conforms_to_engine_interface(engine):
    eng, _heur, _clf = engine
    assert isinstance(eng, GuardrailsEngine)
    # check_output no longer raises NotImplementedError
    allowed, reason = eng.check_output("fine")
    assert allowed is True
    assert reason == ""

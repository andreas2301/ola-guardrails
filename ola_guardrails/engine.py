"""Guardrails engine implementations.

Defines the abstract ``GuardrailsEngine`` interface and ``NemoGuardrailsEngine``,
which wires the real NeMo Guardrails jailbreak-detection heuristics (gpt2-large
perplexity + the NemoGuard ``snowflake.onnx`` classifier) behind that interface.
No LLM is used — the checks are CPU heuristics only.
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class GuardrailsEngine(ABC):
    """Injectable interface for a guardrails (input/output rail) engine.

    Tests use deterministic fakes; production wires the real NeMo Guardrails
    jailbreak-detection engine via :class:`NemoGuardrailsEngine`.
    """

    @abstractmethod
    def check_input(self, text: str) -> tuple[bool, str]:
        """Return (allowed, reason) for an input/prompt rail check."""
        ...

    @abstractmethod
    def check_output(self, text: str) -> tuple[bool, str]:
        """Return (allowed, reason) for an output/response rail check."""
        ...


# ---------------------------------------------------------------------------
# Slice-2: real NeMo jailbreak-detection heuristics (CPU, no LLM)
# ---------------------------------------------------------------------------

# The vendored HuggingFace cache lives at the project root:
#   ./vendored-models/hf-cache
# If it exists, point HF_HOME at it so transformers/optimum load offline.
_VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendored-models"
_VENDOR_HF_CACHE = _VENDOR_DIR / "hf-cache"
_VENDOR_CLASSIFIER_DIR = _VENDOR_DIR / "classifier"


def _maybe_use_vendored_cache() -> None:
    """When vendored artifacts exist, switch transformers into offline mode."""
    if _VENDOR_HF_CACHE.is_dir():
        os.environ.setdefault("HF_HOME", str(_VENDOR_HF_CACHE))
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if _VENDOR_CLASSIFIER_DIR.is_dir():
        os.environ.setdefault("EMBEDDING_CLASSIFIER_PATH", str(_VENDOR_CLASSIFIER_DIR))


class _PatchedSnowflakeEmbed:
    """Offline-safe Snowflake embedder used by the NemoGuard classifier.

    Mirrors the upstream ``SnowflakeEmbed`` class but forces
    ``safe_serialization=True`` so the vendored ``model.safetensors`` is used
    instead of a non-existent ``pytorch_model.bin``.
    """

    _MODEL_ID = "Snowflake/snowflake-arctic-embed-m-long"

    def __init__(self) -> None:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = os.environ.get("JAILBREAK_CHECK_DEVICE")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._MODEL_ID,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self._MODEL_ID,
            trust_remote_code=True,
            add_pooling_layer=False,
            safe_serialization=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, text: str):
        import numpy as np
        import torch

        tokens = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt", max_length=2048
        )
        tokens = tokens.to(self.device)
        embeddings = self.model(**tokens)[0][:, 0]
        return embeddings.detach().cpu().squeeze(0).numpy()


# Module-level singleton: loaded once, shared across engine instances, guarded
# by ``_model_lock`` so concurrent construction is safe.
_model_lock = threading.Lock()
_models: Optional[tuple[Any, Any]] = None


def _load_models() -> tuple[Any, Any]:
    """Load gpt2-large heuristics + NemoGuard snowflake.onnx classifier once."""
    global _models
    if _models is not None:
        return _models

    with _model_lock:
        if _models is not None:
            return _models

        # Force CPU for deterministic, sandbox-friendly behaviour.
        os.environ.setdefault("JAILBREAK_CHECK_DEVICE", "cpu")
        _maybe_use_vendored_cache()

        log.info("Loading NeMo jailbreak-detection heuristics (gpt2-large CPU)...")
        from nemoguardrails.library.jailbreak_detection.heuristics import checks as heuristics

        log.info("Loading NemoGuard snowflake.onnx classifier...")
        from nemoguardrails.library.jailbreak_detection.model_based.checks import initialize_model
        from nemoguardrails.library.jailbreak_detection.model_based.models import JailbreakClassifier
        from nemoguardrails.library.jailbreak_detection.model_based import models as _mb_models

        # The vendored Snowflake model only provides safetensors weights. The
        # remote modeling code defaults to ``safe_serialization=False`` and
        # looks for pytorch_model.bin, which does not exist offline. Patch the
        # embedder so it loads from ``model.safetensors`` instead.
        _mb_models.SnowflakeEmbed = _PatchedSnowflakeEmbed

        classifier_path = os.environ.get("EMBEDDING_CLASSIFIER_PATH")
        if classifier_path:
            classifier = JailbreakClassifier(str(Path(classifier_path) / "snowflake.onnx"))
        else:
            classifier = initialize_model()
            if classifier is None:
                raise RuntimeError(
                    "NemoGuard classifier could not be initialized. "
                    "Set EMBEDDING_CLASSIFIER_PATH to a directory containing snowflake.onnx."
                )

        _models = (heuristics, classifier)
        log.info("NeMo guardrails models loaded.")
        return _models


class NemoGuardrailsEngine(GuardrailsEngine):
    """Real jailbreak rail using NeMo Guard heuristics + snowflake.onnx.

    Models are loaded lazily on first construction and then reused by all
    instances. All checks run on CPU and require no external LLM.
    """

    _LP_THRESHOLD: float = 89.79
    _PS_PPL_THRESHOLD: float = 1845.65
    # The ONNX RF returns class-1 probability as its score. A low score on an
    # innocuous prompt (e.g. ~0.03) must not trigger a false positive, while a
    # confident jailbreak score (e.g. ~0.70) must be caught.
    _CLASSIFIER_THRESHOLD: float = 0.5

    def __init__(self) -> None:
        heuristics, classifier = _load_models()
        self._heuristics = heuristics
        self._classifier = classifier
        self._lock = threading.Lock()

    def _run_classifier(self, text: str) -> tuple[bool, str]:
        """Shared NemoGuard content-classifier check reused by BOTH rails.

        This is the "reuse the classifier on output" seam: the snowflake.onnx
        classifier scores *content* (not prompt structure), so the same verdict
        is valid whether ``text`` is an incoming prompt or an outgoing response.
        Both :meth:`check_input` and :meth:`check_output` funnel through here — no
        second model is loaded.

        The caller MUST already hold ``self._lock`` (it is a plain, non-reentrant
        ``threading.Lock``); this method does not acquire it.
        """
        _classification, score = self._classifier(text)
        if score > self._CLASSIFIER_THRESHOLD:
            return (False, "jailbreak")
        return (True, "")

    def check_input(self, text: str) -> tuple[bool, str]:
        """Run the input rail: input-only jailbreak heuristics + shared classifier.

        A positive signal from any detector blocks the input. The two perplexity
        heuristics detect adversarial PROMPT structure (gibberish length /
        perplexity, adversarial prefix/suffix), so they are input-only; the
        NemoGuard classifier (shared with the output rail) scores content.
        """
        with self._lock:
            lp = self._heuristics.check_jailbreak_length_per_perplexity(
                text, self._LP_THRESHOLD
            )
            if lp.get("jailbreak"):
                return (False, "jailbreak")

            ps = self._heuristics.check_jailbreak_prefix_suffix_perplexity(
                text, self._PS_PPL_THRESHOLD
            )
            if ps.get("jailbreak"):
                return (False, "jailbreak")

            return self._run_classifier(text)

    def check_output(self, text: str) -> tuple[bool, str]:
        """Run the output rail by REUSING the shared classifier over ``text``.

        The output rail runs the SAME NemoGuard content classifier the input rail
        uses — no dedicated output model is loaded (this appliance is airgapped
        and CPU-first). The input-only perplexity *jailbreak* heuristics are NOT
        applied to a response: they score adversarial prompt structure, not
        unsafe output content. Returns the same ``(allowed, reason)`` verdict
        shape as :meth:`check_input`.
        """
        with self._lock:
            return self._run_classifier(text)

FROM python:3.12-slim

# Disable all runtime network access to model hubs / telemetry.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    # Point transformers/optimum at the vendored HuggingFace cache.
    HF_HOME=/app/vendored-models/hf-cache \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8443

WORKDIR /app

# Install build tooling so any packages without wheels for Python 3.12 can be
# compiled inside the image (e.g. annoy, required by nemoguardrails).
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from exact pins.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install the shared ola-gateway-shared mTLS helpers from a vendored wheel so
# ola_gateway_shared.tls / .transport (new_ssl_context, peer_cn_allowed,
# PeerCertProtocol) are importable inside the image. The deployment supplies the
# exact built wheel; do not fetch it from a package index.
COPY wheels/ /app/wheels/
RUN pip install --no-cache-dir /app/wheels/ola_gateway_shared-*.whl

# Copy vendored models (gpt2-large, Snowflake embedding, snowflake.onnx) and checksums.
COPY vendored-models/ /app/vendored-models/
COPY models.sha256 /app/models.sha256

# Verify vendored model integrity before the build continues.
RUN cd /app && sha256sum -c models.sha256

# Copy application source.
COPY ola_guardrails/ /app/ola_guardrails/

# Build-stage smoke: ensure the package imports cleanly before committing the image.
RUN python -c 'import ola_guardrails.main'

# Run as a non-root user.
RUN useradd -m -u 1000 guardrails && chown -R guardrails:guardrails /app
USER guardrails

EXPOSE 8443

CMD ["python", "-m", "ola_guardrails.main"]

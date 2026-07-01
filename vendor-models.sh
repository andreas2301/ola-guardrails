#!/usr/bin/env bash
# Offline model vendoring script for ola-guardrails.
# Downloads the gpt2-large perplexity model, the Snowflake embedding model used
# by the NemoGuard snowflake.onnx classifier, and the classifier itself into
# ./vendored-models/ so the air-gapped Docker build can run without outbound
# network access.
set -euo pipefail

VENDOR_DIR="./vendored-models"
HF_CACHE_DIR="${VENDOR_DIR}/hf-cache/hub"
CLASSIFIER_DIR="${VENDOR_DIR}/classifier"
SHA_FILE="./models.sha256"

# Pin the exact HuggingFace model revisions so the vendored artifact is reproducible.
GPT2_REVISION="32b71b12589c2f8d625668d2335a01cac3249519"
SNOWFLAKE_REVISION="92d97331f1f4b6a366c1f161354b9f3390cc219f"
NEMOGUARD_REVISION="cc8b97e2bd6c1667c31476eedaa9a75b4d7ed282"

mkdir -p "${HF_CACHE_DIR}" "${CLASSIFIER_DIR}"

create_ref() {
    # Create a refs/main pointer so transformers ``from_pretrained(model_id)``
    # (which defaults to revision=main) resolves to the pinned snapshot.
    local repo_id="$1"
    local revision="$2"
    local normalized
    normalized=$(echo "${repo_id}" | tr '/' '-')
    mkdir -p "${HF_CACHE_DIR}/models--${normalized}/refs"
    printf '%s' "${revision}" > "${HF_CACHE_DIR}/models--${normalized}/refs/main"
}

# gpt2-large is used by the NeMo perplexity heuristics.
echo "Vendoring gpt2-large (rev ${GPT2_REVISION})..."
.venv/bin/huggingface-cli download \
    gpt2-large \
    --revision "${GPT2_REVISION}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --local-dir-use-symlinks False
create_ref "gpt2-large" "${GPT2_REVISION}"

# Snowflake embedding model is required by the NemoGuard snowflake.onnx classifier.
echo "Vendoring Snowflake/snowflake-arctic-embed-m-long (rev ${SNOWFLAKE_REVISION})..."
.venv/bin/huggingface-cli download \
    Snowflake/snowflake-arctic-embed-m-long \
    --revision "${SNOWFLAKE_REVISION}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --local-dir-use-symlinks False
create_ref "Snowflake/snowflake-arctic-embed-m-long" "${SNOWFLAKE_REVISION}"

# NemoGuard JailbreakDetect snowflake.onnx random-forest classifier.
echo "Vendoring nvidia/NemoGuard-JailbreakDetect/snowflake.onnx (rev ${NEMOGUARD_REVISION})..."
.venv/bin/huggingface-cli download \
    nvidia/NemoGuard-JailbreakDetect \
    --revision "${NEMOGUARD_REVISION}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --local-dir-use-symlinks False \
    --local-dir "${CLASSIFIER_DIR}" \
    --include "snowflake.onnx"

# huggingface-cli leaves a transient .cache directory under --local-dir.
rm -rf "${CLASSIFIER_DIR}/.cache"

# Record SHA-256 checksums of every vendored artifact.
echo "Writing ${SHA_FILE}..."
find "${VENDOR_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum > "${SHA_FILE}"

echo "Vendoring complete. Artifacts in ${VENDOR_DIR}, checksums in ${SHA_FILE}."

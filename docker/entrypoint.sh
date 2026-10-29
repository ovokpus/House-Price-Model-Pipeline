#!/bin/sh
# Train when this container has no model yet, then serve the API.
# FORCE_RETRAIN=1 refits even if artifacts/latest.json is already present.
set -eu

ARTIFACT_DIR="${ARTIFACT_DIR:-/app/artifacts}"
export ARTIFACT_DIR
PORT="${PORT:-8000}"

mkdir -p "${ARTIFACT_DIR}"

if [ "${FORCE_RETRAIN:-0}" = "1" ] || [ ! -f "${ARTIFACT_DIR}/latest.json" ]; then
    echo "Training house-price model into ${ARTIFACT_DIR}"
    python -m house_price.train --artifact-dir "${ARTIFACT_DIR}"
else
    echo "Using existing model pointer ${ARTIFACT_DIR}/latest.json"
fi

echo "Serving on 0.0.0.0:${PORT}"
exec python -m uvicorn house_price.api:app --host 0.0.0.0 --port "${PORT}"

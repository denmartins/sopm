#!/bin/sh
# Seed the artifacts volume on first boot, then hand over to the CMD.
set -e

ARTIFACT_DIR="${SOPM_ARTIFACT_DIR:-/app/artifacts}"
SEED_DB="/opt/sopm-seed/embeddings.sqlite"

mkdir -p "$ARTIFACT_DIR"

if [ ! -f "$ARTIFACT_DIR/embeddings.sqlite" ] && [ -f "$SEED_DB" ]; then
	echo "[entrypoint] Installing pre-computed embedding cache into $ARTIFACT_DIR"
	cp "$SEED_DB" "$ARTIFACT_DIR/embeddings.sqlite"
fi

exec "$@"

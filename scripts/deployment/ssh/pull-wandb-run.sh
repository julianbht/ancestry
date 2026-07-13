#!/bin/bash
# Pull a single W&B run directory from the running SSH pod (default: the
# latest run) to your local machine.
# Runs on your PC (needs kubectl configured) — no port-forward required.
#
# Why not podcopy.sh /app/wandb: the wandb folder is full of symlinks
# (wandb/latest-run, wandb/debug.log, each run's logs/debug-core.log -> the
# pod's /root/.cache/wandb/logs/...). Extracting symlinks on Windows fails
# with "Operation not permitted" without admin/Developer Mode. This script
# tars only the one run directory and dereferences symlinks (-h) on the pod
# side, so the local side only ever sees plain files.
#
# Usage: ./scripts/deployment/ssh/pull-wandb-run.sh [--cuda|--dev] [run-name] [local-dest]
#
#   run-name     e.g. run-20260624_205709-86f31ghf (default: latest-run)
#   local-dest   where to put it locally (default: ./podcopy)
set -e

case "$1" in
  --cuda) DEPLOY_NAME="dsw-ancestry-ssh-cuda"; shift ;;
  --dev)  DEPLOY_NAME="dsw-ancestry-ssh";      shift ;;
  *)      DEPLOY_NAME="dsw-ancestry-ssh-cuda" ;;  # default
esac

RUN_NAME="$1"
DEST="${2:-./podcopy}"

POD=$(kubectl get pods -l app="$DEPLOY_NAME" --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')
if [ -z "$POD" ]; then
  echo "No running pod found for app=$DEPLOY_NAME — is the deployment up?" >&2
  exit 1
fi

if [ -z "$RUN_NAME" ]; then
  RUN_NAME=$(kubectl exec "$POD" -- bash -c "cd /app/wandb && readlink -f latest-run | xargs basename")
fi

if [ -z "$RUN_NAME" ]; then
  echo "Could not resolve latest-run on $POD — pass a run-name explicitly." >&2
  exit 1
fi

echo "==> Pulling wandb/$RUN_NAME from $POD -> $DEST"
mkdir -p "$DEST"
# -h dereferences symlinks during archive creation (see header comment).
MSYS_NO_PATHCONV=1 kubectl exec "$POD" -- tar czf - -h -C /app/wandb "$RUN_NAME" | tar xzf - -C "$DEST"

echo "==> Done — $DEST/$RUN_NAME"

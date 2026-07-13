#!/bin/bash
# Pull a file or folder from the running SSH pod to your local machine.
# Runs on your PC (needs kubectl configured) — no port-forward required.
#
# Usage: ./scripts/deployment/ssh/pull.sh [--cuda|--dev] <pod-path> [local-dest]
#
#   <pod-path>   absolute, or relative to /app (the pod's working dir)
#   local-dest   where to put it locally (default: current directory)
#   --cuda|--dev which deployment to pull from (default: --cuda)
#
# Examples:
#   ./scripts/deployment/ssh/pull.sh data/debug/frame_crop ./debug
#   ./scripts/deployment/ssh/pull.sh --dev data/steps/frame_crop/x.jpg .
set -e

case "$1" in
  --cuda) DEPLOY_NAME="dsw-ancestry-ssh-cuda"; shift ;;
  --dev)  DEPLOY_NAME="dsw-ancestry-ssh";      shift ;;
  *)      DEPLOY_NAME="dsw-ancestry-ssh-cuda" ;;  # default
esac

SRC="$1"
DEST="${2:-.}"

if [ -z "$SRC" ]; then
  echo "Usage: $0 [--cuda|--dev] <pod-path> [local-dest]" >&2
  echo "  <pod-path> absolute, or relative to the pod's /app working dir" >&2
  exit 1
fi

# Resolve relative paths against the pod's /app working dir.
case "$SRC" in
  /*) REMOTE="$SRC" ;;
  *)  REMOTE="/app/$SRC" ;;
esac

POD=$(kubectl get pods -l app="$DEPLOY_NAME" --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')
if [ -z "$POD" ]; then
  echo "No running pod found for app=$DEPLOY_NAME — is the deployment up?" >&2
  exit 1
fi

echo "==> Copying $POD:$REMOTE -> $DEST"
mkdir -p "$DEST"
# Stream tar over `kubectl exec` rather than `kubectl cp`: cp's internal tar
# stream is flaky and often reports "unexpected EOF" even on success. Archiving
# from the parent dir keeps the extracted layout the same as `cp` would produce
# (DEST/<basename>) and avoids the "Removing leading /" warning.
PARENT=$(dirname "$REMOTE")
BASE=$(basename "$REMOTE")
MSYS_NO_PATHCONV=1 kubectl exec "$POD" -- tar czf - -C "$PARENT" "$BASE" | tar xzf - -C "$DEST"
echo "==> Done."

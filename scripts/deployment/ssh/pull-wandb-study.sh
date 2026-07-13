#!/bin/bash
# Pull every W&B run belonging to one hpopt study (i.e. one run-group: all
# trials + the emissions run) from the running SSH pod to your local machine.
# Runs on your PC (needs kubectl configured) — no port-forward required.
#
# /app/wandb on the pod accumulates runs from every study and ad-hoc dev run
# ever launched there, so a plain `podcopy.sh /app/wandb` pulls way more than
# one study and (on Windows) fails outright on the symlinks wandb sprinkles
# through that tree (latest-run, debug.log, each run's logs/debug-core.log).
# This script instead:
#   1. resolves the run-group (default: the group of the current latest-run)
#   2. greps every run's binary run-*.wandb file for that group string —
#      it's not in any plain-text file, only inside the binary — to find all
#      member run dirs
#   3. tars just those dirs, dereferencing symlinks (-h) so nothing requires
#      symlink privileges to extract on Windows
#
# Usage: ./scripts/deployment/ssh/pull-wandb-study.sh [--cuda|--dev] [run-group] [local-dest]
#
#   run-group    e.g. insightface-20260624-183336 (default: latest-run's group)
#   local-dest   where to put it locally (default: ./podcopy)
set -e

case "$1" in
  --cuda) DEPLOY_NAME="dsw-ancestry-ssh-cuda"; shift ;;
  --dev)  DEPLOY_NAME="dsw-ancestry-ssh";      shift ;;
  *)      DEPLOY_NAME="dsw-ancestry-ssh-cuda" ;;  # default
esac

GROUP="$1"
DEST="${2:-./podcopy}"

POD=$(kubectl get pods -l app="$DEPLOY_NAME" --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')
if [ -z "$POD" ]; then
  echo "No running pod found for app=$DEPLOY_NAME — is the deployment up?" >&2
  exit 1
fi

if [ -z "$GROUP" ]; then
  echo "==> No run-group given — resolving it from wandb/latest-run on $POD"
  GROUP=$(kubectl exec "$POD" -- bash -c '
    set -e
    cd /app/wandb
    RUN=$(readlink -f latest-run)
    METHOD=$(grep -oE "\"(deepface|insightface)\"" "$RUN/files/wandb-metadata.json" | head -1 | tr -d \")
    grep -aoE "${METHOD}-[0-9]{8}-[0-9]{6}" "$RUN"/run-*.wandb | sort -u | head -1
  ')
fi

if [ -z "$GROUP" ]; then
  echo "Could not resolve a run-group automatically — pass one explicitly." >&2
  exit 1
fi

echo "==> Study group: $GROUP"
echo "==> Finding member runs on $POD..."
RUN_DIRS=$(kubectl exec "$POD" -- bash -c "cd /app/wandb && grep -laE '$GROUP' run-*/run-*.wandb | cut -d/ -f1 | sort -u")
N=$(echo "$RUN_DIRS" | grep -c .)
if [ "$N" -eq 0 ]; then
  echo "No runs found for group '$GROUP' on $POD." >&2
  exit 1
fi
echo "==> Found $N run(s). Pulling -> $DEST"

mkdir -p "$DEST"
# -h dereferences symlinks during archive creation (see header comment).
echo "$RUN_DIRS" | MSYS_NO_PATHCONV=1 kubectl exec -i "$POD" -- bash -c \
  'cd /app/wandb && tar czf - -h -T -' | tar xzf - -C "$DEST"

echo "==> Done — $N run(s) in $DEST/"

#!/bin/bash
# Usage: ./scripts/deployment/jobs/run-step.sh <step> [--env <env>]
# Runs a single pipeline step as a k8s Job: deletes any previous run, applies
# the step's kustomize overlay, and follows logs.
#
# <step> must match a directory name under deployment/k8s/jobs/, e.g.:
#   ./scripts/deployment/jobs/run-step.sh frame-crop
#   ./scripts/deployment/jobs/run-step.sh face-crop --env dev
set -e
cd "$(dirname "$0")/../../.."

STEP=$1

if [[ -z "$STEP" || "$STEP" == --* ]]; then
  echo "Usage: $0 <step> [--env <env>]"
  echo "Available steps: $(ls deployment/k8s/jobs | tr '\n' ' ')"
  exit 1
fi
shift

KUSTOMIZE_DIR="deployment/k8s/jobs/$STEP"
if [[ ! -d "$KUSTOMIZE_DIR" ]]; then
  echo "No such step '$STEP' (no directory at $KUSTOMIZE_DIR)"
  echo "Available steps: $(ls deployment/k8s/jobs | tr '\n' ' ')"
  exit 1
fi

bash scripts/deployment/jobs/rerun-job.sh "dsw-ancestry-pipeline-job-$STEP" "$KUSTOMIZE_DIR" "$@"

#!/bin/bash
# Usage: ./scripts/deployment/jobs/rerun-job.sh <job-name> <kustomize-dir> [--env <env>]
# Deletes the job if it exists, applies the kustomize overlay, then follows logs.
# --env overrides the ENV baked into the image (e.g. --env dev to use step.dev.yaml configs).
set -e

JOB_NAME=$1
KUSTOMIZE_DIR=$2
ENV_OVERRIDE=""

shift 2
while [[ $# -gt 0 ]]; do
  case $1 in
    --env) ENV_OVERRIDE=$2; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$JOB_NAME" || -z "$KUSTOMIZE_DIR" ]]; then
  echo "Usage: $0 <job-name> <kustomize-dir> [--env <env>]"
  exit 1
fi

echo "==> Deleting job/$JOB_NAME (if exists)..."
kubectl delete job "$JOB_NAME" --ignore-not-found

echo "==> Applying $KUSTOMIZE_DIR${ENV_OVERRIDE:+ (ENV=$ENV_OVERRIDE)}..."
if [[ -d "$KUSTOMIZE_DIR" ]]; then
  if [[ -n "$ENV_OVERRIDE" ]]; then
    kubectl kustomize "$KUSTOMIZE_DIR" | python -c "
import sys, yaml
doc = yaml.safe_load(sys.stdin)
envs = doc['spec']['template']['spec']['containers'][0].setdefault('env', [])
envs.insert(0, {'name': 'ENV', 'value': sys.argv[1]})
print(yaml.dump(doc, default_flow_style=False))
" "$ENV_OVERRIDE" | kubectl apply -f -
  else
    kubectl apply -k "$KUSTOMIZE_DIR"
  fi
else
  kubectl apply -f "$KUSTOMIZE_DIR"
fi

echo "==> Waiting for pod to start..."
kubectl wait --for=condition=ready pod -l job-name="$JOB_NAME" --timeout=120s 2>/dev/null \
  || true  # jobs may complete before wait succeeds — fall through to logs

echo "==> Following logs..."
kubectl logs -f "job/$JOB_NAME"

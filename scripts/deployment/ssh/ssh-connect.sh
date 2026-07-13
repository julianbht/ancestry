#!/bin/bash
# Usage: ./scripts/deployment/ssh/ssh-connect.sh --cuda | --dev
set -e

case "$1" in
  --cuda)
    DEPLOY_NAME="dsw-ancestry-ssh-cuda"
    KUSTOMIZE_DIR="deployment/k8s/ssh/cuda"
    ;;
  --dev)
    DEPLOY_NAME="dsw-ancestry-ssh"
    KUSTOMIZE_DIR="deployment/k8s/ssh/dev"
    ;;
  *)
    echo "Usage: $0 --cuda | --dev"
    exit 1
    ;;
esac

cd "$(dirname "$0")/../../.."

echo "==> Applying $KUSTOMIZE_DIR..."
APPLY_OUT=$(kubectl apply -k "$KUSTOMIZE_DIR")
echo "$APPLY_OUT"

if echo "$APPLY_OUT" | grep -q "unchanged"; then
  echo "==> No YAML changes — restarting to pick up new image..."
  kubectl rollout restart deployment/"$DEPLOY_NAME"
fi

echo "==> Waiting for rollout..."
kubectl rollout status deployment/"$DEPLOY_NAME"

POD=$(kubectl get pods -l app="$DEPLOY_NAME" --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
echo "==> Pod: $POD"

echo "==> Waiting for sshd to be ready..."
until kubectl exec "$POD" -- pgrep sshd > /dev/null 2>&1; do
  printf '.'
  sleep 3
done
echo

echo "==> Port-forwarding localhost:2222 -> pod:22 (Ctrl+C to stop)"
kubectl port-forward pod/"$POD" 2222:22

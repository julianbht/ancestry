#!/bin/bash
# Rebuilds ancestry-pipeline:ssh-cuda on the cluster via kaniko.
set -e
cd "$(dirname "$0")/../../../.."
bash scripts/deployment/jobs/rerun-job.sh kaniko-build-cuda-dev deployment/k8s/kaniko-build-job.yml

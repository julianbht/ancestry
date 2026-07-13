#!/bin/bash
# Rebuilds ancestry-pipeline:job-cuda on the cluster via kaniko.
set -e
cd "$(dirname "$0")/../../../.."
bash scripts/deployment/jobs/rerun-job.sh kaniko-build-job-cuda deployment/k8s/kaniko-build-job-cuda.yml
